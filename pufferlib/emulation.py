from pdb import set_trace as T

import numpy as np
import warnings

import gymnasium
import inspect

import pufferlib
import pufferlib.spaces
from pufferlib import utils, exceptions


def dtype_from_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        dtype = []
        for i, elem in enumerate(space):
            dtype.append((f'f{i}', dtype_from_space(elem)))
    elif isinstance(space, pufferlib.spaces.Dict):
        dtype = []
        for k, value in space.items():
            dtype.append((k, dtype_from_space(value)))
    else:
        dtype = (space.dtype, space.shape)

    return np.dtype(dtype, align=True)

def flatten_space(space):
    if isinstance(space, pufferlib.spaces.Tuple):
        subspaces = []
        for e in space:
            subspaces.extend(flatten_space(e))
        return subspaces
    elif isinstance(space, pufferlib.spaces.Dict):
        subspaces = []
        for e in space.values():
            subspaces.extend(flatten_space(e))
        return subspaces
    else:
        return [space]

def emulate_observation_space(space):
    emulated_dtype = dtype_from_space(space)

    if isinstance(space, pufferlib.spaces.Box):
        return space, emulated_dtype

    leaves = flatten_space(space)
    dtypes = [e.dtype for e in leaves]
    if dtypes.count(dtypes[0]) == len(dtypes):
        dtype = dtypes[0]
    else:
        dtype = np.dtype(np.uint8)

    mmin, mmax = utils._get_dtype_bounds(dtype)
    numel = emulated_dtype.itemsize // dtype.itemsize
    emulated_space = gymnasium.spaces.Box(low=mmin, high=mmax, shape=(numel,), dtype=dtype)
    return emulated_space, emulated_dtype

def emulate_action_space(space):
    if isinstance(space, pufferlib.spaces.Discrete):
        return space, space.dtype

    emulated_dtype = dtype_from_space(space)
    leaves = flatten_space(space)
    emulated_space = gymnasium.spaces.MultiDiscrete([e.n for e in leaves])
    return emulated_space, emulated_dtype

def emulate(sample, sample_dtype, emulated_dtype):
    emulated = np.zeros(1, dtype=emulated_dtype)
    _emulate(emulated, sample)
    return emulated.view(sample_dtype).ravel()

def _emulate(arr, sample):
    if isinstance(sample, dict):
        for k, v in sample.items():
            _emulate(arr[k], v)
    elif isinstance(sample, tuple):
        for i, v in enumerate(sample):
            _emulate(arr[f'f{i}'], v)
    else:
        arr[()] = sample

def _nativize(sample, space):
    if isinstance(space, pufferlib.spaces.Tuple):
        return tuple(_nativize(sample[f'f{i}'], elem)
            for i, elem in enumerate(space))
    elif isinstance(space, pufferlib.spaces.Dict):
        return {k: _nativize(sample[k], value)
            for k, value in space.items()}
    else:
        return sample.item()

def nativize(sample, sample_space, emulated_dtype):
    sample = np.array(sample).view(emulated_dtype)
    return _nativize(sample, sample_space)

class GymnasiumPufferEnv(gymnasium.Env):
    def __init__(self, env=None, env_creator=None, env_args=[], env_kwargs={}):
        self.env = make_object(env, env_creator, env_args, env_kwargs)

        self.initialized = False
        self.done = True

        self.is_observation_checked = False
        self.is_action_checked = False

        self.observation_space, self.obs_dtype = emulate_observation_space(
            self.env.observation_space)
        self.action_space, self.atn_dtype = emulate_action_space(
            self.env.action_space)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = 1

        self.is_obs_emulated = self.single_observation_space is not self.env.observation_space
        self.is_atn_emulated = self.single_action_space is not self.env.action_space
        self.emulated = pufferlib.namespace(
            observation_dtype = self.observation_space.dtype,
            emulated_observation_dtype = self.obs_dtype,
        )

        self.mem = None
        self.obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self.render_modes = 'human rgb_array'.split()
        self.render_mode = 'rgb_array'

    def _emulate(self, ob):
        if self.is_obs_emulated:
            _emulate(self._obs, ob)
        elif self.mem is not None:
            self.obs[:] = ob
        else:
            self.obs = ob

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, seed=None):
        if not self.initialized:
            if self.mem is not None:
                self.obs = self.mem.obs[0]

            if self.is_obs_emulated:
                self._obs = self.obs.view(self.obs_dtype)

        self.initialized = True
        self.done = False

        ob, info = _seed_and_reset(self.env, seed)
        self._emulate(ob)

        if not self.is_observation_checked:
            self.is_observation_checked = check_space(
                self.obs, self.observation_space)

        return self.obs, info
 
    def step(self, action):
        '''Execute an action and return (observation, reward, done, info)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')
        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')

        # Unpack actions from multidiscrete into the original action space
        action = nativize(action, self.env.action_space, self.atn_dtype)

        if not self.is_action_checked:
            self.is_action_checked = check_space(
                action, self.action_space)

        ob, reward, done, truncated, info = self.env.step(action)
        self._emulate(ob)

        mem = self.mem
        if mem is not None:
            mem.rew[0] = reward
            mem.done[0] = done
            mem.trunc[0] = truncated
            mem.mask[0] = True
                   
        self.done = done

        return self.obs, reward, done, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

class PettingZooPufferEnv:
    def __init__(self, env=None, env_creator=None, env_args=[], env_kwargs={}, to_puffer=False):
        self.env = make_object(env, env_creator, env_args, env_kwargs)
        self.to_puffer = to_puffer
        self.initialized = False
        self.all_done = True

        self.is_observation_checked = False
        self.is_action_checked = False

        # Compute the observation and action spaces
        single_agent = self.possible_agents[0]
        single_observation_space = self.env.observation_space(single_agent)
        single_action_space = self.env.action_space(single_agent)
        self.single_observation_space, self.obs_dtype = (
            emulate_observation_space(single_observation_space))
        self.single_action_space, self.atn_dtype = (
            emulate_action_space(single_action_space))
        self.is_obs_emulated = self.single_observation_space is not single_observation_space
        self.is_atn_emulated = self.single_action_space is not single_action_space
        self.emulated = pufferlib.namespace(
            observation_dtype = self.single_observation_space.dtype,
            emulated_observation_dtype = self.obs_dtype,
        )

        self.num_agents = len(self.possible_agents)

        self.mem = None
        self.obs = np.zeros(self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype)

        #self.observations = np.zeros(self.num_agents, dtype=self.emulated.emulated_observation_dtype)
        #obs = self.observations.view(self.single_observation_space.dtype).reshape(self.num_agents, -1)

    @property
    def agents(self):
        return self.env.agents

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def done(self):
        return len(self.agents) == 0 or self.all_done

    def _emulate(self, ob, i, agent):
        if self.is_obs_emulated:
            _emulate(self._obs[i], ob)
        elif self.mem is not None:
            self.obs[i] = ob
        else:
            self.dict_obs[agent] = ob

    def observation_space(self, agent):
        '''Returns the observation space for a single agent'''
        if agent not in self.possible_agents:
            raise pufferlib.exceptions.InvalidAgentError(agent, self.possible_agents)

        return self.single_observation_space

    def action_space(self, agent):
        '''Returns the action space for a single agent'''
        if agent not in self.possible_agents:
            raise pufferlib.exceptions.InvalidAgentError(agent, self.possible_agents)

        return self.single_action_space

    def reset(self, seed=None):
        if not self.initialized:
            if self.mem is not None:
                self.obs = self.mem.obs

            if self.is_obs_emulated:
                self._obs = self.obs.view(self.obs_dtype).reshape(self.num_agents, -1)

            self.dict_obs = {agent: self.obs[i] for i, agent in enumerate(self.possible_agents)}

        self.initialized = True
        self.all_done = False
        self.mask = {k: False for k in self.possible_agents}

        obs, info = self.env.reset(seed=seed)

        # Call user featurizer and flatten the observations
        for i, agent in enumerate(self.possible_agents):
            if agent not in obs:
                self.observation[i] = 0
                continue

            ob = obs[agent]
            self._emulate(ob, i, agent)
            self.mask[agent] = True

        if not self.is_observation_checked:
            self.is_observation_checked = check_space(
                self.dict_obs[self.possible_agents[0]],
                self.single_observation_space
            )

        return self.dict_obs, info

    def step(self, actions):
        '''Step the environment and return (observations, rewards, dones, infos)'''
        if not self.initialized:
            raise exceptions.APIUsageError('step() called before reset()')
        if self.done:
            raise exceptions.APIUsageError('step() called after environment is done')

        if isinstance(actions, np.ndarray):
            actions = {agent: actions[i] for i, agent in enumerate(self.possible_agents)}

        # Postprocess actions and validate action spaces
        if not self.is_action_checked:
            self.is_action_checked = check_space(
                next(iter(actions.values())),
                self.single_action_space
            )

        # Unpack actions from multidiscrete into the original action space
        unpacked_actions = {}
        for agent, atn in actions.items():
            if agent not in self.possible_agents:
                raise exceptions.InvalidAgentError(agent, self.agents)

            if agent not in self.agents:
                continue

            if self.is_atn_emulated:
                atn = nativize(atn, self.single_action_space, self.atn_dtype)

            unpacked_actions[agent] = atn

        obs, rewards, dones, truncateds, infos = self.env.step(unpacked_actions)
        # TODO: Can add this assert once NMMO Horizon is ported to puffer
        # assert all(dones.values()) == (len(self.env.agents) == 0)
        self.mask = {k: False for k in self.possible_agents}
        for i, agent in enumerate(self.possible_agents):
            if agent not in obs:
                self.obs[i] = 0
                continue

            ob = obs[agent] 
            self.mask[agent] = True
            self._emulate(ob, i, agent)

            if self.mem is not None:
                self.mem.rew[i] = rewards[agent]
                self.mem.done[i] = dones[agent]
                self.mem.trunc[i] = truncateds[agent]
                self.mem.mask[i] = True
     
        self.all_done = all(dones.values())
        rewards = pad_agent_data(rewards, self.possible_agents, 0)
        dones = pad_agent_data(dones, self.possible_agents, False)
        truncateds = pad_agent_data(truncateds, self.possible_agents, False)

        return self.dict_obs, rewards, dones, truncateds, infos

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

def pad_agent_data(data, agents, pad_value):
    return {agent: data[agent] if agent in data else pad_value
        for agent in agents}
 
def make_object(object_instance=None, object_creator=None, creator_args=[], creator_kwargs={}):
    if (object_instance is None) == (object_creator is None):
        raise ValueError('Exactly one of object_instance or object_creator must be provided')

    if object_instance is not None:
        if callable(object_instance) or inspect.isclass(object_instance):
            raise TypeError('object_instance must be an instance, not a function or class')
        return object_instance

    if object_creator is not None:
        if not callable(object_creator):
            raise TypeError('object_creator must be a callable')
        
        if creator_args is None:
            creator_args = []

        if creator_kwargs is None:
            creator_kwargs = {}

        return object_creator(*creator_args, **creator_kwargs)

def check_space(data, space):
    try:
        contains = space.contains(data)
    except:
        raise exceptions.APIUsageError(
            f'Error checking space {space} with sample :\n{data}')

    if not contains:
        raise exceptions.APIUsageError(
            f'Data:\n{data}\n not in space:\n{space}')
    
    return True

def _seed_and_reset(env, seed):
    if seed is None:
        # Gym bug: does not reset env correctly
        # when seed is passed as explicit None
        return env.reset()

    try:
        obs, info = env.reset(seed=seed)
    except:
        try:
            env.seed(seed)
            obs, info = env.reset()
        except:
            obs, info = env.reset()
            warnings.warn('WARNING: Environment does not support seeding.', DeprecationWarning)

    return obs, info
