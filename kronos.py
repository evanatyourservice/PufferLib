"""Kronos (Kron One-Sided) optimizer for PyTorch."""

from typing import Optional, Any
import numpy as np
import torch
from torch import Tensor


class ProbScheduler:
    """Scheduler for annealing preconditioner update probability.
    
    Implements an exponential anneal with a flat start.
    """
    
    def __init__(self, max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
        self.max_prob = torch.tensor(max_prob, dtype=torch.float32)
        self.min_prob = torch.tensor(min_prob, dtype=torch.float32)
        self.decay = torch.tensor(decay, dtype=torch.float32)
        self.flat_start = torch.tensor(flat_start, dtype=torch.float32)
        self._compiled = False
        try:
            self._compiled_schedule = torch.compile(self._schedule_fn)
            self._compiled = True
        except Exception:
            pass
    
    def _schedule_fn(self, n):
        """Exponential anneal with flat start."""
        prob = self.max_prob * torch.exp(-self.decay * (n - self.flat_start))
        prob.clamp_(min=self.min_prob, max=self.max_prob)
        return prob
    
    def __call__(self, n):
        """Call schedule function, using compiled version if available."""
        if self._compiled:
            return self._compiled_schedule(n)
        else:
            return self._schedule_fn(n)
    
    def __reduce__(self):
        """Enable proper pickling by serializing only the parameters."""
        return (self.__class__, (
            self.max_prob.item(),
            self.min_prob.item(),
            self.decay.item(),
            self.flat_start.item()
        ))


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """
    return ProbScheduler(max_prob, min_prob, decay, flat_start)


class OneSidedKron(torch.optim.Optimizer):
    """PSGD Kronos (Kron One-Sided) optimizer.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        b1: Momentum
        weight_decay: Weight decay
        preconditioner_update_probability: Prob of updating preconditioner (default: anneals 1.0->0.03 by 4000 steps)
        reset_precond_every_n: If set, reinitialize preconditioner every N steps
        precond_lr: Preconditioner learning rate (default: 0.1)
        clip_update_rms: Clip update RMS at 1.1
        merge_dims: Whether to combine dims to make grad tensor a matrix
        dtype: Data type for params/grads
    """

    def __init__(
        self,
        params: list[torch.Tensor | dict[str, Any]],
        lr: float = 0.0003,
        b1: float = 0.9,
        weight_decay: float = 0.0,
        preconditioner_update_probability: Optional[ProbScheduler] = None,
        reset_precond_every_n: Optional[int] = None,
        precond_lr: float = 0.1,
        clip_update_rms: bool = True,
        merge_dims: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        params = [*params]
        kron_param_groups = []
        adam_param_groups = []
        if isinstance(params[0], dict):
            for group in params:
                kron_params = []
                adam_params = []
                for p in group["params"]:
                    if p.ndim < 2 or max(p.shape) == np.prod(p.shape):
                        adam_params.append(p)
                    else:
                        kron_params.append(p)
                
                if kron_params:
                    kron_param_groups.append({
                        "params": kron_params,
                        **{k: v for k, v in group.items() if k != "params"}
                    })
                if adam_params:
                    adam_param_groups.append({
                        "params": adam_params,
                        **{k: v for k, v in group.items() if k != "params"}
                    })
        else:
            kron_params = []
            adam_params = []
            for p in params:
                if p.ndim < 2 or max(p.shape) == np.prod(p.shape):
                    adam_params.append(p)
                else:
                    kron_params.append(p)
            
            if kron_params:
                kron_param_groups.append({"params": kron_params})
            if adam_params:
                adam_param_groups.append({"params": adam_params})

        if adam_param_groups:
            self._adam = torch.optim.Adam(adam_param_groups, lr=lr * 3.0, fused=True)
        else:
            self._adam = None

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            reset_precond_every_n=reset_precond_every_n,
            precond_lr=precond_lr,
            clip_update_rms=clip_update_rms,
            merge_dims=merge_dims,
            dtype=dtype,
        )
        
        super().__init__(kron_param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device="cuda")
        self._prob_step = torch.tensor(0, dtype=torch.int32)
        self._update_counter = torch.tensor(0, dtype=torch.int32)
        self.dtype = dtype

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        reset_every_n = self.defaults["reset_precond_every_n"]
        if reset_every_n is None:
            update_prob = self.defaults["preconditioner_update_probability"]
            if callable(update_prob):
                update_prob = update_prob(self._prob_step.to(dtype=torch.float32))
                self._prob_step += 1
            self._update_counter += 1
            do_update = self._update_counter >= 1 / update_prob
            if do_update:
                self._update_counter = torch.tensor(0, dtype=torch.int32)
        else:
            # in this case, we update preconditioner every step and reinitialize it every n steps
            do_update = True

        update_energy = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.to(self.dtype)
                state = self.state[p]

                if g.dim() > 2 and group.get('merge_dims', True):
                    if "merged_shape" not in state:
                        shape1 = [np.prod(g.shape[:-1]), g.shape[-1]]
                        shape2 = [g.shape[0], np.prod(g.shape[1:])]
                        shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                        state["merged_shape"] = shape
                    g = g.view(*state["merged_shape"])

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["Q"] = torch.eye(min(g.shape), dtype=self.dtype, device=g.device)
                    state["step"] = torch.tensor(0, dtype=torch.int32, device="cuda")
                state["step"] += 1

                # possibly reset preconditioner
                if reset_every_n is not None and state["step"] % reset_every_n == 0:
                    state["Q"] = torch.eye(min(g.shape), dtype=self.dtype, device=g.device)

                g = _update_momentum(
                    state["momentum_buffer"],
                    g,
                    torch.tensor(group["b1"], dtype=self.dtype, device="cuda"),
                    state["step"]
                )

                # update preconditioner
                if do_update:
                    state["Q"] = _oneside_precond_update(
                        g,
                        state["Q"],
                        torch.tensor(group["precond_lr"], dtype=self.dtype, device="cuda"),
                    )
                
                # precondition gradient
                g = _oneside_precond_g(g, state["Q"])

                update_energy.append(g.square().mean())

                # normalize to RMS 1.0
                g /= g.square().mean().sqrt()
                # soft cap
                cap_amount = 2.0
                g.div_(cap_amount).tanh_().mul_(cap_amount)
                
                # weight decay
                g = g.view(p.shape)
                if group["weight_decay"] > 0:
                    g.add_(p, alpha=group["weight_decay"])
                
                # update parameters
                p.add_(g.to(p.dtype), alpha=-group["lr"])

        # adam for 1d params
        if self._adam is not None:
            self._adam.step()
        
        update_energy = torch.stack(update_energy).mean().sqrt().item()

        return loss, update_energy

    def state_dict(self):
        """Return the state of the optimizer as a dict."""
        state_dict = super().state_dict()
        if self._adam is not None:
            state_dict['adam_state'] = self._adam.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the optimizer state."""
        if 'adam_state' in state_dict:
            adam_state = state_dict.pop('adam_state')
            if self._adam is not None:
                self._adam.load_state_dict(adam_state)
        super().load_state_dict(state_dict)


@torch.compile
def _update_momentum(momentum_buffer, grad, beta, step):
    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
    return momentum_buffer.div(1 - beta**step)


def _lb(A: Tensor, max_abs: Tensor):
    """Cheap lower bound for the spectral norm of A."""
    A /= max_abs
    a0 = torch.einsum("ij,ij->j", A, A)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("j,kj->k", x, A)
    x = x.norm()
    x *= max_abs
    return x


@torch.compile
def _oneside_precond_update(G: Tensor, Q: Tensor, lr: Tensor):
    m, n = G.shape
    if m < n:
        G = G.T
    V = torch.randn_like(G, dtype=torch.float32)
    # damping
    eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=G.dtype, device=G.device).sqrt()
    G += eps * G.abs().mean() * V.to(dtype=G.dtype)
    # roughly same complexity as a matmul
    Bh = torch.linalg.solve_triangular(Q.float(), V, upper=True, left=False).to(dtype=G.dtype)
    BBh = Bh.T @ Bh
    A = G @ Q.T
    AhA = A.T @ A
    A = AhA + BBh
    max_abs = A.norm(float("inf"))
    Q = Q - lr / torch.where(max_abs > 0, _lb(A, max_abs), max_abs) * torch.triu(AhA - BBh) @ Q
    return Q


@torch.compile
def _oneside_precond_g(G: Tensor, Q: Tensor):
    m, n = G.shape
    if m < n:
        return torch.einsum("ji,jk,kl->il", Q, Q, G)
    else:
        return torch.einsum("ij,kj,kl->il", G, Q, Q)


@torch.compile
def _clip_update_rms(g):
    g.mul_(
        torch.minimum(
            torch.tensor(1.0, dtype=g.dtype, device=g.device),
            1.1 / g.square().mean().sqrt().add(1e-12),
        )
    )
