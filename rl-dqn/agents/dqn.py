# agents/dqn.py

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.atari_cnn import QNetwork
from memory.replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    """Structured container of all DQN hyperparameters"""
    # Environment
    obs_shape: Tuple[int, int, int]  # (H,W,C) from env
    n_actions: int
    device: str = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
    )

    # Replay buffer
    replay_capacity: int = 1_000_000
    replay_warmup: int = 50_000
    batch_size: int = 32

    # Training
    gamma: float = 0.99
    lr: float = 2.5e-4
    rmsprop_alpha: float = 0.95
    rmsprop_eps: float = 0.01
    optimize_every: int = 4
    target_update_interval: int = 10_000
    huber_delta: float = 1.0
    max_grad_norm: Optional[float] = None

    # Exploration
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_anneal_frames: int = 250_000
    eval_epsilon: float = 0.05


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        
        # Create replay buffer (agent owns its memory)
        self.replay = ReplayBuffer(cfg.replay_capacity, cfg.obs_shape, cfg.device)

        # Determine input channels robustly
        if cfg.obs_shape[-1] in (1, 3, 4):  # HWC
            C = cfg.obs_shape[-1]
        else:  # CHW
            C = cfg.obs_shape[0]

        # Networks
        self.q = QNetwork(C, cfg.n_actions).to(cfg.device)
        self.q_tgt = QNetwork(C, cfg.n_actions).to(cfg.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        # Optimizer
        self.opt = torch.optim.RMSprop(
            self.q.parameters(),
            lr=cfg.lr,
            alpha=cfg.rmsprop_alpha,
            eps=cfg.rmsprop_eps,
            momentum=0.0,
            centered=False,
        )

        self.step_count = 0
        self.update_count = 0

    def _to_tensor(self, obs):
        """Convert observation to tensor and handle HWC → NCHW conversion
        
        Args:
            obs: numpy array or tensor, shape (H,W,C) or (B,H,W,C)
        
        Returns:
            tensor in (B,C,H,W) format
        """
        # Convert to tensor if needed
        if not isinstance(obs, torch.Tensor):
            x = torch.from_numpy(obs).to(self.cfg.device)
        else:
            x = obs
        
        # Add batch dimension if single observation
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (H,W,C) -> (1,H,W,C)
        
        # Convert HWC to NCHW if needed
        if x.shape[-1] in (1, 3, 4):  # Channel-last format detected
            x = x.permute(0, 3, 1, 2)
        
        return x

    @torch.no_grad()
    def act(self, obs, eval_mode: bool = False, epsilon: Optional[float] = None) -> int:
        x = self._to_tensor(obs)

        # 1) Choix d'epsilon : paramètre > cfg.eval_epsilon > schedule
        if epsilon is not None:
            eps = float(epsilon)
        elif eval_mode:
            eps = float(self.cfg.eval_epsilon)
        else:
            eps = float(self._epsilon())

        # 2) Epsilon-greedy (y compris en éval)
        if torch.rand(1).item() < eps:
            return torch.randint(0, self.cfg.n_actions, ()).item()

        if eval_mode:
            self.q.eval()
        q_values = self.q(x)
        return q_values.argmax(dim=1).item()


    def observe(self, s, a, r, ns, done) -> None:
        """Store transition in replay buffer"""
        self.replay.push(s, a, r, ns, done)
        self.step_count += 1

    def update(self) -> Dict[str, float]:
        """Perform one gradient update step"""
        # Guard conditions
        if self.step_count < self.cfg.replay_warmup:
            return {}
        if self.step_count % self.cfg.optimize_every != 0:
            return {}
        if not self.replay.can_sample(self.cfg.batch_size):
            return {}

        # Sample batch (returns HWC uint8)
        s, a, r, ns, d = self.replay.sample(self.cfg.batch_size, device=self.cfg.device)

        # Convert to NCHW
        s = self._to_tensor(s)
        ns = self._to_tensor(ns)

        # Q-learning target
        q = self.q(s)  # (B, |A|)
        q_a = q.gather(1, a.view(-1, 1)).squeeze(1)  # (B,)
        
        with torch.no_grad():
            qn = self.q_tgt(ns).max(dim=1).values
            target = r + (1.0 - d.float()) * self.cfg.gamma * qn

        # Huber loss
        loss = F.smooth_l1_loss(q_a, target, beta=self.cfg.huber_delta)

        # Optimize
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.max_grad_norm:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.cfg.target_update_interval == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return {
            "loss": float(loss.item()),
            "epsilon": self._epsilon(),
            "updates": self.update_count,
            "q_max": float(q.max(dim=1).values.mean().item()),
        }

    def _epsilon(self) -> float:
        t = min(self.step_count, self.cfg.eps_anneal_frames)
        return self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * (
            t / float(self.cfg.eps_anneal_frames)
        )


    def state_dict(self) -> dict:
        """Return agent state for checkpointing"""
        return {
            "q": self.q.state_dict(),
            "q_tgt": self.q_tgt.state_dict(),
            "opt": self.opt.state_dict(),
            "step_count": self.step_count,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """Load agent state from checkpoint"""
        self.q.load_state_dict(state["q"], strict=strict)
        self.q_tgt.load_state_dict(state["q_tgt"], strict=strict)
        self.opt.load_state_dict(state["opt"])
        self.step_count = state.get("step_count", 0)
        self.update_count = state.get("update_count", 0)

    def save(self, path: str) -> None:
        """Save agent checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            **self.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load agent from checkpoint"""
        payload = torch.load(path, map_location=self.cfg.device,weights_only=False)
        self.load_state_dict(payload, strict=strict)