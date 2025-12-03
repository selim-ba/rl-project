# agents/ppo.py (last update 03/12/2025)

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PPOConfig:
    obs_shape: Tuple[int, int, int]  # (C, H, W)
    n_actions: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    device: str = "cpu"


class NatureCNN(nn.Module):
    """Nature DQN CNN architecture for Atari"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class ActorCritic(nn.Module):
    """Actor-Critic network with shared backbone (expects CHW)"""
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        C, H, W = obs_shape               # CHW
        self.backbone = NatureCNN(in_channels=C)
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        # x: (N, C, H, W) uint8 -> float32 in [0,1]
        x = x.float() / 255.0             # no permute!
        feat = self.backbone(x)
        logits = self.policy(feat)
        value = self.value(feat).squeeze(-1)
        return logits, value


class PPOAgent(nn.Module):
    """PPO Agent with standardized interface"""
    def __init__(self, cfg: PPOConfig):
        super().__init__()
        self.cfg = cfg
        self.net = ActorCritic(cfg.obs_shape, cfg.n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.to(cfg.device)

    # ------------- Acting -------------

    @torch.no_grad()
    def act(self,
            obs: np.ndarray | torch.Tensor,
            value_only: bool = False,
            deterministic: bool = False):
        """
        Standardized act() interface:
        - Training: return (actions, logprobs, values)
        - Evaluation (deterministic=True): return action (int or np.ndarray)
        - Value bootstrap (value_only=True): return (None, None, values)
        
        Args:
            obs: Observation(s) - single frame (HWC) or batch (NHWC)
            value_only: If True, only compute values (for GAE bootstrapping)
            deterministic: If True, use greedy actions (for evaluation)
        
        Returns:
            Training: (actions, logprobs, values) as numpy arrays
            Evaluation: action as int (single) or np.ndarray (batch)
            Value-only: (None, None, values)
        """
        # Convert to tensor on device, add batch dim if needed
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).to(self.cfg.device)
        else:
            obs_t = obs.to(self.cfg.device)
        
        added_batch = False
        if obs_t.ndim == 3:  # HWC single frame
            obs_t = obs_t.unsqueeze(0)
            added_batch = True

        logits, value = self.net(obs_t)

        if value_only:
            return None, None, value.cpu().numpy()

        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            # EVAL PATH: return JUST the action (not a tuple)
            act_t = torch.argmax(dist.probs, dim=-1)
            if added_batch:
                # Single env -> return Python int
                return int(act_t.squeeze(0).item())
            else:
                # Batched eval (rare) -> return numpy array
                return act_t.cpu().numpy()

        # TRAINING PATH: return (actions, logprobs, values)
        act_t = dist.sample()
        logprob = dist.log_prob(act_t)
        return (
            act_t.cpu().numpy(),
            logprob.cpu().numpy(),
            value.cpu().numpy(),
        )

    def evaluate_actions(self, obs_t: torch.Tensor, actions_t: torch.Tensor):
        """Evaluate actions for PPO update (returns logprobs, entropy, values)"""
        logits, values = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions_t)
        entropy = dist.entropy()
        return logprobs, entropy, values

    # ------------- Update -------------
    
    def update(self, rb, n_epochs: int, batch_size: int, clip_range: Optional[float] = None):
        """
        PPO update using rollout buffer
        
        Args:
            rb: RolloutBuffer with collected experience
            n_epochs: Number of epochs to train on the data
            batch_size: Minibatch size
            clip_range: Clipping parameter (if None, uses self.cfg.clip_range)
        
        Returns:
            Tuple of (approx_kl, clipfrac, pg_loss, v_loss, entropy)
        """
        cfg = self.cfg
        clip_range = clip_range if clip_range is not None else cfg.clip_range
        approx_kls, clipfracs, pg_losses, v_losses, entropies = [], [], [], [], []

        for _ in range(n_epochs):
            for batch in rb.iterate_minibatches(batch_size):
                obs_b = batch["obs"].to(cfg.device)
                actions_b = batch["actions"].to(cfg.device)
                old_logprobs_b = batch["logprobs"].to(cfg.device)
                adv_b = batch["advantages"].to(cfg.device)
                returns_b = batch["returns"].to(cfg.device)

                # Normalize advantages (critical for stability)
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                # Evaluate actions with current policy
                logprobs, entropy, values = self.evaluate_actions(obs_b, actions_b)
                entropy_mean = entropy.mean()

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(logprobs - old_logprobs_b)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (MSE)
                v_loss = F.mse_loss(values, returns_b)

                # Total loss
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_mean

                # Optimization step
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # Logging metrics
                with torch.no_grad():
                    approx_kl = (old_logprobs_b - logprobs).mean().abs()
                    clipfrac = (ratio.gt(1.0 + clip_range) | ratio.lt(1.0 - clip_range)).float().mean()

                approx_kls.append(approx_kl.item())
                clipfracs.append(clipfrac.item())
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy_mean.item())

        # Return average metrics
        return (
            float(np.mean(approx_kls) if approx_kls else 0.0),
            float(np.mean(clipfracs) if clipfracs else 0.0),
            float(np.mean(pg_losses) if pg_losses else 0.0),
            float(np.mean(v_losses) if v_losses else 0.0),
            float(np.mean(entropies) if entropies else 0.0),
        )

    # ------------- Checkpoint I/O -------------
    
    def save(self, path: str, extra: Dict[str, Any] | None = None):
        """Save agent checkpoint with metadata"""
        payload = {
            "net_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, device: str | None = None):
        """Load agent checkpoint and return metadata"""
        ckpt = torch.load(path, map_location=device or self.cfg.device, weights_only=False)
        
        # Handle both old and new checkpoint formats
        net_sd = ckpt.get("net_state_dict", ckpt.get("state_dict", ckpt))
        self.net.load_state_dict(net_sd)
        
        opt_sd = ckpt.get("optimizer_state_dict", ckpt.get("optimizer"))
        if opt_sd is not None:
            self.optimizer.load_state_dict(opt_sd)
        
        return ckpt
