# memory/rollout_buffer.py — storage for PPO rollouts with GAE
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class _Buf:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_shape: Tuple[int, int, int], device: str = "cpu", dtype=torch.uint8):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        n, e = self.n_steps, self.n_envs
        H, W, C = self.obs_shape
        self.storage = _Buf(
            obs=torch.zeros((n, e, H, W, C), dtype=self.dtype),
            actions=torch.zeros((n, e), dtype=torch.long),
            rewards=torch.zeros((n, e), dtype=torch.float32),
            dones=torch.zeros((n, e), dtype=torch.bool),
            values=torch.zeros((n, e), dtype=torch.float32),
            logprobs=torch.zeros((n, e), dtype=torch.float32),
            advantages=torch.zeros((n, e), dtype=torch.float32),
            returns=torch.zeros((n, e), dtype=torch.float32),
        )
        self._step = 0

    def add(self, obs, actions, rewards, dones, values, logprobs):
        i = self._step
        self.storage.obs[i].copy_(torch.as_tensor(obs, dtype=self.dtype))
        self.storage.actions[i].copy_(torch.as_tensor(actions, dtype=torch.long))
        self.storage.rewards[i].copy_(torch.as_tensor(rewards, dtype=torch.float32))
        self.storage.dones[i].copy_(torch.as_tensor(dones, dtype=torch.bool))
        self.storage.values[i].copy_(torch.as_tensor(values, dtype=torch.float32))
        self.storage.logprobs[i].copy_(torch.as_tensor(logprobs, dtype=torch.float32))
        self._step += 1

    def compute_returns_advantages(self, last_values, gamma: float, gae_lambda: float):
        n, e = self.n_steps, self.n_envs
        values = torch.cat([self.storage.values, torch.as_tensor(last_values, dtype=torch.float32).unsqueeze(0)], dim=0)
        adv = torch.zeros((n, e), dtype=torch.float32)
        last_gae = torch.zeros((e,), dtype=torch.float32)

        for t in reversed(range(n)):
            nonterminal = (~self.storage.dones[t]).float()
            delta = self.storage.rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
            adv[t] = last_gae

        self.storage.advantages.copy_(adv)
        self.storage.returns.copy_(adv + self.storage.values)

    def iterate_minibatches(self, batch_size: int):
        # flatten time & env
        obs = self.storage.obs.reshape(self.n_steps * self.n_envs, *self.obs_shape)
        actions = self.storage.actions.reshape(-1)
        logprobs = self.storage.logprobs.reshape(-1)
        advantages = self.storage.advantages.reshape(-1)
        returns = self.storage.returns.reshape(-1)

        idxs = torch.randperm(obs.size(0))
        for start in range(0, obs.size(0), batch_size):
            mb_idx = idxs[start:start + batch_size]
            yield {
                "obs": obs[mb_idx],
                "actions": actions[mb_idx],
                "logprobs": logprobs[mb_idx],
                "advantages": advantages[mb_idx],
                "returns": returns[mb_idx],
            }