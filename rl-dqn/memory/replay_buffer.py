# memory/replay_buffer.py
from __future__ import annotations
import numpy as np
import torch

class ReplayBuffer:
    """
       Stores past experiences (state, action, reward, next_state, done)

       Breaks correlations between consecutive frames
       Allows data reuse, and keeps a fixed-sized memory of recent experience
    """
    def __init__(self, capacity: int, obs_shape: tuple[int,int,int], device : str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.idx = 0 #write pointer
        self.full = False # has wrapped around at least once

        # Pre-allocate memory
        self.obs      = np.empty((self.capacity, *obs_shape), dtype=np.uint8)   # (N,H,W,C)
        self.next_obs = np.empty((self.capacity, *obs_shape), dtype=np.uint8)   # (N,H,W,C)
        self.actions  = np.empty((self.capacity,), dtype=np.int64)              # (N,)
        self.rewards  = np.empty((self.capacity,), dtype=np.float32)            # (N,)
        self.dones    = np.empty((self.capacity,), dtype=np.bool_)              # (N,)

    def push(self, s, a, r, s_next, done) -> None:
        self.obs[self.idx]      = s
        self.next_obs[self.idx] = s_next
        self.actions[self.idx]  = a
        self.rewards[self.idx]  = r
        self.dones[self.idx]    = done

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def can_sample(self, batch_size: int) -> bool:
        # to check if the buffer holds at least one full batch
        return len(self) >= batch_size

    def sample(self, batch_size: int, device: str | None = None):
        """
        Returns:
            s   : (B, H, W, C)  uint8  tensor (HWC)
            a   : (B,)          int64  tensor
            r   : (B,)          float32 tensor
            ns  : (B, H, W, C)  uint8  tensor (HWC)
            d   : (B,)          uint8/bool tensor (0 or 1)

        Right before the network forward, convert HWC -> NCHW:
              s  = s.permute(0, 3, 1, 2)
              ns = ns.permute(0, 3, 1, 2)
        """
        n = len(self)
        idxs = np.random.randint(0, n, size=batch_size) # randomly select batch_size indices uniformly

        dev = device or self.device
        s   = torch.from_numpy(self.obs[idxs]).to(dev, non_blocking=True)
        ns  = torch.from_numpy(self.next_obs[idxs]).to(dev, non_blocking=True)
        a   = torch.from_numpy(self.actions[idxs]).to(dev)
        r   = torch.from_numpy(self.rewards[idxs]).to(dev)
        d   = torch.from_numpy(self.dones[idxs].astype(np.uint8)).to(dev)  # keep as 0/1

        return s, a, r, ns, d