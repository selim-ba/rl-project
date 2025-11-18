# env/wrappers.py
"""
Atari preprocessing wrappers following the Nature DQN paper.
Reference: https://www.nature.com/articles/nature14236
"""
import cv2
import numpy as np
import ale_py  # version 0.11.2
import gymnasium as gym  # version 1.2.1
from collections import deque


class NoopResetEnv(gym.Wrapper):
    """Apply random number of no-ops at episode start for stochasticity"""
    
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        noops = self.override_num_noops if self.override_num_noops is not None \
                else np.random.randint(1, self.noop_max + 1)
        
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self.has_fire = "FIRE" in meanings
        self.fire_action = meanings.index("FIRE") if self.has_fire else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.has_fire:
            return obs, info

        # FIRE 1 — passer état “serve”
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        # FIRE 2 — réellement lancer la balle
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action for 'skip' frames and return pixel-wise max over last 2 frames"""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)
        
        obs_space = env.observation_space
        assert hasattr(obs_space, "shape") and len(obs_space.shape) == 3
    
    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_info = {}
        last_obs = None
        
        self._obs_buffer.clear()
        
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += float(reward)
            terminated = terminated or term
            truncated = truncated or trunc
            
            if i >= self._skip - 2:
                self._obs_buffer.append(obs)
            
            last_obs = obs
            last_info = info
            
            if terminated or truncated:
                break
        
        # Pixel-wise max over last two frames
        if len(self._obs_buffer) == 2:
            obs = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            obs = self._obs_buffer[0] if self._obs_buffer else last_obs
        
        return obs, total_reward, terminated, truncated, last_info

class PongUpDownActionMap(gym.ActionWrapper):
    """
    Remap a 3-action discrete space {0:NOOP, 1:UP, 2:DOWN} to the
    corresponding ALE actions by name lookup.
    """
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        # find indices by name, robust to minimal/full sets
        def find(name):
            for i, m in enumerate(meanings):
                if name in m:  # e.g. 'UP', 'DOWN', 'NOOP'
                    return i
            raise RuntimeError(f"Action {name} not found in {meanings}")
        self._noop = find("NOOP")
        # Some ALE builds use 'UP'/'DOWN'; others use 'LEFT'/'RIGHT' for Pong paddles.
        # Try UP/DOWN first, else fallback to LEFT/RIGHT.
        try:
            self._up = find("UP")
            self._down = find("DOWN")
        except Exception:
            self._up = find("LEFT")
            self._down = find("RIGHT")

        self.action_space = gym.spaces.Discrete(3)

    def action(self, a):
        if a == 0: return self._noop
        if a == 1: return self._up
        return self._down



class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 grayscale as in Nature paper"""
    
    def __init__(self, env, width=84, height=84, grayscale=True, channel_first=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.channel_first = channel_first
        
        if grayscale:
            shape = (1, height, width) if channel_first else (height, width, 1)
        else:
            shape = (3, height, width) if channel_first else (height, width, 3)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
    
    def observation(self, obs):
        if self.grayscale:
            if obs.ndim == 3 and obs.shape[-1] == 3:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            
            obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
            
            if self.channel_first:
                obs = np.expand_dims(obs, 0)  # 1xHxW
            else:
                obs = np.expand_dims(obs, -1)  # HxWx1
        else:
            obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if self.channel_first:
                obs = np.moveaxis(obs, -1, 0)
        
        return obs.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack k last frames to give temporal information"""
    
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        
        shp = env.observation_space.shape
        channel_last = (shp[-1] in (1, 3))
        
        if channel_last:
            H, W, C = shp
            self.channel_first = False
            new_shape = (H, W, C * k)
        else:
            C, H, W = shp
            self.channel_first = True
            new_shape = (C * k, H, W)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        if self.channel_first:
            return np.concatenate(list(self.frames), axis=0)
        else:
            return np.concatenate(list(self.frames), axis=-1)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} as in Nature paper (training only)"""
    
    def reward(self, r):
        return np.sign(r).astype(np.int8)


def make_atari_env(
    env_id: str,
    training: bool = True,
    render_mode=None,
    noop_max: int = 30,
    skip: int = 4,
    grayscale: bool = True,
    width: int = 84,
    height: int = 84,
    channel_first: bool = False,
    seed: int = None,
    full_action_space: bool | None = None,
    sticky_action_prob: float | None = None,
):
    """
    Build an ALE Atari env with Nature DQN preprocessing.
    
    Wrapper order (matters!):
        NoopReset -> FireReset -> MaxAndSkip -> Warp -> FrameStack -> ClipReward (if training)
    
    Args:
        env_id: Atari environment ID (e.g., "ALE/Breakout-v5")
        training: If True, apply reward clipping
        render_mode: Rendering mode (None, "human", "rgb_array")
        noop_max: Maximum number of no-op actions at reset
        skip: Number of frames to repeat each action
        grayscale: Convert to grayscale
        width: Target width
        height: Target height
        channel_first: If True, return CHW format, else HWC
        seed: Random seed for environment
    
    Returns:
        Wrapped Gymnasium environment
    """
    make_kwargs = {"render_mode": render_mode}
    if full_action_space is not None:
        make_kwargs["full_action_space"] = full_action_space
    make_kwargs.setdefault("frameskip", 1)  # avoid double-skipping; we do our own skip=4

    if sticky_action_prob is not None:
        make_kwargs["repeat_action_probability"] = float(sticky_action_prob)

    env = gym.make(env_id, **make_kwargs)
    assert env.unwrapped.get_action_meanings()[0] == "NOOP", "NOOP must be index 0"
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    
    # Reset-time logic
    env = NoopResetEnv(env, noop_max=noop_max)
    env = FireResetEnv(env)

    if "Pong" in env_id:
        env = PongUpDownActionMap(env)

    
    # Per-step preprocessing
    env = MaxAndSkipEnv(env, skip=skip)
    env = WarpFrame(env, width=width, height=height, grayscale=grayscale, channel_first=channel_first)
    env = FrameStack(env, k=4)
    
    # Training-only reward clipping
    if training:
        env = ClipRewardEnv(env)
    
    return env