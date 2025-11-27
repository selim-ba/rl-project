# env/wrappers.py (last update by selim, 27/11/2025)

import cv2
import numpy as np
import ale_py
import gymnasium as gym
from collections import deque

gym.register_envs(ale_py)

class NoopResetEnv(gym.Wrapper):
    """Apply random number of no-ops at episode start for stochasticity."""
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

class FireResetEnv(gym.Wrapper):
    """Take FIRE actions on reset if required by the game."""
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        self.has_fire = "FIRE" in meanings
        self.fire_action = meanings.index("FIRE") if self.has_fire else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self.has_fire:
            return obs, info

        # FIRE twice to start the game (Pong/Breakout)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action and take max over last frames (Nature DQN)."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = None
        last_info = {}

        self._obs_buffer.clear()

        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += float(reward)
            terminated = terminated or term
            truncated = truncated or trunc

            last_obs = obs
            last_info = info

            # store last 2 frames for max-pooling
            if i >= self._skip - 2:
                self._obs_buffer.append(obs)

            if terminated or truncated:
                break

        # to ensure at least 1 frame exists even if episode ended immediately
        if len(self._obs_buffer) == 0:
            self._obs_buffer.append(last_obs)

        # pixel-wise max over last two frames
        if len(self._obs_buffer) == 2:
            obs = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            obs = self._obs_buffer[0]

        return obs, total_reward, terminated, truncated, last_info


class PongUpDownActionMap(gym.ActionWrapper):
    """Map actions {0: NOOP, 1: UP, 2: DOWN} to ALE actions."""
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()

        def find(name):
            for i, m in enumerate(meanings):
                if name in m:
                    return i
            raise RuntimeError(f"Action {name} not in {meanings}")

        self._noop = find("NOOP")
        try:
            self._up = find("UP")
            self._down = find("DOWN")
        except Exception:
            self._up = find("LEFT")
            self._down = find("RIGHT")

        self.action_space = gym.spaces.Discrete(3)

    def action(self, a):
        if a == 0:
            return self._noop
        if a == 1:
            return self._up
        return self._down


class WarpFrame(gym.ObservationWrapper):
    """Warp to 84x84 grayscale."""
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        shape = (height, width, 1)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

    def observation(self, obs):
        # obs comes as HxWx3 (RGB) from gymnasium Atari
        if obs.ndim == 3 and obs.shape[-1] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return obs[:, :, None].astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack k last frames (H,W,C*k), DeepMind-style."""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, c * k), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()

        # duplicate first frame k times
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, term, trunc, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1}."""
    def reward(self, r):
        return np.sign(r).astype(np.int8)

def make_atari_env(env_id, training=True, render_mode=None, seed=None):
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)

    if "Pong" in env_id:
        env = PongUpDownActionMap(env)

    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = FrameStack(env, k=4)

    if training:
        env = ClipRewardEnv(env)

    return env

