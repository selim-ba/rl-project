### Deep Q-Learning : Atari/Breakout
import ale_py #version 0.11.2  
import gymnasium as gym #version 1.2.1
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformReward

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# Variables
ENV_ID = "ALE/Breakout-v5"
SEED = 42

# Wrappers
class FireResetEnv(gym.Wrapper):
    """Press FIRE after reset so Breakout actually starts."""
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in meanings, "Env has no FIRE action."
        self.fire_action = meanings.index("FIRE")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Press FIRE once; if that incidentally ends the episode, reset again.
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)
    

class EpisodicLifeEnv(gym.Wrapper):
    """
    Training trick: treat loss of life as terminal for the learner,
    but only reset on true game over.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        # Prefer lives from info; fallback to ALE if needed
        lives = info.get("lives")
        if lives is None:
            try:
                lives = self.env.unwrapped.ale.lives()
            except Exception:
                lives = self.lives

        # Life lost but game not actually over: signal terminal to learner
        if (lives is not None) and (lives < self.lives) and (lives > 0):
            terminated = True

        self.lives = lives if lives is not None else self.lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # advance from life-terminal state with a NOOP
            obs, _, _, _, info = self.env.step(0)

        lives = info.get("lives")
        if lives is None:
            try:
                lives = self.env.unwrapped.ale.lives()
            except Exception:
                lives = self.lives
        self.lives = lives
        return obs, info
# ------------------------------------------------------- #

# Environment initialization
env = gym.make(id=ENV_ID,render_mode=None)
env.reset(seed=SEED)

possible_actions = env.unwrapped.get_action_meanings()
num_actions = len(possible_actions)
print(f"Environment : {ENV_ID}")
print(f"Possible actions for {ENV_ID} : {possible_actions}")
print(f"Nb. of actions for {ENV_ID} : {num_actions}")

env = FireResetEnv(env) # The game starts with FIRE
env = EpisodicLifeEnv(env) # end-of-life == end-of-episode
env = FrameStack(env,num_stack=4) #stack the last processed frames (H,W,C=4)
env = TransformReward(env,lambda r: np.float32(np.sign(r))) #optional but standard : reward clipping to {-1,0,+1}



# Preprocessing
env = AtariPreprocessing(
    env,
    noop_max = 30, # Table 1 - Nature paper
    screen_size = 84, # Resizing from 210x160 to 84x84
    grayscale_obs=True, # Extracts Y (luminance)
    scale_obs=False, # Values remains 0..255 (not 0..1)
    frame_skip=4, # Preprocessing applied to the 4 most recent frames
    terminal_on_life_loss=False, # life-termination already handled by EpisodicLifeEnv
)


# Q-Network (based on the Nature paper)
class QNetworkCNN(nn.Module):
    """
        CNN du Deep Q-Network décrit dans https://www.nature.com/articles/nature14236
        Input : preprocessed image of size 84x84
        Output : Q(s,a)
    """
    def __init__(self,input_channels: int, num_actions: int):
        super(QNetworkCNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1)
        
        self.fcn1 = nn.Linear(in_features=7*7*64,out_features=512) # 7*7*64 = 3136
        self.fcn2 = nn.Linear(in_features=512,out_features=num_actions)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x,start_dim=1) #start_dim = 1 conserve la dim du batch (x.shape = torch.Size([taille du batch, channel * h *w]))

        x = F.relu(self.fcn1(x))
        x = self.fcn2(x) # <-- q-values Q(s,a)
        return x

