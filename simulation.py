import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers.frame_stack import FrameStack

# Même modèle DQN que pendant l'entraînement
class DQN(torch.nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1), torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64*7*7, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )
    def forward(self, x_uint8):
        return self.net(x_uint8.float() / 255.0)

# Recréation de l'environnement identique à celui du training
def make_env(seed=123, render_mode="human", stack=4):
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=False,
        frame_skip=4,
        screen_size=84,
        noop_max=30,
        terminal_on_life_loss=False,
    )
    env = FrameStack(env, num_stack=stack)
    env.reset(seed=seed)
    return env

# Sélection d'action (greedy)
@torch.no_grad()
def select_action(qnet, s_uint8, eps, action_space):
    if np.random.rand() < eps:
        return action_space.sample()
    s = torch.tensor(s_uint8[None], dtype=torch.uint8, device=next(qnet.parameters()).device)
    return int(torch.argmax(qnet(s)).item())

# Fonction pour rejouer l'agent
def play_agent(model_path="dqn_breakout_best.pt", episodes=1, eps=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(render_mode="human")  # affiche le jeu à l'écran
    nA = env.action_space.n
    qnet = DQN(nA).to(device)
    qnet.load_state_dict(torch.load(model_path, map_location=device))
    qnet.eval()

    for ep in range(episodes):
        s, _ = env.reset()
        s = np.array(s)
        total_r = 0
        done = False
        while not done:
            a = select_action(qnet, s, eps, env.action_space)
            s, r, term, trunc, _ = env.step(a)
            s = np.array(s)
            total_r += r
            done = term or trunc
        print(f"Épisode {ep+1} terminé : score total = {total_r}")
    env.close()

# Lance une partie (eps=0.05 => un peu d'exploration)
play_agent("dqn_breakout_best.pt", episodes=1, eps=0.05)
