

import numpy as np
import cv2, gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt



import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
try:
    # selon la version
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers.frame_stack import FrameStack


import os, math, random, collections
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers.frame_stack import FrameStack


# =====================================================
# === ENVIRONNEMENT BREAKOUT ==========================
# =====================================================
def make_env(seed=0, render_mode=None, stack=4):
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


# =====================================================
# === RESEAU DQN ======================================
# =====================================================
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x_uint8):
        return self.net(x_uint8.float() / 255.0)


# =====================================================
# === REPLAY BUFFER ===================================
# =====================================================
Transition = collections.namedtuple("T", ["s","a","r","s2","d"])
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buf = [None]*capacity
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.device = device

    def push(self, s, a, r, s2, d):
        self.buf[self.pos] = Transition(s,a,r,s2,d)
        self.pos = (self.pos+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, n):
        batch = random.sample(self.buf[:self.size], n)
        s  = torch.tensor(np.stack([b.s  for b in batch]), dtype=torch.uint8, device=self.device)
        a  = torch.tensor([b.a for b in batch], dtype=torch.long, device=self.device)
        r  = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.stack([b.s2 for b in batch]), dtype=torch.uint8, device=self.device)
        d  = torch.tensor([b.d for b in batch], dtype=torch.float32, device=self.device)
        return s,a,r,s2,d

    def __len__(self): return self.size


# =====================================================
# === EPSILON-GREEDY ==================================
# =====================================================
class EpsSchedule:
    def __init__(self, start=0.9, end=0.05, decay=2_500):
        self.start, self.end, self.decay = start, end, decay
    def __call__(self, step):
        return self.end + (self.start - self.end) * math.exp(-step / self.decay)

@torch.no_grad()
def select_action(qnet, state_uint8, eps, action_space):
    if random.random() < eps:
        return action_space.sample()
    s = torch.tensor(state_uint8[None], dtype=torch.uint8, device=next(qnet.parameters()).device)
    qvals = qnet(s).squeeze(0)
    return int(torch.argmax(qvals).item())


# =====================================================
# === EVALUATION ======================================
# =====================================================
def evaluate(env_eval, qnet, episodes=5, eps_eval=0.05, max_steps=50_000):
    scores = []
    for _ in range(episodes):
        s, _ = env_eval.reset()
        s = np.array(s)
        ep_ret, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            a = select_action(qnet, s, eps_eval, env_eval.action_space)
            s2, r, term, trunc, _ = env_eval.step(a)
            s2 = np.array(s2)
            ep_ret += float(r)
            done = term or trunc
            s = s2; steps += 1
        scores.append(ep_ret)
    return float(np.mean(scores)), float(np.std(scores))


# =====================================================
# === ENTRAINEMENT PRINCIPAL ==========================
# =====================================================
def train(
    steps=300_000,
    eval_every=25_000,
    eval_episodes=5,
    replay_size=200_000,
    batch=32,
    start_learn=20_000,
    learn_every=4,
    gamma=0.99,
    lr=1e-4,
    target_sync=10_000,
    warmup=10_000,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=2_500,
    grad_clip=10.0,
    seed=0,
    cpu=False
):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    print("Device:", device)

    env = make_env(seed=seed)
    env_eval = make_env(seed=seed+1)
    nA = env.action_space.n

    qnet, tgt = DQN(nA).to(device), DQN(nA).to(device)
    tgt.load_state_dict(qnet.state_dict())
    tgt.eval()
    opt = optim.Adam(qnet.parameters(), lr=lr)
    rb = ReplayBuffer(replay_size, device)
    eps_fn = EpsSchedule(eps_start, eps_end, eps_decay)

    # ---- Warmup (collecte aléatoire) ----
    s,_ = env.reset(); s = np.array(s)
    for _ in trange(warmup, desc="Warmup"):
        a = env.action_space.sample()
        s2,r,term,trunc,_ = env.step(a)
        s2 = np.array(s2)
        rb.push(s,a,float(np.sign(r)),s2,float(term or trunc))
        s = s2 if not (term or trunc) else np.array(env.reset()[0])

    # ---- Entraînement principal ----
    pbar = trange(steps, dynamic_ncols=True)
    s,_ = env.reset(); s = np.array(s)
    ep_ret, best_eval = 0.0, -1e9

    for step in pbar:
        eps = eps_fn(step)
        a = select_action(qnet, s, eps, env.action_space)
        s2,r,term,trunc,_ = env.step(a)
        s2 = np.array(s2)
        rb.push(s,a,float(np.sign(r)),s2,float(term or trunc))
        s = s2
        ep_ret += r
        done = term or trunc
        if done:
            s,_ = env.reset(); s = np.array(s)
            pbar.set_description(f"EpRet={ep_ret:.1f} eps={eps:.3f}")
            ep_ret = 0.0

        if len(rb) >= start_learn and step % learn_every == 0:
            b_s,b_a,b_r,b_s2,b_d = rb.sample(batch)
            q_sa = qnet(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt_q = tgt(b_s2).max(1).values
                y = b_r + (1.0 - b_d) * gamma * tgt_q
            loss = nn.functional.smooth_l1_loss(q_sa, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), grad_clip)
            opt.step()

        if step % target_sync == 0:
            tgt.load_state_dict(qnet.state_dict())

        if step % eval_every == 0 and step > 0:
            mean_score, std = evaluate(env_eval, qnet, episodes=eval_episodes)
            pbar.write(f"[Eval] step={step:,} | score={mean_score:.2f} ± {std:.2f}")
            if mean_score > best_eval:
                best_eval = mean_score
                torch.save(qnet.state_dict(), "dqn_breakout_best.pt")

    env.close(); env_eval.close()
    print(f"Training done. Best eval score={best_eval:.2f}")


# =====================================================
# === LANCEMENT DIRECT (dans Jupyter) ================
# =====================================================
train(
    steps=300_000,        # plus long run
    start_learn=10_000,   # apprend plus tôt
    eps_decay=300_000,    # ε diminue plus vite
    eval_every=25_000,
    lr=1e-4,
)













