### Deep Q-Learning : Atari/Breakout
import ale_py #version 0.11.2  
import gymnasium as gym #version 1.2.1
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation as FrameStack
from gymnasium.wrappers import RecordVideo

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import time

PLOT_DIR = "dqn_plot_bis"
os.makedirs(PLOT_DIR, exist_ok=True)

VIDEO_DIR = "dqn_videos_bis"
os.makedirs(VIDEO_DIR, exist_ok=True)
VIDEO_RUN_ID = time.strftime("%Y%m%d-%H%M%S")
eval_capture_idx = 0

CHECKPOINT_DIR = "dqn_checkpoint_bis"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Variables
ENV_ID = "ALE/Breakout-v5"
SEED = 42

gamma = 0.99
batch_size = 32
buffer_capacity = 250_000 # instead of 1M          
train_freq = 4                     # optimize every 4 env steps
target_update_freq = 10_000        # in OPTIMIZER STEPS (not env steps)
max_frames = 1_000_000 #instead of 10M
learning_starts = 50_000 #instead of 50k

# -- logging containers ----
ep_frames, ep_returns = [], []      # training episode returns
eval_frames, eval_scores = [], []   # evaluation scores over time (mean)
loss_steps, loss_values = [], []    # TD loss per optimizer step
eps_frames, eps_values = [], []     # epsilon schedule
buf_frames, buf_sizes = [], []      # replay buffer fill

def moving_average(arr, window):
    if window <= 1 or len(arr) == 0:
        return arr
    out = []
    s = 0.0
    q = deque()
    for x in arr:
        q.append(x); s += x
        if len(q) > window:
            s -= q.popleft()
        out.append(s / len(q))
    return out

np.random.seed(SEED) #sets seed for numpy's random number generator
random.seed(SEED) #sets seed for python's built-in random module
torch.manual_seed(SEED) #sets seed for python's RNG (affects weight init, dropout (if any), and any random tensor generation in the model or training loop)
torch.backends.cudnn.deterministic = True # to use deterministic algorithms, it makes GPU computation bit-reproducible (so that runs are exactly repeatble with the same SEED
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment initialization
env = gym.make(id=ENV_ID,render_mode=None,frameskip=1)

# Info : Actions
possible_actions = env.unwrapped.get_action_meanings()
num_actions = len(possible_actions)
print(f"Environment : {ENV_ID}")
print(f"Possible actions for {ENV_ID} : {possible_actions}")
print(f"Nb. of actions for {ENV_ID} : {num_actions}")

# Preprocessing
env = AtariPreprocessing(
    env,
    noop_max = 30, # Table 1 - Nature paper
    frame_skip=4, # Preprocessing applied to the 4 most recent frames
    screen_size = 84, # Resizing from 210x160 to 84x84
    grayscale_obs=True, # Extracts Y (luminance)
    scale_obs=False, # Values remains 0..255 (not 0..1)
    terminal_on_life_loss=True, # for training targets
)

# Stack 4 frames as the state s_t
env = FrameStack(env,4)
print(env.observation_space)

# Preparing for evaluation
def make_eval_env():
    e = gym.make(id=ENV_ID, render_mode=None, frameskip=1)
    e = AtariPreprocessing(
        e,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False,  # <-- different from training
    )
    e = FrameStack(e, 4)
    return e
eval_env = make_eval_env()


# FIRE helper
def fire(env,obs):
    actions = env.unwrapped.get_action_meanings()
    if "FIRE" in actions:
        fire_idx = actions.index("FIRE")
        obs, _, terminated, truncated, _ =  env.step(fire_idx)
        return obs
    return obs

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
        if x.dtype == torch.uint8:
            x = x.float() # to keep 0..255 like in the paper
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x,start_dim=1) #start_dim = 1 conserve la dim du batch (x.shape = torch.Size([taille du batch, channel * h *w]))
        x = F.relu(self.fcn1(x))
        x = self.fcn2(x) # <-- q-values Q(s,a)
        return x
    
class ReplayBuffer:
    def __init__(self,capacity,obs_shape=(4,84,84),device=device):
        self.capacity = capacity #the buffer will hold at most 'capacity' transitions
        self.device = device
        self.idx = 0 # the write pointer (where the next transition will be stored)
        self.full = False #tells whether the buffer has wrapped around at least once
        self.obs = np.zeros((capacity, *obs_shape),np.uint8) # preallocate array for obs
        self.next_obs = np.zeros((capacity, *obs_shape),np.uint8) # preallocate array for next obs
        self.actions = np.zeros((capacity, ), np.int64) # Pytorch's gather expects LongTensor (int64)
        self.rewards = np.zeros((capacity,), dtype=np.int8)#rewards will be clipped to -1/0/1
        self.dones = np.zeros((capacity,),np.bool_) # episode/terminal flags (True when a transition ends an episode or life)

    def add(self, o, a, r, d, no):
        self.obs[self.idx] = o #observation, state s_t | This is the stack of 4 frames (shape [4, 84, 84]), representing what the agent saw before acting
        self.next_obs[self.idx] = no # s_t+1 | what the agent sees after the action
        self.actions[self.idx] = a # action taken at s_t | Stores the action the agent took, an integer (e.g., 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT in Breakout)
        self.rewards[self.idx] = np.clip(r,-1,1) # reward clipping
        # if r > 1, it becomes 1
        # if r < -1 it becomes -1
        # else, kept as is
        self.dones[self.idx] = d #done flag, i.e. wheter the episode ended (True if terminated or truncated)
        # when we compute targets later : y = r + max_a Q'(s_t+1,a)(1-done)
        # the done flag ensures that we don't bootstrap beyond terminal states
        # if done=True, then y = r (no future reward added)
        self.idx = (self.idx + 1) % self.capacity  # moves the pointer to the next storage slot (circular buffer)
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def sample(self,batch):
        rmemory_current_size = len(self)
        ids = np.random.randint(0,rmemory_current_size,size=batch) #samples 32 random transition from  anywhere in the buffer (to decorrelate experiences, and prevent overfitting)
        obs = torch.from_numpy(self.obs[ids]).to(self.device).float() # a mini-batch of state tensors ready for the model
        next_obs = torch.from_numpy(self.next_obs[ids]).to(self.device).float() # same as above but for next states s_t+1
        actions = torch.from_numpy(self.actions[ids]).to(self.device) #select the random actions corresponding to those states
        rewards = torch.from_numpy(self.rewards[ids]).to(self.device).float() #pulls the clipped reward values for those sampled transitions
        dones = torch.from_numpy(self.dones[ids]).to(self.device).float() #gets the done flags for each sampled transition
        return obs, actions, rewards, dones, next_obs #returns a full minibatch of tensors
        # obs, next_obs : [B,4,84,84]
        # actions, rewards, dones : [B]

# Policy / Target nets
policy_net = QNetworkCNN(4,num_actions).to(device)
target_net = QNetworkCNN(4,num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer, Loss
optimizer = torch.optim.RMSprop(
    policy_net.parameters(), lr = 2.5e-4, alpha=0.95, eps=0.01
)

huber_loss = nn.SmoothL1Loss()

# Epsilon-greedy action selection
def epsilon_by_frame(frame_idx):
    # We want to eps to start high (1.0)
    # Decay gradually as the agent learns more about the env
    # Reach a low floor (0.1)
    eps_start = 1.0
    eps_final = 0.1
    eps_decay = 1_000_000
    t = min(frame_idx/eps_decay,1.0)
    return eps_start + (eps_final - eps_start) * t

def select_action(obs_np,eps):
    if np.random.rand() < eps:
        # when eps = 1.0 : always random (pure exploration at the start)
        # when eps = 0.1 : 10% chance of random actions
        return np.random.randint(num_actions)
    
    with torch.no_grad():
        s = torch.from_numpy(obs_np).unsqueeze(0).to(device) # [4,84,84] -> [1,4,84,84]
        q = policy_net(s) # outputs a vector of Q-values (one per action) : Q(s,a_0), ..., Q(s,a_3)
        return int(q.argmax(1).item()) # returns the action index with the highest predicted return


def optimize_step(batch):
    # Performs one gradient update step using a minibatch sampled from the replay buffer

    obs, actions, rewards, dones, next_obs = batch

    # Q(s,a)
    q_values = policy_net(obs) # predicted Q-values for all actions in each state :shape [B, num_actions]
    q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # picks the q-value corresponding to the action actually taken in the replayed transition a_t [B]

    # target y = r + γ * max_a' Q_target(s',a') * (1 - done)

    with torch.no_grad():
        q_next = target_net(next_obs).max(1).values          # [B]
        target = rewards + gamma * q_next * (1.0 - dones)    # [B]
        # computes q-values for the next states using the frozen target network (for stability)
        # target is the fixed learning target

    loss = huber_loss(q, target)
    # L(x) = 0.5*x*x if abs(x) < 1
    # L(x) =  abs(x) - 0.5 if abs(x) >= 1

    optimizer.zero_grad(set_to_none=True) #set_to_none = true saves memory and avoids unnecessary zero tensors
    loss.backward() #computes dLoss/dtheta for all policy network params
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0) # caps gradients to +-1.0 to avoid exploding gradients
    optimizer.step() #RMSprop
    return float(loss.item())

# Evaluation function
def evaluate(n_episodes=5, eps_eval=0.05):
    scores = []
    for _ in range(n_episodes):
        ob, _ = eval_env.reset()

        # FIRE once for Breakout
        ob = np.array(fire(eval_env, np.array(ob)))

        # rollout
        ret, done = 0.0, False
        while not done:
            with torch.no_grad():
                s = torch.from_numpy(np.array(ob)).unsqueeze(0).to(device)
                if np.random.rand() < eps_eval:
                    a = np.random.randint(num_actions)
                else:
                    a = int(policy_net(s).argmax(1).item())
            ob, r, t, tr, _ = eval_env.step(a)
            ret += r
            done = t or tr

        scores.append(ret)

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    return mean, std


# Kaiming weight init for stability
def init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

policy_net.apply(init_kaiming)
target_net.load_state_dict(policy_net.state_dict())

def _save_plot(x, y, title, xlabel, ylabel, filename, extra_curves=None, ylog10=False):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)

    if ylog10:
        # Replace nonpositive values with a tiny epsilon to make log10 safe.
        eps = 1e-8
        y = [max(float(v), eps) for v in (y or [])]
        if extra_curves:
            extra_curves = [
                (xc, [max(float(v), eps) for v in (yc or [])], lab)
                for (xc, yc, lab) in extra_curves
            ]
        plt.yscale("log", base=10)
    if x is not None and y is not None and len(x) == len(y) and len(x) > 0:
        plt.plot(x, y, label="value")
    if extra_curves:
        for (x2, y2, label2) in extra_curves:
            if x2 is not None and y2 is not None and len(x2) == len(y2) and len(x2) > 0:
                plt.plot(x2, y2, label=label2)
    if extra_curves or (x is not None and y is not None and len(x) > 0):
        plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()


def save_all_plots():
    # 1) Training episode return (+ 100-epi moving avg)
    ma = moving_average(ep_returns, 100) if len(ep_returns) > 0 else []
    x_ma = ep_frames[-len(ma):] if len(ma) > 0 else []

    plt.figure()
    plt.title("Training Episode Return vs Frames")
    plt.xlabel("Frames")
    plt.ylabel("Episode Return (clipped)")

    # episode returns
    if len(ep_frames) > 0:
        plt.scatter(ep_frames, ep_returns, s=12, color="tab:blue", alpha=0.6, label="Episode returns")

    # moving average
    if len(ma) > 0:
        plt.plot(x_ma, ma, color="tab:red", linewidth=2, label="Moving avg (100 eps)")

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "train_return.png"), dpi=150)
    plt.close()

    # 2) Eval score vs frames
    _save_plot(eval_frames, eval_scores,
               "Evaluation Score vs Frames (30 no-ops, ε≈0.05)",
               "Frames", "Eval mean score",
               "eval_score.png")

    # 3) TD loss vs optimizer steps (+ moving avg 1k) — on log10 y-axis
    loss_ma = moving_average(loss_values, 1000) if len(loss_values) > 0 else []
    x_ma = loss_steps[-len(loss_ma):] if len(loss_ma) > 0 else []
    _save_plot(loss_steps, loss_values,
            "TD Loss (Huber) vs Optimizer Steps",
            "Optimizer Steps", "Huber Loss (log10)",
            "loss.png",
            extra_curves=[(x_ma, loss_ma, "moving avg (1k)")] if len(loss_ma) > 0 else None,
            ylog10=True)
    
    # 4) Epsilon vs frames
    _save_plot(eps_frames, eps_values,
               "Epsilon vs Frames",
               "Frames", "ε",
               "epsilon.png")

    # 5) Replay buffer size vs frames
    _save_plot(buf_frames, buf_sizes,
               "Replay Buffer Size vs Frames",
               "Frames", "Buffer size",
               "buffer_size.png")
    
class NoAutoFireOnReset(gym.Wrapper):
    """
    Ensures that reset() never auto-issues FIRE inside wrappers.
    We simply call the underlying reset and return immediately,
    leaving NOOPs and FIRE entirely to the caller / policy.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    
def make_record_env(name_prefix):
    e = gym.make(id=ENV_ID, render_mode="rgb_array", frameskip=1)
    e = AtariPreprocessing(
        e,
        noop_max=0,                  # we will do NOOPs ourselves so they get recorded
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=False
    )
    e = FrameStack(e, 4)
    e = NoAutoFireOnReset(e)         # <<< ensure reset() doesn’t auto-FIRE
    e = RecordVideo(
        e,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda ep: True,  # <<< record from reset()
        video_length=2000,
        name_prefix=name_prefix
    )
    return e



@torch.no_grad()
def record_eval_clip(policy_net, eps_eval=0.05, max_steps=6000):
    global eval_capture_idx
    prefix = f"eval_{VIDEO_RUN_ID}_{eval_capture_idx:03d}"
    rec_env = make_record_env(prefix)
    eval_capture_idx += 1

    ob, _ = rec_env.reset()

    # Record 1..30 NOOPs so the video shows the paddle + ball BEFORE launch
    for _ in range(np.random.randint(1, 31)):
        ob, _, t, tr, _ = rec_env.step(0)  # this is now recorded
        if t or tr:
            ob, _ = rec_env.reset()

    # Launch the ball with FIRE (also recorded)
    ob = np.array(fire(rec_env, np.array(ob)))

    ret, done, steps = 0.0, False, 0
    while not done and steps < max_steps:
        s = torch.from_numpy(np.array(ob)).unsqueeze(0).to(device)
        a = np.random.randint(num_actions) if np.random.rand() < eps_eval else int(policy_net(s).argmax(1).item())
        ob, r, t, tr, _ = rec_env.step(a)
        ret += r
        done = t or tr
        steps += 1

    rec_env.close()
    print(f"[RECORDED EVAL] return={ret:.1f} (clips in {VIDEO_DIR}/)")


# Track best/worst episode
best_return = float("-inf")
best_episode = -1
best_end_frame = -1

worst_return = float("inf")
worst_episode = -1
worst_end_frame = -1

episode_idx = 0

# Training loop
start_time = time.time()
planned_opt_steps = max_frames // train_freq  # how many optimizer steps you’ll do in total

buffer = ReplayBuffer(buffer_capacity,device=device)

obs, info = env.reset(seed=SEED)
obs = np.array(fire(env, np.array(obs)))

episode_return = 0.0
frame = 0
optimizer_steps = 0

while frame < max_frames:
    eps = epsilon_by_frame(frame)

    if frame % 10_000 == 0:
        eps_frames.append(frame)
        eps_values.append(eps)
        buf_frames.append(frame)
        buf_sizes.append(len(buffer))

    action = select_action(obs,eps)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_obs = np.array(next_obs)
    done = terminated or truncated

    buffer.add(obs, action, reward, done, next_obs)
    obs = next_obs
    episode_return += reward
    frame += 1

    # progress heartbeat every 10k frames
    if frame % 10_000 == 0:
        elapsed = time.time() - start_time
        fps = frame / elapsed if elapsed > 0 else 0.0
        pct_frames = 100.0 * frame / max_frames
        pct_opt = 100.0 * (optimizer_steps / max(1, planned_opt_steps))

        print(
            f"[{frame:,}/{max_frames:,} frames | {pct_frames:6.2f}%] "
            f"fps={fps:7.1f} | opt_steps={optimizer_steps:,} ({pct_opt:6.2f}%) "
            f"| eps={eps:0.3f} | buffer={len(buffer):,}"
        )

    # optimization every 4 env steps after warm-up
    if frame > learning_starts and (frame % train_freq == 0) and len(buffer) >= batch_size:
        loss = optimize_step(buffer.sample(batch_size))
        optimizer_steps += 1

        loss_steps.append(optimizer_steps)
        loss_values.append(loss)

        # hard target copy every 10k optimizer steps
        if optimizer_steps % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if done:
        print(f"[frame {frame}] return={episode_return:.1f}  eps={eps:.3f}  buffer={len(buffer)}")

        ep_frames.append(frame)
        ep_returns.append(episode_return)

        # update best / worst trackers
        cur_ret = ep_returns[-1]
        cur_frame_end = ep_frames[-1]

        if cur_ret > best_return:
            best_return = cur_ret
            best_episode = episode_idx
            best_end_frame = cur_frame_end
            print(f"[NEW BEST] ep #{best_episode} return={best_return:.1f} end_frame={best_end_frame:,}")

        if cur_ret < worst_return:
            worst_return = cur_ret
            worst_episode = episode_idx
            worst_end_frame = cur_frame_end
            print(f"[NEW WORST] ep #{worst_episode} return={worst_return:.1f} end_frame={worst_end_frame:,}")

        episode_idx += 1

        obs, info = env.reset()
        obs = np.array(fire(env, np.array(obs)))
        episode_return = 0.0

    # periodic evaluation & checkpoint
    if frame % 50_000 == 0 and frame > 0:
        mean_eval, std_eval = evaluate(n_episodes=5, eps_eval=0.05)
        eval_frames.append(frame)
        eval_scores.append(mean_eval)
        save_all_plots()
        record_eval_clip(policy_net,eps_eval=0.05)
        print(f"[EVAL @ {frame}] mean score = {mean_eval:.2f} ± {std_eval:.2f}")

        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"ckpt_{VIDEO_RUN_ID}_frame-{frame:07d}_opt-{optimizer_steps:06d}.pt"
        )

        torch.save({
            "policy": policy_net.state_dict(),
            "target": target_net.state_dict(),
            "opt": optimizer.state_dict(),
            "frame": frame,
            "optimizer_steps": optimizer_steps,
        }, ckpt_path)
        print(f"[CHECKPOINT] saved {ckpt_path}")

total_elapsed = time.time() - start_time
avg_fps = frame / total_elapsed if total_elapsed > 0 else 0.0
print(f"\n[FINISHED] {frame:,} frames in {total_elapsed/3600:.2f} h | avg fps={avg_fps:7.1f} | opt_steps={optimizer_steps:,}")
