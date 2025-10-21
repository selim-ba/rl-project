# Source base: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import os
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===============================
# ENV
# ===============================
env = gym.make("CartPole-v1", render_mode="rgb_array")
# env = gym.make("MountainCar-v0", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",      
    name_prefix="cartpole_run",  
    episode_trigger=lambda ep: ep % 25 == 0  # recored every 10 episodes
)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# ===============================
# REPLAY MEMORY
# ===============================
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ===============================
# Q-NETWORK
# ===============================
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_features=n_observations, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# ===============================
# HYPERPARAMS
# ===============================
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

# ===============================
# INIT
# ===============================
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
episode_durations = []

# ===============================
# METRICS / LOGS
# (global per-step logs unless noted)
# ===============================
epsilon_history = []                  # ε_t
steps_done_history = []               # t (global step)
loss_history = []                     # loss per optimize step

q_policy_mean_history = []            # mean over batch, per action -> averaged -> scalar
q_target_mean_history = []            # same for target
q_gap_abs_mean_history = []           # mean absolute |Q_pol - Q_tgt| over actions

next_state_values_mean_history = []   # mean(V(s_{t+1}))
expected_state_values_mean_history = [] # mean(target y = r + γ V(s'))

# State components (CartPole obs = [x, x_dot, theta, theta_dot])
state_episode = 0
state_traces = {                       # per-step traces with episode tag
    "episode": [],
    "t": [],
    "position": [],
    "speed": [],
    "angle": [],
    "ang_speed": [],
}

def log_state_components(ep, t, state_tensor):
    # state_tensor shape: (1, 4)
    s = state_tensor[0].detach().cpu().numpy()
    state_traces["episode"].append(ep)
    state_traces["t"].append(t)
    state_traces["position"].append(s[0])
    state_traces["speed"].append(s[1])
    state_traces["angle"].append(s[2])
    state_traces["ang_speed"].append(s[3])

def current_epsilon():
    # deterministic function of steps_done (same formula as used to threshold)
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)

# ===============================
# ACTION SELECTION (ε-greedy)
# ===============================
def select_action(state):
    global steps_done
    sample = random.random()

    eps_threshold = current_epsilon()
    # log BEFORE we increment steps_done (this is the ε used for this decision)
    epsilon_history.append(eps_threshold)
    steps_done_history.append(steps_done)

    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# ===============================
# DURATIONS PLOT (original helper)
# ===============================
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# ===============================
# OPTIMIZATION STEP
# ===============================
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) if any(non_final_mask) \
                            else torch.empty((0, n_observations), device=device)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s,a) from policy
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s') from target
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if non_final_next_states.numel() > 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Target y = r + γ V(s')
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # ====== METRICS (batch-level means) ======
    with torch.no_grad():
        q_pol_all = policy_net(state_batch)                   # [B, A]
        q_pol_mean_scalar = q_pol_all.mean().item()

        if non_final_next_states.numel() > 0:
            q_tgt_all = target_net(non_final_next_states)     # [B_nf, A]
            # Align action-dimension means for an apples-to-apples aggregate:
            q_tgt_mean_scalar = q_tgt_all.mean().item()
            # For the gap, compare means per action (pad using policy means if nf differs? use global mean diff):
            q_gap_abs_mean = abs(q_pol_mean_scalar - q_tgt_mean_scalar)
        else:
            q_tgt_mean_scalar = 0.0
            q_gap_abs_mean = abs(q_pol_mean_scalar - 0.0)

        q_policy_mean_history.append(q_pol_mean_scalar)
        q_target_mean_history.append(q_tgt_mean_scalar)
        q_gap_abs_mean_history.append(q_gap_abs_mean)

        next_state_values_mean_history.append(next_state_values.mean().item())
        expected_state_values_mean_history.append(expected_state_action_values.mean().item())
        loss_history.append(loss.item())

        # Optional debug prints
        # print("---- OPTIMIZE STEP ----")
        # print(f"Q_policy mean: {q_pol_mean_scalar:.4f}")
        # print(f"Q_target mean: {q_tgt_mean_scalar:.4f}")
        # print(f"|ΔQ| mean:    {q_gap_abs_mean:.6f}")
        # print(f"V(s') mean:   {next_state_values_mean_history[-1]:.4f}")
        # print(f"y mean:       {expected_state_values_mean_history[-1]:.4f}")
        # print(f"Loss:         {loss_history[-1]:.6f}")

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ===============================
# TRAIN
# ===============================
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

global_step_in_episode = 0

for i_episode in range(num_episodes):
    # reset
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    state_episode = i_episode
    global_step_in_episode = 0

    # log initial state
    log_state_components(state_episode, global_step_in_episode, state)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # store
        memory.push(state, action, next_state, reward)

        # step state pointer
        state = next_state

        # log state (if not terminal, we have next_state)
        if state is not None:
            global_step_in_episode += 1
            log_state_components(state_episode, global_step_in_episode, state)

        # learn
        optimize_model()

        # soft-update target
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    if i_episode % 25 == 0:
        env.render()

print('Complete')

plot_durations(show_result=True)


os.makedirs("plots", exist_ok=True)

def save_plot(x, y, title, xlabel, ylabel, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(x) == len(y) and len(x) > 0:
        plt.plot(x, y)
    elif len(y) > 0:
        plt.plot(y)
    plt.tight_layout()
    save_path = os.path.join("plots", filename)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved: {save_path}")
    plt.close()

# 1) epsilon over steps
save_plot(steps_done_history, epsilon_history,
          "Epsilon (ε) vs Steps", "Global step", "ε",
          "epsilon_vs_steps.png")

# 2) steps_done progression
save_plot(range(len(steps_done_history)), steps_done_history,
          "Steps Done (cumulative)", "Logged step", "steps_done",
          "steps_done.png")

# 3) State components
save_plot(state_traces["t"], state_traces["position"],
          "Position (x)", "t (in-episode step)", "x", "position.png")
save_plot(state_traces["t"], state_traces["speed"],
          "Vitesse (x_dot)", "t (in-episode step)", "m/s", "speed.png")
save_plot(state_traces["t"], state_traces["angle"],
          "Angle (theta)", "t (in-episode step)", "rad", "angle.png")
save_plot(state_traces["t"], state_traces["ang_speed"],
          "Vitesse angulaire (theta_dot)", "t (in-episode step)", "rad/s",
          "angular_speed.png")

# 4) Q metrics
save_plot(range(len(q_policy_mean_history)), q_policy_mean_history,
          "Q_policy (mean over batch x actions)", "Optimize step", "Q",
          "q_policy.png")
save_plot(range(len(q_target_mean_history)), q_target_mean_history,
          "Q_target (mean over batch x actions)", "Optimize step", "Q",
          "q_target.png")
save_plot(range(len(q_gap_abs_mean_history)), q_gap_abs_mean_history,
          "Écart moyen |Q_policy - Q_target|", "Optimize step", "|ΔQ|",
          "q_gap.png")

# 5) Value targets + loss
save_plot(range(len(next_state_values_mean_history)), next_state_values_mean_history,
          "next_state_values (mean V(s'))", "Optimize step", "V",
          "next_state_values.png")
save_plot(range(len(expected_state_values_mean_history)), expected_state_values_mean_history,
          "expected_state_action_values (mean y)", "Optimize step", "y",
          "expected_state_values.png")
save_plot(range(len(loss_history)), loss_history,
          "Loss (SmoothL1)", "Optimize step", "loss",
          "loss.png")

print("\n All plots saved in the 'plots/' directory.\n")
