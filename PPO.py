# ============================================================
# PPO sur Breakout (Gymnasium Atari) — avec logs + sauvegarde des plots
# ============================================================
# Dépendances (si besoin) :
# pip install "gymnasium[atari,accept-rom-license]==0.29.1" ale-py autorom \
#             torch tqdm opencv-python==4.8.1.78 numpy==1.26.4 matplotlib
# python -m AutoROM --accept-license

import os, random
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers.frame_stack import FrameStack


# ===================== Wrapper: FireReset =====================
class FireReset(gym.Wrapper):
    """Après reset, presse 2x FIRE pour lancer la balle."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc:
            obs, info = self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc:
            obs, info = self.env.reset(**kwargs)
        return obs, info


# ===================== Environnement =====================
def make_env(seed=0, render_mode=None, stack=4, fire_reset=True):
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=False,      # normalisation faite dans le réseau
        frame_skip=4,         # 1 step = 4 frames Atari
        screen_size=84,
        noop_max=30,
        terminal_on_life_loss=True,
    )
    env = FrameStack(env, num_stack=stack)  # (4,84,84) uint8
    if fire_reset:
        env = FireReset(env)
    env.reset(seed=seed)
    return env


# ===================== Réseau Actor-Critic =====================
class ActorCritic(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.fc = nn.Sequential(nn.Linear(64*7*7, 512), nn.ReLU(inplace=True))
        self.pi = nn.Linear(512, n_actions)  # logits
        self.v  = nn.Linear(512, 1)

    def forward(self, x_uint8):
        x = x_uint8.float() / 255.0
        z = self.fc(self.conv(x))
        logits = self.pi(z)
        value  = self.v(z).squeeze(-1)
        return logits, value

    def act(self, state_uint8, device, greedy=False):
        s = torch.tensor(state_uint8[None], dtype=torch.uint8, device=device)
        logits, value = self.forward(s)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            a = torch.argmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1).gather(1, a.view(-1,1)).squeeze(1)
            ent = torch.zeros_like(logp)
        else:
            a = dist.sample()
            logp = dist.log_prob(a)
            ent = dist.entropy()
        return int(a.item()), logp.squeeze(0), value.squeeze(0), ent.squeeze(0)

    def logprob(self, logits, actions):
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


# ===================== Utils RL =====================
def evaluate(env_eval, policy: ActorCritic, episodes=5, greedy=True, max_steps=50_000):
    device = next(policy.parameters()).device
    scores = []
    for _ in range(episodes):
        s, _ = env_eval.reset()
        s = np.array(s)
        ret, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            with torch.no_grad():
                a, _, _, _ = policy.act(s, device, greedy=greedy)
            s2, r, term, trunc, _ = env_eval.step(a)
            s = np.array(s2)
            ret += float(r)            # reward non clippée en éval
            done = term or trunc
            steps += 1
        scores.append(ret)
    return float(np.mean(scores)), float(np.std(scores))


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T,)
    values:  (T+1,) avec v_{t+1} = bootstrap final
    dones:   (T,) bool
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


# ===================== PPO Training =====================
def train_ppo(
    steps=1_000_000,          # nombre total d'interactions env
    rollout_steps=4096,       # T (par update)
    gamma=0.99,
    lam=0.95,                 # GAE
    lr=2.5e-4,
    clip_eps=0.1,             # epsilon ppo (policy)
    vf_clip=0.2,              # clipping value
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=10.0,
    epochs=4,                 # nb de passes PPO par batch
    minibatch_size=256,
    reward_clip=True,         # clip sign(r) au train
    eval_every=50_000,
    eval_episodes=5,
    seed=0,
    cpu=True,
    ckpt_best="ppo_breakout_best.pt",
    ckpt_last="ppo_breakout_last.pt",
):
    # Seeds & device
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
    print("Device:", device)

    # Envs
    env = make_env(seed=seed, render_mode=None, fire_reset=True)
    env_eval = make_env(seed=seed+1, render_mode=None, fire_reset=True)

    nA = env.action_space.n
    net = ActorCritic(n_actions=nA).to(device)
    opt = optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    total_steps, best_eval = 0, -1e9
    pbar = trange(10**9, desc="Train (updates)", dynamic_ncols=True)

    # ---------- LOGS ----------
    logs = {
        "step": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "loss": [],
        "approx_kl": [],
        "clip_frac": [],
        "eval_step": [],
        "eval_mean": [],
        "eval_std": [],
    }

    next_eval = eval_every

    # ======== Main loop ========
    while total_steps < steps:
        # ----- Rollout -----
        obs_buf   = np.zeros((rollout_steps, 4, 84, 84), dtype=np.uint8)
        act_buf   = np.zeros((rollout_steps,), dtype=np.int64)
        logp_buf  = np.zeros((rollout_steps,), dtype=np.float32)
        val_buf   = np.zeros((rollout_steps+1,), dtype=np.float32)  # +1 pour bootstrap
        rew_buf   = np.zeros((rollout_steps,), dtype=np.float32)
        done_buf  = np.zeros((rollout_steps,), dtype=np.bool_)

        s, _ = env.reset()
        s = np.array(s)

        for t in range(rollout_steps):
            obs_buf[t] = s
            a, logp, v, _ = net.act(s, device, greedy=False)
            s2, r, term, trunc, _ = env.step(a)

            act_buf[t]  = a
            logp_buf[t] = float(logp.item())
            val_buf[t]  = float(v.item())
            rew_buf[t]  = float(np.sign(r) if reward_clip else r)
            done = term or trunc
            done_buf[t] = done

            s = np.array(s2)
            total_steps += 1
            if done or total_steps >= steps:
                s, _ = env.reset()
                s = np.array(s)

        # bootstrap value pour la dernière obs
        with torch.no_grad():
            s_t = torch.tensor(s[None], dtype=torch.uint8, device=device)
            _, v_last = net.forward(s_t)
            val_buf[rollout_steps] = float(v_last.squeeze(0).item())

        # ----- GAE -----
        adv_np, ret_np = compute_gae(rew_buf, val_buf, done_buf, gamma=gamma, lam=lam)
        # normalisation des avantages
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        # tensors
        obs_t   = torch.tensor(obs_buf, dtype=torch.uint8, device=device)
        act_t   = torch.tensor(act_buf, dtype=torch.long,  device=device)
        oldlogp = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t   = torch.tensor(adv_np, dtype=torch.float32, device=device)
        ret_t   = torch.tensor(ret_np, dtype=torch.float32, device=device)
        oldv_t  = torch.tensor(val_buf[:-1], dtype=torch.float32, device=device)

        # ----- PPO Updates -----
        n = rollout_steps
        idx = np.arange(n)
        mb = minibatch_size
        clip_fracs = []
        approx_kls = []

        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb):
                end = start + mb
                mb_idx = idx[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                mb_oldlogp = oldlogp[mb_idx]
                mb_oldv = oldv_t[mb_idx]

                logits, values = net.forward(mb_obs)
                newlogp, entropy = net.logprob(logits, mb_act)

                # ratio
                ratio = torch.exp(newlogp - mb_oldlogp)
                # policy loss (clipped)
                unclipped = ratio * mb_adv
                clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                # value loss (clipped)
                v_pred = values
                v_clipped = mb_oldv + torch.clamp(v_pred - mb_oldv, -vf_clip, vf_clip)
                v_loss_unclipped = (v_pred - mb_ret).pow(2)
                v_loss_clipped   = (v_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))

                # entropy bonus
                ent = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * ent

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()

                with torch.no_grad():
                    # track
                    approx_kl = torch.mean(mb_oldlogp - newlogp).abs().item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()
                    approx_kls.append(approx_kl)
                    clip_fracs.append(clip_frac)

        # ---- Logging (par update) ----
        logs["step"].append(total_steps)
        logs["policy_loss"].append(float(policy_loss.item()))
        logs["value_loss"].append(float(value_loss.item()))
        logs["entropy"].append(float(ent.item()))
        logs["loss"].append(float(loss.item()))
        logs["approx_kl"].append(float(np.mean(approx_kls)))
        logs["clip_frac"].append(float(np.mean(clip_fracs)))

        pbar.set_description(f"steps={total_steps:,} loss={loss.item():.3f} kl={logs['approx_kl'][-1]:.4f}")

        # ---- Évaluation + checkpoints (seuil glissant) ----
        while total_steps >= next_eval or total_steps >= steps:
            mean_score, std = evaluate(env_eval, net, episodes=eval_episodes, greedy=True)
            pbar.write(f"[Eval] steps={total_steps:,} | score={mean_score:.2f} ± {std:.2f}")
            torch.save(net.state_dict(), ckpt_last)
            if mean_score > best_eval:
                best_eval = mean_score
                torch.save(net.state_dict(), ckpt_best)
            logs["eval_step"].append(total_steps)
            logs["eval_mean"].append(mean_score)
            logs["eval_std"].append(std)
            next_eval += eval_every
            if total_steps >= steps:
                break

    env.close(); env_eval.close()
    print(f"Training finished. Best eval score={best_eval:.2f}")
    return net, logs


# ===================== Helpers graphiques (avec sauvegarde) =====================
def _moving_average(x, w=5):
    if len(x) == 0: return np.array([])
    w = max(1, int(w))
    c = np.convolve(x, np.ones(w)/w, mode="valid")
    pad = [np.nan]*(len(x)-len(c))
    return np.array(list(pad) + list(c))

def plot_logs(logs, save_dir=None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _save_or_show(name):
        if save_dir:
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}.png"))
            plt.close()
        else:
            plt.show()

    # Eval score
    if len(logs.get("eval_step", [])) > 0:
        s = np.array(logs["eval_step"]); m = np.array(logs["eval_mean"]); sd = np.array(logs["eval_std"])
        plt.figure(figsize=(7,4))
        plt.plot(s, m, "-o", label="Score moyen (éval)")
        plt.fill_between(s, m-sd, m+sd, alpha=0.2, label="± écart-type")
        plt.xlabel("Steps"); plt.ylabel("Score (éval)")
        plt.title("PPO – Score d'évaluation")
        plt.legend(); plt.grid(True)
        _save_or_show("eval_score")

    # Losses
    if len(logs.get("step", [])) > 0:
        s = np.array(logs["step"])
        for key, title in [("loss","Loss totale"), ("policy_loss","Policy loss"),
                           ("value_loss","Value loss"), ("entropy","Entropie")]:
            if key in logs:
                y = np.array(logs[key])
                plt.figure(figsize=(7,4))
                plt.plot(s, y, label=key)
                plt.plot(s, _moving_average(y, 5), label=f"{key} (moy. mob.)")
                plt.xlabel("Steps"); plt.ylabel(key); plt.title(f"PPO – {title}")
                plt.legend(); plt.grid(True)
                _save_or_show(f"{key}")

        # KL & clip frac
        if "approx_kl" in logs:
            y = np.array(logs["approx_kl"])
            plt.figure(figsize=(7,4))
            plt.plot(s, y, label="approx_kl")
            plt.xlabel("Steps"); plt.ylabel("KL approx")
            plt.title("PPO – Approx KL"); plt.legend(); plt.grid(True)
            _save_or_show("approx_kl")

        if "clip_frac" in logs:
            y = np.array(logs["clip_frac"])
            plt.figure(figsize=(7,4))
            plt.plot(s, y, label="clip_frac")
            plt.xlabel("Steps"); plt.ylabel("Fraction clippée")
            plt.title("PPO – Clip fraction"); plt.legend(); plt.grid(True)
            _save_or_show("clip_frac")


# ===================== Lancement =====================
if __name__ == "__main__":
    net, logs = train_ppo(
        steps=100_000,         # augmente si tu peux
        rollout_steps=4096,
        gamma=0.99,
        lam=0.95,
        lr=2.5e-4,
        clip_eps=0.1,
        vf_clip=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=10.0,
        epochs=4,
        minibatch_size=256,
        reward_clip=True,
        eval_every=50_000,
        eval_episodes=5,
        seed=0,
        cpu=True
    )

    plot_logs(logs, save_dir="plots_ppo_breakout")





