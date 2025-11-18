# Deep Q-Network (DQN) for Atari

PyTorch implementation of the DQN algorithm from the Nature paper: ["Human-level control through deep reinforcement learning"](https://www.nature.com/articles/nature14236) (Mnih et al., 2015).

## 📁 Project Structure

```
.
├── agents/
│   └── dqn.py                 # DQN agent implementation with replay buffer integration
├── configs/
│   ├── dqn_breakout.yaml      # Hyperparameters for Breakout (Nature paper settings)
│   └── dqn_pong.yaml          # Hyperparameters for Pong (optimized for faster training)
├── env/
│   └── wrappers.py            # Atari preprocessing wrappers (Nature DQN standard)
├── memory/
│   └── replay_buffer.py       # Experience replay buffer
├── networks/
│   └── atari_cnn.py           # Q-Network architecture (Nature paper CNN)
├── utils/
│   ├── config.py              # YAML config loading utilities
│   ├── eval_utils.py          # Evaluation utilities
│   ├── logger.py              # CSV logging for metrics
│   ├── plot_training.py       # Plotting utilities for training curves
│   └── seed.py                # Random seed utilities
├── train.py                   # Main training script
├── eval.py                    # Evaluation script with video recording
└── README.md                  # This file
```

### File Descriptions

#### Core Scripts
- **`train.py`**: Main training loop. Handles environment interaction, agent updates, logging, and checkpointing. Supports resuming from checkpoints.
- **`eval.py`**: Evaluates trained agents and records gameplay videos. Useful for visualizing learned policies.

#### Agent & Networks
- **`agents/dqn.py`**: Core DQN agent with:
  - Q-network and target network
  - Epsilon-greedy exploration schedule
  - Experience replay integration
  - Huber loss optimization
  - Gradient clipping support
- **`networks/atari_cnn.py`**: 3-layer CNN from Nature paper (32→64→64 filters, 512 FC units)

#### Environment & Memory
- **`env/wrappers.py`**: Atari preprocessing pipeline:
  - NoopReset: Random no-ops at episode start
  - FireReset: Automatic FIRE action for games that need it
  - MaxAndSkip: Frame skipping (4 frames) with max pooling
  - WarpFrame: Resize to 84×84 grayscale
  - FrameStack: Stack 4 consecutive frames
  - ClipReward: Clip rewards to {-1, 0, +1} (training only)
- **`memory/replay_buffer.py`**: Fixed-size circular buffer storing (s, a, r, s', done) transitions

#### Utilities
- **`utils/config.py`**: Loads YAML configs and converts to DQNConfig dataclass
- **`utils/eval_utils.py`**: Runs evaluation episodes and computes statistics
- **`utils/logger.py`**: CSV logger that appends metrics (supports resume mode)
- **`utils/plot_training.py`**: Generates training curves with moving averages
- **`utils/seed.py`**: Sets random seeds for reproducibility

## 🚀 Quick Start

### Installation

```bash
pip install torch gymnasium[atari] ale-py opencv-python pyyaml matplotlib pandas
```

### Training

Train DQN on Breakout (Nature paper settings):
```bash
python train.py --config configs/dqn_breakout.yaml
```

Train on Pong (faster convergence):
```bash
python train.py --config configs/dqn_pong.yaml
```

Override config parameters via CLI:
```bash
python train.py --config configs/dqn_breakout.yaml \
    --seed 42 \
    --total_steps 5000000 \
    --eval_interval 100000
```

Resume training from checkpoint:
```bash
python train.py --config configs/dqn_breakout.yaml \
    --resume runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/ckpt_1000000.pt
```

### Evaluation

Evaluate a trained agent:
```bash
python eval.py \
    --checkpoint runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/final.pt \
    --episodes 30 \
    --env_id ALE/Breakout-v5
```

Record all episodes as videos:
```bash
python eval.py \
    --checkpoint runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/final.pt \
    --episodes 10 \
    --record_all
```

### Plotting Results

Generate training curves from logged metrics:
```bash
python utils/plot_training.py runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/metrics.csv
```

With custom moving average window:
```bash
python utils/plot_training.py runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/metrics.csv 200
```

## 📊 Expected Results

### Breakout
- **Training time**: ~24-48 hours on GPU for 10M steps
- **Target performance**: 400+ average reward after 10M steps (Nature paper reports ~400)
- **Convergence**: Visible improvement after ~2-3M steps

### Pong
- **Training time**: ~12-24 hours on GPU for 10M steps  
- **Target performance**: +18 to +21 average reward
- **Convergence**: Usually solves the game (>18 reward) within 2-4M steps

## ⚙️ Configuration

### Nature DQN Hyperparameters (Breakout)

Based on the Nature paper, the following settings are used for Breakout:

```yaml
agent:
  replay_capacity: 1000000      # 1M transitions
  replay_warmup: 50000          # 50k steps before training
  batch_size: 32
  gamma: 0.99
  lr: 0.00025                   # 2.5e-4 with RMSprop
  rmsprop_alpha: 0.95
  rmsprop_eps: 0.01
  optimize_every: 4             # Update every 4 steps
  target_update_interval: 10000 # Sync target net every 10k steps
  huber_delta: 1.0

exploration:
  eps_start: 1.0
  eps_end: 0.1
  eps_anneal_frames: 1000000    # Linear decay over 1M steps
```

### Key Differences for Pong

Pong converges faster and needs less memory:
- Smaller replay buffer (200k vs 1M)
- Same exploration schedule works well
- Typically converges in 2-4M steps vs 8-10M for Breakout

## 📂 Run Directory Structure

Each training run creates a timestamped directory:

```
runs/dqn-ALE_Breakout_v5-seed42-20241112-143022/
├── config.yaml              # Saved configuration for reproducibility
├── run_info.txt            # Training parameters and system info
├── metrics.csv             # Training/eval metrics (step, loss, rewards, etc.)
├── ckpt_250000.pt         # Periodic checkpoints
├── ckpt_500000.pt
├── ...
└── final.pt               # Final trained model
```

**Metrics logged:**
- `step`: Environment steps
- `updates`: Gradient updates performed
- `loss`: TD loss
- `q_max`: Maximum Q-value (tracks learning progress)
- `epsilon`: Current exploration rate
- `fps`: Training throughput
- `episode_return`: Episode returns
- `eval_return_mean/std/min/max`: Evaluation statistics

## 🔍 Code Audit Notes

### ✅ Strengths
1. **Faithful Nature DQN implementation**: Follows paper specifications closely
2. **Modular design**: Clean separation of concerns (agent, network, environment, utils)
3. **Resume support**: Can resume training from checkpoints
4. **Comprehensive logging**: Tracks all relevant metrics
5. **Proper preprocessing**: Correct wrapper order and frame stacking

### ⚠️ Observations
1. **Action space handling**: Code correctly handles both minimal and full action spaces for Atari
2. **Epsilon schedule**: Uses linear annealing over frames (not steps), consistent with Nature paper
3. **Reward clipping**: Only applied during training, not evaluation (correct)
4. **Target network**: Updated via hard copy every 10k steps (Nature paper approach)

### 🐛 Potential Issues
1. **Pong action space**: The `PongUpDownActionMap` wrapper reduces action space to 3 actions (NOOP, UP, DOWN), which is fine but ensure checkpoint compatibility
2. **Evaluation epsilon**: Config files use different eval epsilons (0.05 vs 0.001) - Nature paper uses 0.05
3. **Gradient clipping**: Pong config has `max_grad_norm: 10.0` but Nature paper doesn't use clipping - consider setting to `null`

## 🎯 Tips for Best Results

1. **Use multiple seeds**: Run 3-5 seeds for robust results
2. **Monitor Q-values**: Increasing `q_max` indicates learning progress
3. **Check epsilon decay**: Should reach 0.1 around 1M steps
4. **GPU recommended**: Training on CPU is very slow (10-20x slower)
5. **Eval frequency**: 250k steps is good for Breakout, 100k for Pong

## 📚 References

- Nature DQN Paper: [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- OpenAI Baselines: [Stable Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- DeepMind Lab: [Atari preprocessing](https://github.com/deepmind/dqn)

## 📝 License

This implementation is for educational purposes. Please cite the original Nature paper if you use this code in research.