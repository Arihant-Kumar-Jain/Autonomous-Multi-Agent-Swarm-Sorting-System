# MAPPO – Multi-Agent Ball Delivery

A production-quality implementation of **Multi-Agent PPO (MAPPO)** with:

- **Shared Actor** with stochastic (Categorical) policy
- **Centralized Critic** (CTDE) — sees full global state during training
- **Multi-Head Attention Communication** — differentiable, learned end-to-end
- **GAE + entropy regularization** for stable PPO updates
- **3-phase curriculum learning** (no obstacles → sparse → full)
- **Observation & reward normalization**
- **TensorBoard logging + matplotlib training curves**
- **Pygame visualization** (demo + training render modes)

---

## Project Structure

```
marl_project/
├── main.py              # Entry point
├── config.py            # All hyperparameters
├── env/
│   └── environment.py   # 30×30 GridWorld
├── agents/
│   └── agent.py         # Lightweight agent wrapper
├── marl/
│   ├── actor.py         # SharedActor (policy network)
│   ├── critic.py        # CentralizedCritic V(s)
│   ├── communication.py # Multi-Head Attention comm
│   ├── mappo.py         # PPO trainer (CTDE)
│   └── buffer.py        # Rollout buffer + GAE
├── utils/
│   └── helpers.py       # Normalization, logging, curriculum
├── models/              # Saved checkpoints
├── logs/                # CSV logs + TensorBoard events
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train (fast, no render)
python main.py

# Train with custom episode count
python main.py --episodes 3000

# Train with pygame visualization
python main.py --render

# Demo mode (loads saved model)
python main.py --demo

# Evaluation (100 episodes, no exploration)
python main.py --eval

# Ablation: disable communication module
python main.py --no-comm

# TensorBoard
tensorboard --logdir logs/
```

---

## Architecture

### CTDE (Centralized Training, Decentralized Execution)

```
Training time:
  GlobalState ──► CentralizedCritic ──► V(s)  [for GAE]
  LocalObs_i  ──► SharedActor ──► π(a|o_i)

Execution time:
  LocalObs_i  ──► SharedActor ──► action_i   [NO global state needed]
```

### Communication Module

```
  hidden_i ──► msg_encoder ──► message_i
                                    │
  [msg_0, msg_1, ..., msg_N] ──► MultiHeadAttention
                                    │
                              comm_embedding_i
                                    │
  [hidden_i || comm_i] ──► action_head ──► logits
```

### PPO Update

```
For each update interval (2048 steps):
  1. Compute GAE advantages A_t
  2. For K=4 epochs:
     For each mini-batch:
       ratio = exp(log π_new(a|o) - log π_old(a|o))
       L_clip = min(ratio * A, clip(ratio, 1±ε) * A)
       L_value = MSE(V(s), returns)
       L_entropy = -H[π(·|o)]
       L = -L_clip + c1*L_value + c2*L_entropy
```

---

## Reward Design

| Event | Reward |
|-------|--------|
| Successful delivery | +25 |
| Successful pickup | +5 |
| All balls delivered | +50 (team bonus) |
| Distance reduction to target | +scale × delta |
| Per step | -0.2 |
| Collision (wall/agent) | -5 |
| Invalid PICK or DROP | -2 |
| Oscillation (revisit) | -1 |

---

## Curriculum

| Phase | Obstacles | Episodes |
|-------|-----------|----------|
| 0 | 0 | 300 |
| 1 | 15 | 300 |
| 2 | 30 | ∞ |

---

## Hyperparameters (config.py)

| Parameter | Value |
|-----------|-------|
| Grid size | 30×30 |
| Agents | 3 |
| Balls | 10 |
| Max steps/ep | 500 |
| Hidden dim | 256 |
| Comm dim | 64 |
| Attention heads | 4 |
| PPO clip ε | 0.2 |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Entropy coef | 0.01 |
| PPO epochs | 4 |
| Mini-batch | 256 |
| Update interval | 2048 steps |
| Actor LR | 3e-4 |
| Critic LR | 1e-3 |