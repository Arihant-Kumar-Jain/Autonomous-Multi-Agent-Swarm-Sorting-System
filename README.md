# Multi-Agent Warehouse: Collaborative Task Execution

3 autonomous robots collaboratively collect 5 objects and deliver them to a drop zone in a warehouse, using **BFS**, **DQN**, **PPO**, or **MAPPO** navigation — with comparison metrics.

## Project Structure

```
multi_agent_warehouse/
├── 🎮 Pygame Simulation (RL training — fast)
│   ├── config.py              # Grid layout, rewards, all hyperparameters
│   ├── warehouse_env.py       # Multi-agent environment (15×15 grid)
│   ├── pathfinding.py         # BFS / A* pathfinding
│   ├── task_allocator.py      # Greedy + congestion-aware task allocation
│   ├── dqn_agent.py           # Dueling Double DQN agent
│   ├── ppo_agent.py           # PPO (Proximal Policy Optimization) agent
│   ├── mappo_agent.py         # MAPPO (Multi-Agent PPO, CTDE) agent
│   ├── trainer.py             # Train all RL variants
│   ├── main.py                # Run simulation + 6-way comparison
│   ├── visualizer.py          # Pygame real-time visualization
│   └── plot_results.py        # Generate comparison plots
│
└── 🤖 ROS2 + Gazebo (deployment — realistic)
    └── ros2_ws/src/warehouse_multi_agent/
        ├── worlds/warehouse.world
        ├── launch/warehouse.launch.py
        ├── scripts/
        │   ├── task_coordinator.py    # Central task allocation
        │   ├── rl_navigator.py        # RL bridge: trained model → cmd_vel
        │   ├── bfs_navigator.py       # BFS baseline navigator
        │   └── metrics_logger.py      # CSV logging
        └── config/warehouse_config.yaml
```

---

## Environment

### Warehouse Grid (15×15)

```
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 🟢 . . . . 🔵 . . . . . 🔴 . 1    ← 3 robots spawn at top
1 . ██ . ██ . ██ . ██ . ██ . 1    ← shelving racks (walls)
1 . ██ . ██ . ██ . ██ . ██ . 1
1 . . . . ⭐ . . . . . . . 1    ← 5 collectible objects
1 . ██ . ██ . ██ . ██ . ██ . 1
1 . ██ . ██ . ██ . ██ . ██ . 1
1 . . . . . . . . . . . . . 1
1 . ██ . ██ . ██ . ██ . ██ . 1
1 . ██ . ██ . ██ . ██ . ██ . 1
1 . . . . . . . . . . . . . 1
1 . . . . . . . . . . . . . 1
1 . . . . . 🟪🟪🟪 . . . . 1    ← 3×3 drop zone
1 . . . . . 🟪🟪🟪 . . . . 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

### Task Lifecycle

```
Robot idle → task_allocator assigns object → robot navigates to object
  → auto-pickup (+10 reward) → target switches to drop zone
  → robot navigates to drop zone → auto-delivery (+20 reward)
  → robot recycled for remaining objects → repeat until all 5 delivered
```

### Observation (State Vector)

| Dims | Feature | Source |
|------|---------|--------|
| `[0-1]` | Direction to goal (Δrow, Δcol, normalized) | Grid position |
| `[2]` | Distance to goal (normalized) | Manhattan distance |
| `[3]` | Carrying object (0/1) | Boolean flag |
| `[4-7]` | Relative position of 2 other robots | Grid positions |
| `[8-11]` | Obstacle sensors (up/down/left/right) | Grid wall check |
| `[12-14]` | Robot identity (one-hot: 3 robots) | Robot ID |
| `[15-16]` | Congestion + local density *(improved/MAPPO only)* | Nearby robot count |

- **Base state**: 15 dimensions (DQN, PPO)
- **Congestion-aware state**: 17 dimensions (DQN+Congestion, PPO+Congestion, MAPPO)

### Reward Function

| Signal | Value | Purpose |
|--------|-------|---------|
| Deliver to drop zone | +20.0 | Main goal |
| Pick up object | +10.0 | Sub-goal |
| Move closer to target | +0.5 | Reward shaping |
| Move farther from target | −0.3 | Discourage wandering |
| Per-step penalty | −0.05 | Encourage efficiency |
| Wait action | −0.1 | Discourage idling |
| Hit wall | −2.0 | Boundary awareness |
| Robot-robot collision | −10.0 | Collision avoidance |
| **Proximity penalty** | −0.3/nearby robot | Spread out *(congestion modes only)* |
| **Collision congestion** | −1.0 | Extra collision cost *(congestion modes only)* |

### Actions

5 discrete actions: `UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`

---

## Algorithms

### 1. BFS (Baseline)

Shortest-path grid search with priority-based collision avoidance. No learning — pure pathfinding.

### 2. DQN (Dueling Double DQN)

- **Policy**: ε-greedy over learned Q-values (argmax)
- **Architecture**: Shared feature extractor (128-dim) → Value stream + Advantage stream
- **Key features**: Experience replay (100K buffer), Double DQN (reduces overestimation), dueling architecture (separates state value from action advantage)
- **State**: 15-dim (base) or 17-dim (congestion-aware)

### 3. PPO (Proximal Policy Optimization)

- **Policy**: Softmax probability distribution over actions
- **Architecture**: Shared backbone (256-dim, LayerNorm) → Actor head (policy) + Critic head (value)
- **Key features**: Clipped surrogate objective (±0.2), GAE (λ=0.95), entropy bonus (0.05) for stable exploration
- **Update**: On-policy — collects rollouts (1024 steps), then does 10 epochs of mini-batch updates

### 4. DQN + Congestion / PPO + Congestion

Same as above but with:
- **17-dim state** (adds congestion score + local density)
- **Per-step proximity penalty** (−0.3 per nearby robot within radius 3)
- **Congestion-aware task allocation** (assigns objects to less crowded robots)

### 5. MAPPO (Multi-Agent PPO — CTDE)

**Centralized Training, Decentralized Execution:**
- **Actor**: Shared per-robot policy network (same as PPO actor) — each robot uses only its own observation
- **Critic**: Centralized — takes concatenated observations of ALL 3 robots (51-dim global state) to estimate value
- **Why it helps**: The critic "sees" the full system during training, learning things like "robot 0 is heading left while robot 2 is heading right — this is good coordination"
- **Deployment**: Only the actor is needed — runs decentralized, same as PPO

```
Training:                           Deployment:
┌─────────────────────┐            ┌──────────────┐
│ Centralized Critic  │            │              │
│ V(obs_0 ⊕ obs_1 ⊕  │            │  Actor only  │
│   obs_2) → value    │            │  obs → action│
├─────────────────────┤            └──────────────┘
│ Shared Actor        │              (per robot)
│ obs_i → action_i    │
└─────────────────────┘
```

---

## Quick Start

### Install

```bash
pip install torch numpy pygame matplotlib
```

### Train All Variants

```bash
# Train all 5 variants sequentially (DQN → DQN+Cong → PPO → PPO+Cong → MAPPO)
python trainer.py --mode all

# Or train individually
python trainer.py --mode rl --episodes 5000              # DQN baseline
python trainer.py --mode improved_rl --episodes 5000      # DQN + congestion
python trainer.py --mode ppo --episodes 10000             # PPO
python trainer.py --mode improved_ppo --episodes 10000    # PPO + congestion
python trainer.py --mode mappo --episodes 10000           # MAPPO (CTDE)

# Background training on GPU server (recommended)
nohup python -u trainer.py --mode all > training.log 2>&1 &
tail -f training.log
```

### Run Demos

```bash
# Pygame visualization
python main.py --mode bfs                                           # BFS baseline
python main.py --mode rl --model checkpoints/rl_best.pt             # DQN
python main.py --mode improved_rl --model checkpoints/improved_rl_best.pt
python main.py --mode ppo --model checkpoints/ppo_best.pt           # PPO
python main.py --mode improved_ppo --model checkpoints/improved_ppo_best.pt
python main.py --mode mappo --model checkpoints/mappo_best.pt       # MAPPO

# With failure injection (robot 1 fails at step 30, recovers at 50)
python main.py --mode rl --model checkpoints/rl_best.pt --failure
```

### Compare All Algorithms

```bash
# Headless comparison (10 runs each, 6 methods)
python main.py --mode compare --runs 10

# Generate plots
python plot_results.py
```

Outputs:
- `results/comparison.json` — raw metrics
- `results/training_curves.png` — reward, completion, collisions, exploration curves
- `results/comparison_bars.png` — bar chart comparing all methods

---

## ROS2 + Gazebo Deployment

### Prerequisites

```bash
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description
export TURTLEBOT3_MODEL=burger
```

### Build & Run

```bash
cd multi_agent_warehouse/ros2_ws
colcon build --packages-select warehouse_multi_agent
source install/setup.bash

# BFS mode
ros2 launch warehouse_multi_agent warehouse.launch.py mode:=bfs

# RL mode (with trained model)
ros2 launch warehouse_multi_agent warehouse.launch.py \
    mode:=rl model_path:=/path/to/checkpoints/rl_best.pt

# MAPPO mode
ros2 launch warehouse_multi_agent warehouse.launch.py \
    mode:=mappo model_path:=/path/to/checkpoints/mappo_best.pt
```

### Sim-to-Real Bridge

The ROS2 navigator translates between grid training and Gazebo:
- **Odometry** → normalized state vector (same format as training)
- **Known map** → wall obstacle sensors (matches training grid lookups)
- **Robot ID** → one-hot encoding (matches training)
- **Discrete action** → `/cmd_vel` Twist (linear/angular velocity)

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│              Gazebo Warehouse World               │
│   [Shelf] [Aisle] [Shelf]  ...  [Drop Zone]     │
│      🟢 R0      🔵 R1      🔴 R2                │
│              ⭐ Objects × 5                      │
└──────────────┬───────────────────────────────────┘
               │ /robotX/odom
               ▼
┌──────────────────────────────────────────────────┐
│           Task Coordinator Node                   │
│  • Tracks all robot positions                     │
│  • Assigns objects (greedy + congestion)           │
│  • Detects failures → reassigns                   │
└──────────────┬───────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Nav R0 │ │ Nav R1 │ │ Nav R2 │   ← BFS / DQN / PPO / MAPPO
│ /cmd_vel│ │ /cmd_vel│ │ /cmd_vel│
└────────┘ └────────┘ └────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│           Metrics Logger Node                     │
│  • Tracks collisions, pickups, deliveries         │
│  • Logs to CSV for analysis                       │
└──────────────────────────────────────────────────┘
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Task Completion Rate** | % of 5 objects delivered (target: 100%) |
| **Collisions** | Robot-robot overlaps per episode (target: 0) |
| **Steps** | Time to complete all deliveries (lower = better) |
| **Reward** | Total reward across all robots (higher = better) |

---

## Features

- ✅ **3 collaborative robots** with dynamic task assignment
- ✅ **5 algorithms**: BFS, DQN, DQN+Congestion, PPO, PPO+Congestion, MAPPO
- ✅ **Congestion-aware navigation** — proximity penalty forces robots to spread out
- ✅ **Robot recycling** — robots pick up multiple objects per episode
- ✅ **Failure handling** — stuck detection → task reassignment to remaining robots
- ✅ **CTDE (MAPPO)** — centralized critic for coordination, decentralized execution
- ✅ **Priority-based collision resolution** with swap detection
- ✅ **Dual deployment**: Pygame (fast training) + Gazebo (realistic demo)
- ✅ **Metrics & plots**: Training curves + algorithm comparison charts
