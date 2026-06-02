# Multi-Agent Warehouse: Collaborative Task Execution

3 autonomous robots collaboratively collect 10 objects and deliver them to a drop zone in a 30×30 warehouse with **partial observability**, **exploration**, and **SLAM-ready** training — using **BFS**, **DQN**, **PPO**, or **MAPPO** navigation.

## Project Structure

```
multi_agent_warehouse/
├── 🎮 Pygame Simulation (RL training — fast)
│   ├── config.py              # 30×30 grid, rewards, all hyperparameters
│   ├── warehouse_env.py       # Multi-agent env with exploration + discovery
│   ├── pathfinding.py         # BFS / A* pathfinding
│   ├── task_allocator.py      # Greedy + congestion-aware task allocation
│   ├── dqn_agent.py           # Dueling Double DQN agent
│   ├── ppo_agent.py           # PPO (Proximal Policy Optimization) agent
│   ├── mappo_agent.py         # MAPPO (Multi-Agent PPO, CTDE) agent
│   ├── trainer.py             # Train all RL variants (versioned logs)
│   ├── main.py                # Run simulation + 6-way comparison
│   ├── visualizer.py          # Pygame real-time visualization
│   └── plot_results.py        # Generate comparison plots
│
├── 📊 Outputs
│   ├── checkpoints/<variant>/  # Versioned model saves per variant
│   │   ├── best.pt            # Best model checkpoint
│   │   ├── latest.pt          # Latest model checkpoint
│   │   ├── training_log.json  # Training metrics
│   │   └── history/           # Archived previous runs (timestamped)
│   └── results/               # Reports, CSVs, plots
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

### Warehouse Grid (30×30)

- **6 shelf zones** with narrow 1-cell aisles and bottleneck corridors
- **10 randomized objects** scattered across the warehouse (new positions each episode)
- **2×10 drop zone** at the bottom center
- **3 robots** spawning at the top row, spread across the map
- **800 max steps** per episode

### Partial Observability & Exploration

Objects are **hidden** until a robot discovers them within `SENSOR_RANGE` (3 cells). Robots must **explore** the warehouse to find objects before they can collect them. Only the drop zone location is known from the start.

- **Shared exploration map**: 30×30 binary grid tracking which cells any robot has seen
- **Discovery mechanic**: objects revealed only when within sensor range
- **Frontier reward**: bonus for exploring new cells (decays over the episode to transition from exploration → exploitation)
- **Communication**: robots share discovered object locations (simulating ROS2 topic sharing)

### Observation (State Vector — Frame-Stacked ×3)

Each robot sees a **frame-stacked observation** (last 3 timesteps concatenated for temporal memory):

| Raw Dims | Feature | Source |
|----------|---------|--------|
| `[0-1]` | Direction to goal (Δrow, Δcol, normalized) | Grid position |
| `[2]` | Distance to goal (normalized) | Manhattan distance |
| `[3]` | Carrying object (0/1) | Boolean flag |
| `[4]` | **Has target** (0=exploring, 1=navigating) | Discovery state |
| `[5-8]` | Relative position of 2 other robots | Grid positions |
| `[9-12]` | Obstacle sensors (up/down/left/right, ray-cast) | LiDAR-like |
| `[13]` | **Fraction of map explored** (0→1) | Shared map |
| `[14-16]` | Robot identity (one-hot: 3 robots) | Robot ID |
| `[17-18]` | Congestion + local density *(congestion modes)* | Nearby robots |

- **Base state**: 17 raw × 3 frames = **51 dimensions**
- **Congestion-aware**: 19 raw × 3 frames = **57 dimensions**

### Reward Function

| Signal | Value | Purpose |
|--------|-------|---------|
| Deliver to drop zone | +20.0 | Main goal |
| Pick up object | +10.0 | Sub-goal |
| **Discover hidden object** | **+5.0** | Incentivize exploration |
| **Frontier (new cell explored)** | **+0.3 × cells × decay** | SLAM-like coverage |
| Move closer to target | +0.5 | Reward shaping |
| Move farther from target | −0.3 | Discourage wandering |
| Per-step penalty | −0.05 | Encourage efficiency |
| Wait action | −0.1 | Discourage idling |
| Hit wall | −2.0 | Boundary awareness |
| Robot-robot collision | −10.0 | Collision avoidance |
| **Proximity penalty** | −0.3/nearby robot | Spread out *(congestion modes)* |

**Frontier decay**: Exploration reward scales from 1.0→0.0 over first 40% of episode steps, naturally transitioning agents from explore → exploit.

**Mixed rewards (MAPPO only)**: `0.7 × individual + 0.3 × team_avg` — encourages cooperative behavior.

### Actions

5 discrete actions: `UP`, `DOWN`, `LEFT`, `RIGHT`, `WAIT`

---

## Algorithms

### 1. BFS (Baseline)

Shortest-path grid search with priority-based collision avoidance. Random exploration fallback when no target is assigned (no discovered objects).

### 2. DQN (Dueling Double DQN)

- **Architecture**: Shared feature extractor (128-dim) → Value/Advantage streams
- **Key features**: Experience replay (100K), Double DQN, dueling architecture
- **State**: 51-dim (base) or 57-dim (congestion-aware), frame-stacked

### 3. PPO (Proximal Policy Optimization)

- **Architecture**: Shared backbone (256-dim, LayerNorm) → Actor + Critic heads
- **Key features**: Clipped surrogate (±0.2), GAE (λ=0.95), entropy bonus (0.05)
- **Update**: On-policy — 1024-step rollouts, 10 epochs of mini-batch updates

### 4. DQN + Congestion / PPO + Congestion

Same as above with 57-dim state (adds congestion + density) and proximity penalty.

### 5. MAPPO (Multi-Agent PPO — CTDE) ⭐

**Centralized Training, Decentralized Execution** with cooperative improvements:

| Feature | Detail |
|---------|--------|
| **Centralized Critic** | Sees all 3 robots' observations (171-dim global state) |
| **Shared Actor** | Each robot uses only its own 57-dim observation |
| **Mixed Rewards** | 70% individual + 30% team average |
| **Frame Stacking** | 3-frame memory for temporal patterns |
| **Exploration Map** | Shared grid of discovered cells |

```
Training:                           Deployment (Gazebo-ready):
┌─────────────────────┐            ┌──────────────────┐
│ Centralized Critic  │            │  Actor only      │
│ V(obs_0 ⊕ obs_1 ⊕  │            │  obs → action    │
│   obs_2) → value    │            │  (per robot)     │
├─────────────────────┤            │  Same as PPO     │
│ Shared Actor        │            │  deployment      │
│ obs_i → action_i    │            └──────────────────┘
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
# Train all 5 variants sequentially
python trainer.py --mode all

# Or train individually (recommended: run in parallel on GPU server)
python -u trainer.py --mode mappo --episodes 10000 > results/mappo.log 2>&1 &
python -u trainer.py --mode rl --episodes 5000 > results/rl.log 2>&1 &
python -u trainer.py --mode improved_rl --episodes 5000 > results/improved_rl.log 2>&1 &
python -u trainer.py --mode ppo --episodes 10000 > results/ppo.log 2>&1 &
python -u trainer.py --mode improved_ppo --episodes 10000 > results/improved_ppo.log 2>&1 &

tail -f results/*.log   # monitor all
```

Training logs are auto-archived on restart to `checkpoints/<variant>/history/<timestamp>/`.

### Run Demos

```bash
# Pygame visualization
python main.py --mode bfs
python main.py --mode rl --model checkpoints/rl/best.pt
python main.py --mode ppo --model checkpoints/ppo/best.pt
python main.py --mode mappo --model checkpoints/mappo/best.pt

# With failure injection
python main.py --mode mappo --model checkpoints/mappo/best.pt --failure
```

### Compare All Algorithms

```bash
python main.py --mode compare --runs 10
python plot_results.py
```

Outputs:
- `results/training_summary.txt` — report-ready comparison table
- `results/<variant>_curve.csv` — per-variant CSV for LaTeX pgfplots
- `results/training_curves.png` — reward, completion, collisions curves
- `results/comparison_bars.png` — bar chart comparing all methods

---

## Sim-to-Gazebo Transfer

The environment is designed for direct policy transfer to ROS2 + Gazebo:

| Pygame Feature | Gazebo Equivalent |
|----------------|-------------------|
| Grid obstacle sensors (ray-cast) | LiDAR scan |
| Object discovery (SENSOR_RANGE) | YOLO camera detection |
| Shared exploration map | SLAM (GMapping/Cartographer) |
| Frame stacking (3 frames) | Sensor history buffer |
| Relative robot positions | Odometry + TF transforms |
| Discrete actions (5) | cmd_vel (linear/angular) |

### Build & Run (ROS2 Humble)

```bash
cd multi_agent_warehouse/ros2_ws
colcon build --packages-select warehouse_multi_agent
source install/setup.bash
ros2 launch warehouse_multi_agent warehouse.launch.py \
    mode:=mappo model_path:=/path/to/checkpoints/mappo/best.pt
```

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Task Completion Rate** | % of 10 objects delivered (target: 100%) |
| **Collisions** | Robot-robot overlaps per episode (target: 0) |
| **Steps** | Time to complete all deliveries (lower = better) |
| **Map Explored** | % of walkable cells discovered (target: >90%) |
| **Reward** | Total reward across all robots (higher = better) |

---

## Features

- ✅ **30×30 complex warehouse** with 6 shelf zones, bottlenecks, narrow aisles
- ✅ **Partial observability** — objects hidden until discovered via exploration
- ✅ **Shared exploration map** with decaying frontier reward
- ✅ **Frame stacking ×3** for temporal memory (LSTM-lite)
- ✅ **6 algorithms**: BFS, DQN, DQN+Congestion, PPO, PPO+Congestion, MAPPO
- ✅ **Mixed cooperative rewards** (MAPPO: 70% individual + 30% team)
- ✅ **Congestion-aware navigation** with proximity penalty
- ✅ **Robot recycling** — robots pick up multiple objects per episode
- ✅ **Failure handling** — stuck detection → task reassignment
- ✅ **CTDE (MAPPO)** — centralized critic, decentralized execution
- ✅ **Versioned model saving** with automatic log archival
- ✅ **Report generation** — training_summary.txt + per-variant CSVs
- ✅ **Gazebo-ready** — state vector maps directly to SLAM/YOLO/LiDAR
