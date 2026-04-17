# Multi-Agent Warehouse: Collaborative Task Execution

3 robots collaboratively collect objects and deliver them to a drop zone in a warehouse, using **BFS**, **RL**, or **Congestion-Aware RL** navigation — with comparison metrics.

## Project Structure

```
multi_agent_warehouse/
├── 🎮 Pygame Simulation (RL training — fast)
│   ├── config.py              # Grid layout + RL hyperparameters
│   ├── warehouse_env.py       # Multi-agent environment
│   ├── pathfinding.py         # BFS / A*
│   ├── task_allocator.py      # Greedy + congestion-aware allocation
│   ├── dqn_agent.py           # Dueling Double DQN
│   ├── trainer.py             # Train RL agents
│   ├── main.py                # Run simulation + comparison
│   ├── visualizer.py          # Pygame real-time visualization
│   └── plot_results.py        # Generate comparison plots
│
└── 🤖 ROS2 + Gazebo (deployment — realistic)
    └── ros2_ws/src/warehouse_multi_agent/
        ├── worlds/warehouse.world     # Gazebo warehouse (shelves, objects, drop zone)
        ├── launch/warehouse.launch.py # Spawns 3 TurtleBot3 + all nodes
        ├── scripts/
        │   ├── task_coordinator.py    # Central task allocation + failure handling
        │   ├── rl_navigator.py        # RL bridge: trained model → cmd_vel
        │   ├── bfs_navigator.py       # BFS baseline navigator
        │   └── metrics_logger.py      # Collision/pickup/delivery CSV logging
        └── config/warehouse_config.yaml
```

---

## Quick Start

### 1. Train RL (on any machine — no ROS needed)

```bash
cd multi_agent_warehouse

# Install dependencies
pip install torch numpy pygame matplotlib

# Train basic RL
python trainer.py --mode rl --episodes 1500

# Train improved RL (congestion-aware)
python trainer.py --mode improved_rl --episodes 1500

# Run Pygame demo (BFS baseline)
python main.py --mode bfs

# Run RL demo
python main.py --mode rl --model checkpoints/rl_best.pt

# Run comparison (headless, 10 runs each)
python main.py --mode compare --runs 10

# Generate plots
python plot_results.py
```

### 2. Deploy on ROS2 Humble + Gazebo (work computer)

```bash
# Prerequisites
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-description
export TURTLEBOT3_MODEL=burger

# Build
cd multi_agent_warehouse/ros2_ws
colcon build --packages-select warehouse_multi_agent
source install/setup.bash

# Run BFS mode
ros2 launch warehouse_multi_agent warehouse.launch.py mode:=bfs

# Run RL mode (with trained model)
ros2 launch warehouse_multi_agent warehouse.launch.py \
    mode:=rl model_path:=/path/to/checkpoints/rl_best.pt

# Run improved RL mode
ros2 launch warehouse_multi_agent warehouse.launch.py \
    mode:=improved_rl model_path:=/path/to/checkpoints/improved_rl_best.pt
```

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
│  • Publishes /coordinator/assignments             │
└──────────────┬───────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Nav R0 │ │ Nav R1 │ │ Nav R2 │   ← BFS or RL navigator
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

## Algorithm Comparison

| Method | Description |
|--------|-------------|
| **BFS (Baseline)** | Shortest path on grid, priority-based collision avoidance |
| **RL (DQN)** | Learned navigation policy, 12-dim state, 5 actions |
| **Improved RL** | DQN + congestion awareness (14-dim state), penalizes crowding |

### Key Metrics
- **Task Completion Rate** — % of objects delivered
- **Collisions** — robot-robot overlaps
- **Steps/Time** — time to complete all deliveries
- **Distance Traveled** — total path efficiency

---

## Features

- ✅ **3 collaborative robots** with role assignment
- ✅ **Collision-free navigation** (priority-based + learned avoidance)
- ✅ **Dynamic task allocation** (greedy + congestion-aware)
- ✅ **Failure handling** (stuck detection → task reassignment)
- ✅ **BFS vs RL vs Improved RL** comparison
- ✅ **Warehouse environment** (shelving racks, aisles, drop zone)
- ✅ **Metrics logging** (CSV + plots)
- ✅ **Dual deployment**: Pygame (training) + Gazebo (demo)
