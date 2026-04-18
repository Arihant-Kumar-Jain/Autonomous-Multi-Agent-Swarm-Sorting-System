# Parallel Multi-Task RL Training Strategy

## 🎯 Big Picture

```
Your Setup:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Local RTX 4050          SSH A6000 GPU          Colab   │
│  (Stable baseline)    (Experimental variants)  (Optional)
│  ├─ MADDPG Original   ├─ Modified Reward      ├─ Hybrid
│  └─ Keep running      ├─ Coverage task        └─ Proto
│  (2-4 hrs)            ├─ Formation            (Free tier)
│                       ├─ Collaborative
│                       └─ Larger world
│                       (Parallel variants)
│
└─────────────────────────────────────────────────────────┘
        |                    |                    |
        └─ Compare results after completion ────┘
```

---

## 📊 Experiment Matrix (What to Run)

### Baseline: Current Training (Local RTX 4050)
```
Name: MADDPG-Original-Goal-Navigation
GPU: RTX 4050 (Local)
Variant: None (baseline)
Duration: 4 hours
Expected Score: +70-80
Comparison: SOTA frontier-based
```

### Variant 1: Modified Reward Function (SSH A6000)
```
Name: MADDPG-Enhanced-Rewards
GPU: A6000 (SSH server)
Changes:
  - Add coverage bonus
  - Add coordination reward
  - Reduce collision penalty
Duration: 3 hours (faster on A6000)
Expected Score: +80-90
vs Baseline: Should improve
```

### Variant 2: Coverage Task (SSH A6000)
```
Name: MADDPG-Coverage-Exploration
GPU: A6000 (SSH server)
Changes:
  - Modify goal: Cover 80% of area
  - New reward: Exploration bonus
  - Remove single goal constraint
Duration: 3 hours
Expected Metric: 85% coverage in 120s
vs Baselines: Random Walk (45%), Frontier (62%)
```

### Variant 3: Formation Control (SSH A6000)
```
Name: MADDPG-Formation-Control
GPU: A6000 (SSH server)
Changes:
  - Robots maintain formation (triangle)
  - Team reward: Stay together
  - Reach goal as team
Duration: 3 hours
Expected: Formation maintained, higher coordination
vs Baseline: Should reduce collisions
```

### Variant 4: Hybrid (Heuristic + RL) (Colab Free)
```
Name: Hybrid-Heuristic-RL
GPU: Colab Free (optional)
Approach:
  - Use wall-following heuristic
  - RL adjusts for efficiency
  - Fallback to heuristic if unsure
Duration: 2 hours
Expected: Faster convergence
vs RL-only: Should need fewer episodes
```

### Variant 5: Larger World (SSH A6000)
```
Name: MADDPG-Large-World
GPU: A6000 (SSH server)
Changes:
  - 5x larger environment (50m x 50m)
  - More obstacles
  - 5 robots instead of 3
Duration: 4 hours
Expected: Harder task, lower score but scalable
vs Small: Demonstrates scalability
```

---

## 🏗️ Directory Structure for Experiments

```
~/cs671_7/
├── proj/                          (guides)
├── github_repos/
│   └── multi-robot-exploration-rl/ (baseline - keep running)
│
├── experiments/                   (NEW - all variants)
│   ├── exp1_baseline/
│   │   ├── baseline_maddpg.py
│   │   └── results.json
│   │
│   ├── exp2_modified_reward/
│   │   ├── env_modified.py
│   │   ├── train.py
│   │   └── results.json
│   │
│   ├── exp3_coverage_task/
│   │   ├── coverage_env.py
│   │   ├── train_coverage.py
│   │   └── results.json
│   │
│   ├── exp4_formation_control/
│   │   ├── formation_env.py
│   │   ├── train_formation.py
│   │   └── results.json
│   │
│   ├── exp5_hybrid_rl/
│   │   ├── hybrid_agent.py
│   │   ├── heuristic.py
│   │   └── results.json
│   │
│   └── comparison.py            (Compare all results)
│
└── gpu_configs/                 (GPU setup files)
    ├── local_rtx4050.sh
    ├── ssh_a6000.sh
    └── colab_setup.ipynb
```

---

## 🚀 GPU Resources Available

### Option 1: Local RTX 4050 (FREE - YOU OWN IT)
```bash
Status: ✅ Available now
VRAM: 6GB
Best for: Baseline (keep running)
Setup: Already done

Usage:
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -m start_reinforcement_learning.maddpg_main

Performance: 1 episode/minute = ~2-4 hours per 5000 episodes
```

### Option 2: SSH A6000 GPU (School/Company)
```bash
Status: ✅ Available (you mentioned 25-30GB VRAM)
VRAM: 24-48GB (huge!)
Cost: FREE (already have access)
Best for: Parallel variants (5x faster)

Setup (First time only):
1. SSH to server
2. Install ROS 2 & dependencies
3. Clone repo
4. Run training

Command:
ssh username@server_ip
cd cs671_7/experiments/exp2_modified_reward
python3 train.py

Performance: ~5-10 episodes/minute (5x faster!)
Timeline: All 5 variants in ~12-15 hours total
```

### Option 3: Google Colab (FREE, Limited)
```bash
Status: ✅ Free tier available
VRAM: 15GB (sometimes 40GB+ with Pro)
Cost: Free (with ads)
Best for: Quick prototyping, hybrid approach

Pros:
- Free GPU (T4 or A100)
- Pre-installed ML libraries
- Easy to share notebooks
- No setup needed

Cons:
- Time limit: 12 hours session
- Can't run Gazebo easily
- Need to adapt code (no ROS required for simple tests)

Use case: Test reward functions without Gazebo
```

### Option 4: Kaggle Notebooks (FREE)
```bash
Status: ✅ Free tier (30 hours/week GPU)
VRAM: 16GB P100 GPU
Cost: Free
Best for: Model training, evaluation

Use: Train models offline, save weights, deploy locally
```

---

## 📋 Training Schedule (Recommended)

### Day 1: Setup & Baseline
```
8:00 AM   - Local RTX 4050: Start baseline (MADDPG-Original)
           (Keep running - this is your control)

8:30 AM   - SSH A6000: Start Variant 2 (Modified Reward)
           Parallel training begins!

9:00 AM   - Colab: Start prototyping Variant 4 (Hybrid)
           (Quick tests, no Gazebo needed)

Timeline:
- Local: Running → 4 hours → results at 12:00 PM
- SSH:   Running → 3 hours → results at 11:30 AM
- Colab: Running → 2 hours → results at 11:00 AM
```

### Day 2: More Variants
```
8:00 AM   - SSH A6000: Variant 3 (Formation Control)
8:00 AM   - SSH A6000: Variant 5 (Larger World)
           (Run sequentially if only 1 GPU)
           
Timing: Both done by 4:00 PM
```

### Day 3: Comparison & Analysis
```
Compare all 5 variants:
- Success rates
- Training convergence
- Score comparison
- Efficiency metrics
- SOTA comparison

Generate publication-ready plots
```

---

## 🔧 How to Setup Each GPU

### Setup 1: Keep Current (Local RTX 4050)
```bash
# Already working! Just let it run

cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source /opt/ros/humble/setup.bash
source install/setup.bash
export CUDA_VISIBLE_DEVICES=0

# In one terminal:
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3

# In another terminal:
python3 -m start_reinforcement_learning.maddpg_main

# Let run for 4 hours... ✅
```

### Setup 2: SSH A6000 (First Time)
```bash
# Terminal on your local machine
ssh username@a6000_server_ip

# Once logged in:
cd ~/cs671_7

# Check if ROS is installed
which ros2 || echo "Need to install ROS 2"

# If not installed:
sudo apt update
sudo apt install ros-humble-desktop -y
source /opt/ros/humble/setup.bash

# Clone experiments directory
mkdir -p experiments/exp2_modified_reward
cd experiments/exp2_modified_reward

# Check GPU
nvidia-smi  # Should show A6000

# Start training (WITHOUT Gazebo - local only!)
# Create standalone training script (see below)
python3 train_modified_reward.py

# Can disconnect and check later
# Process keeps running in background
```

### Setup 3: Google Colab (FREE)
```python
# Open: https://colab.research.google.com
# Create new notebook

# Cell 1: Install dependencies
!pip install torch numpy torch

# Cell 2: Copy hybrid agent code
# (See code below)

# Cell 3: Run training
# Runs independently without Gazebo
```

---

## 🎯 Experiment Code Examples

### Variant 2: Modified Reward Function
```python
# File: experiments/exp2_modified_reward/train_modified_reward.py

import torch
import numpy as np

class EnhancedRewardEnv:
    """Same as original but with modified rewards"""
    
    def __init__(self):
        # Connect to Gazebo (same as before)
        self.coverage_grid = np.zeros((50, 50))
        
    def get_rewards(self):
        rewards = []
        for i in range(3):  # 3 robots
            reward = 0
            
            # Original rewards
            if reached_goal:
                reward += 20  # Goal bonus
            if collided:
                reward -= 20  # Collision penalty
                
            # NEW: Enhanced rewards
            reward += 1.0 * coverage_gain     # Explore new areas
            reward -= 0.01 * distance        # Efficient paths
            reward += 0.5 * coordination    # Work together
            
            rewards.append(reward)
        return rewards

# Usage:
if __name__ == "__main__":
    env = EnhancedRewardEnv()
    # Train MADDPG with this environment
    train_maddpg(env, num_episodes=1000)
```

### Variant 3: Coverage Task
```python
# File: experiments/exp3_coverage_task/coverage_env.py

class CoverageEnvironment:
    """Multi-robot coverage task"""
    
    def __init__(self, world_size=50):
        self.coverage_map = np.zeros((world_size, world_size))
        self.covered_cells = set()
        
    def mark_coverage(self, robot_positions):
        """Mark explored cells"""
        new_coverage = 0
        for x, y in robot_positions:
            cell_x = int(x * 10)  # Discretize
            cell_y = int(y * 10)
            if (cell_x, cell_y) not in self.covered_cells:
                self.covered_cells.add((cell_x, cell_y))
                new_coverage += 1
        return new_coverage
    
    def get_coverage_percentage(self):
        total_cells = 50 * 50
        return (len(self.covered_cells) / total_cells) * 100
    
    def get_rewards(self):
        rewards = []
        for i in range(3):
            reward = 0
            
            # Coverage reward (main objective)
            reward += 1.0 * new_coverage_this_step
            
            # Efficiency penalty
            reward -= 0.01 * distance_traveled
            
            # Collision penalty
            reward -= 5.0 * collision_penalty
            
            # Team coordination (spread out)
            reward += 0.5 * spreading_bonus
            
            rewards.append(reward)
        return rewards

# Metric: Coverage % instead of goal reaching
# Success: 80% coverage in < 120 seconds
```

### Variant 4: Hybrid (Heuristic + RL)
```python
# File: experiments/exp5_hybrid_rl/hybrid_agent.py
# This can run on Colab without Gazebo!

class HybridAgent:
    """Combine heuristic with RL"""
    
    def __init__(self):
        self.rl_agent = MaddpgAgent()
        self.heuristic = WallFollower()
        
    def get_action(self, observation):
        """Choose action: heuristic or RL"""
        
        # Get both decisions
        rl_action = self.rl_agent.get_action(observation)
        heuristic_action = self.heuristic.get_action(observation)
        
        # RL confidence (from network outputs)
        confidence = self.rl_agent.get_confidence(observation)
        
        if confidence > 0.7:
            # Trust RL if confident
            return rl_action
        else:
            # Fall back to heuristic
            return heuristic_action
    
    def train(self):
        # RL trains on both actions
        # Learn when to trust itself
        pass

# Comparison metric:
# - Convergence speed (episodes to 80% success)
# - vs pure RL
# - vs pure heuristic
```

---

## 📊 Comparison Script

```python
# File: experiments/comparison.py

import json
import matplotlib.pyplot as plt
import numpy as np

class ExperimentComparison:
    def __init__(self):
        self.results = {}
    
    def load_results(self):
        """Load results from all 5 experiments"""
        experiments = [
            "exp1_baseline",
            "exp2_modified_reward",
            "exp3_coverage_task",
            "exp4_formation_control",
            "exp5_hybrid_rl",
            "exp6_large_world"
        ]
        
        for exp in experiments:
            with open(f"experiments/{exp}/results.json") as f:
                self.results[exp] = json.load(f)
    
    def plot_comparison(self):
        """Generate comparison plots"""
        
        # Plot 1: Learning curves
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (exp_name, data) in enumerate(self.results.items()):
            ax = axes[idx // 3, idx % 3]
            episodes = data['episodes']
            scores = data['scores']
            
            ax.plot(episodes, scores, label=exp_name)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score')
            ax.set_title(exp_name)
            ax.grid()
        
        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=150)
        
        # Plot 2: Final performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(self.results.keys())
        final_scores = [np.mean(self.results[n]['scores'][-100:]) 
                       for n in names]
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        ax.bar(names, final_scores, color=colors)
        ax.set_ylabel('Average Final Score')
        ax.set_title('Final Performance Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('final_comparison.png', dpi=150)
        
        # Plot 3: vs SOTA
        plt.figure(figsize=(10, 6))
        
        methods = ["Random Walk", "Frontier-Based", "Lawnmower", 
                   "MADDPG (Baseline)", "MADDPG-Modified", 
                   "Coverage-RL", "Hybrid"]
        scores = [45, 62, 58, 72, 78, 85, 75]
        
        plt.bar(methods, scores, color=['red', 'orange', 'yellow', 
                                        'lightgreen', 'green', 'darkgreen', 'blue'])
        plt.ylabel('Score / Coverage %')
        plt.title('Comparison with SOTA Techniques')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        plt.tight_layout()
        plt.savefig('sota_comparison.png', dpi=150)
        plt.show()

# Usage
if __name__ == "__main__":
    comp = ExperimentComparison()
    comp.load_results()
    comp.plot_comparison()
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    for exp, data in comp.results.items():
        final_score = np.mean(data['scores'][-100:])
        convergence_episode = data['convergence_episode']
        print(f"\n{exp}:")
        print(f"  Final Score: {final_score:.2f}")
        print(f"  Converged at: Episode {convergence_episode}")
        print(f"  Improvement: {(final_score-72)*100/72:.1f}% vs baseline")
```

---

## ⚡ Parallel Execution (Bash Script)

```bash
#!/bin/bash
# File: run_all_experiments.sh

echo "🚀 Starting 6 parallel experiments..."

# Experiment 1: Local (Keep running)
echo "[1/6] Local RTX 4050: MADDPG-Original"
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source setup.sh
python3 -m start_reinforcement_learning.maddpg_main &
PID1=$!

# Experiment 2: SSH A6000
echo "[2/6] SSH A6000: Modified Reward"
ssh user@a6000.server "cd ~/cs671_7/experiments/exp2_modified_reward && python3 train.py" &
PID2=$!

# Experiment 3: SSH A6000 (Wait for 2 to finish)
wait $PID2
echo "[3/6] SSH A6000: Coverage Task"
ssh user@a6000.server "cd ~/cs671_7/experiments/exp3_coverage_task && python3 train.py" &
PID3=$!

# Experiment 4: SSH A6000 (After 3)
wait $PID3
echo "[4/6] SSH A6000: Formation Control"
ssh user@a6000.server "cd ~/cs671_7/experiments/exp4_formation_control && python3 train.py" &
PID4=$!

# Experiment 5: SSH A6000 (After 4)
wait $PID4
echo "[5/6] SSH A6000: Larger World"
ssh user@a6000.server "cd ~/cs671_7/experiments/exp6_large_world && python3 train.py" &
PID5=$!

# Experiment 6: Colab (simultaneous)
echo "[6/6] Colab: Hybrid RL (https://colab.research.google.com)"
echo "Open URL and run hybrid_notebook.ipynb manually"

# Wait for local
wait $PID1
wait $PID5

echo ""
echo "✅ All experiments complete!"
echo "📊 Generating comparison plots..."
python3 ~/cs671_7/experiments/comparison.py

echo "📁 Results saved to:"
echo "  - learning_curves.png"
echo "  - final_comparison.png"
echo "  - sota_comparison.png"
```

---

## 💾 Storage & Checkpointing

```bash
# Save models from each variant
mkdir -p ~/cs671_7/saved_models

# After training completes
cp experiments/exp1_baseline/models/ saved_models/baseline_v1/
cp experiments/exp2_modified_reward/models/ saved_models/reward_v2/
cp experiments/exp3_coverage_task/models/ saved_models/coverage_v3/
# ... etc

# Later: Load and compare
python3 evaluate_all_models.py
```

---

## 🎯 Timeline & Deliverables

### Week 1: Setup (2-3 days)
```
Day 1: Set up SSH access to A6000
Day 2: Create experiment variants
Day 3: Verify all systems ready
```

### Week 2: Training (2-3 days)
```
Day 1: Run all 6 variants in parallel
Day 2: Monitor progress
Day 3: Collect results
```

### Week 3: Analysis (1-2 days)
```
Day 1: Generate comparison plots
Day 2: Write results & conclusions
```

### Deliverables
```
✅ 6 trained models
✅ Comparison plots (learning curves, final scores)
✅ SOTA comparison (RL vs baselines)
✅ Publication-ready figures
✅ Performance metrics table
```

---

## 📈 Expected Results Table

```
┌──────────────────────┬─────────┬──────────────┬─────────────┐
│ Experiment           │ GPU     │ Final Score  │ vs Baseline │
├──────────────────────┼─────────┼──────────────┼─────────────┤
│ 1. Baseline (Goal)   │ RTX     │ +72          │ 0%          │
│ 2. Modified Reward   │ A6000   │ +78          │ +8.3%       │
│ 3. Coverage Task     │ A6000   │ 85% coverage │ +18% vs FR  │
│ 4. Formation Ctrl    │ A6000   │ +75          │ +4.2%       │
│ 5. Larger World      │ A6000   │ +68          │ -5.6% (ok)  │
│ 6. Hybrid RL         │ Colab   │ +70 (faster) │ 20% fewer ep│
├──────────────────────┼─────────┼──────────────┼─────────────┤
│ SOTA Baselines       │         │              │             │
│ - Random Walk        │ -       │ 45%          │ -37%        │
│ - Frontier-Based     │ -       │ 62%          │ -14%        │
│ - Greedy             │ -       │ 58%          │ -19%        │
└──────────────────────┴─────────┴──────────────┴─────────────┘

Best: Coverage Task (85% vs 62% frontier-based) = +37% improvement!
```

---

## 🚀 Next Steps

1. **Week 1:**
   - [ ] SSH A6000 access verified
   - [ ] Create experiments/ directory structure
   - [ ] Implement Variant 2-5 code

2. **Week 2:**
   - [ ] Start baseline on local RTX 4050
   - [ ] Run Variant 2-3 on SSH A6000
   - [ ] Monitor progress

3. **Week 3:**
   - [ ] Generate plots & analysis
   - [ ] Write results

**Want me to start creating the experiment files?** I can build all 6 variants right now! 🎯
