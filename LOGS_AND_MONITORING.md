# 🔍 Detailed Log Locations & Monitoring Guide

This guide explains exactly where every output, log, and result file is saved, and how to access them one by one.

---

## 📁 COMPLETE DIRECTORY MAP

```
/home/aman/cs671_7/
│
├── 📄 EXECUTION_GUIDE.md                    # This file's sibling (step-by-step execution)
├── 📄 QUICK_REFERENCE.md                    # Quick commands
├── 📄 RESEARCH_INFRASTRUCTURE.md            # System overview
├── 📄 LOGS_AND_MONITORING.md               # This file
│
├── 🐍 main_train.py                         # Main orchestrator script
├── 🐍 baselines.py                          # Baseline comparison script
├── 🐍 comparison.py                         # Results analysis script
│
├── 📊 baseline_comparison_TIMESTAMP.json    # ⭐ Classical baseline results
├── 📊 comparison_results.json              # ⭐ RL experiments analysis
├── 📊 comparison_learning_curves.png       # ⭐ Learning curves plot
│
└── 📂 experiments/                          # ⭐ MAIN RESULTS DIRECTORY
    │
    ├── 📂 baseline/                         # Experiment 0: Baseline MADDPG
    │   ├── 📂 20260418_140000/             # Run 1 (timestamp)
    │   │   ├── 📄 README.md                # Task specification
    │   │   ├── 📊 training_scores.json     # ⭐ Main results file
    │   │   ├── 📂 log/                     # ROS 2 training logs
    │   │   ├── 📂 best/                    # Best checkpoint weights
    │   │   │   ├── actor_agent0.pt
    │   │   │   ├── actor_agent1.pt
    │   │   │   ├── critic_agent0.pt
    │   │   │   └── critic_agent1.pt
    │   │   └── 📂 last/                    # Last checkpoint (for resume)
    │   │       ├── actor_agent0.pt
    │   │       ├── actor_agent1.pt
    │   │       ├── critic_agent0.pt
    │   │       └── critic_agent1.pt
    │   │
    │   └── 📂 20260419_090000/             # Run 2 (resumed/continued)
    │       └── (same structure)
    │
    ├── 📂 modified_reward/
    │   └── 📂 20260418_145000/
    │       ├── README.md
    │       ├── training_scores.json
    │       ├── log/
    │       ├── best/
    │       └── last/
    │
    ├── 📂 coverage_task/                   # ⭐ Your novel experiment
    │   └── 📂 20260418_150000/
    │       └── (same structure)
    │
    ├── 📂 frontier_baseline/
    │   └── 📂 20260418_151000/
    │       └── (same structure)
    │
    ├── 📂 hybrid/
    │   └── 📂 20260418_152000/
    │       └── (same structure)
    │
    └── 📂 large_world/
        └── 📂 20260418_153000/
            └── (same structure)
```

---

## 📋 KEY FILES EXPLAINED

### 1️⃣ Classical Baselines Results

**Location:** `/home/aman/cs671_7/baseline_comparison_*.json`

**Created by:** `python3 baselines.py`  
**Update frequency:** Once (at end of execution)  
**Size:** ~5-10 KB

**View it:**
```bash
cat /home/aman/cs671_7/baseline_comparison_*.json | python3 -m json.tool
```

**Content example:**
```json
{
  "frontier_based": {
    "final_coverage": 0.625,
    "avg_coverage": 0.485,
    "episodes_to_convergence": 150,
    "coverages": [0.0, 0.05, 0.10, ..., 0.625]
  },
  "greedy": {
    "final_coverage": 0.583,
    "avg_coverage": 0.421,
    "episodes_to_convergence": 180
  },
  "potential_field": {
    "final_coverage": 0.612,
    "avg_coverage": 0.465,
    "episodes_to_convergence": 160
  }
}
```

**What to look for:**
- frontier_based should be ~62% (classical SOTA)
- All three methods should be in 55-65% range
- greedy typically lowest, frontier typically highest

---

### 2️⃣ Individual Experiment Scores

**Location:** `/home/aman/cs671_7/experiments/EXPERIMENT_NAME/TIMESTAMP/training_scores.json`

**Created by:** Training script (main_train.py)  
**Update frequency:** Every 50 episodes  
**Size:** 500 bytes - 1 MB (depends on num episodes)

**List all:**
```bash
find /home/aman/cs671_7/experiments -name "training_scores.json" -type f
```

**View specific experiment:**
```bash
cat /home/aman/cs671_7/experiments/baseline/*/training_scores.json | python3 -m json.tool
```

**Quick summary (without all scores):**
```bash
python3 << 'EOF'
import json
from pathlib import Path

# Find all experiment results
exp_root = Path("/home/aman/cs671_7/experiments")
for exp_dir in sorted(exp_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        scores_file = run_dir / "training_scores.json"
        if scores_file.exists():
            with open(scores_file) as f:
                data = json.load(f)
            
            print(f"\n{exp_dir.name:20} {run_dir.name}")
            print(f"  Episodes:      {data['num_episodes']}")
            print(f"  Avg Score:     {data['current_avg_score']:.2f}")
            print(f"  Best Score:    {data['best_score']:.2f}")
            print(f"  Status:        {data['status']}")
            print(f"  Last 3 scores: {data['episode_scores'][-3:]}")
EOF
```

**Content example:**
```json
{
  "episode_scores": [-43.67, -55.5, -30.83, ..., -19.83],
  "num_episodes": 401,
  "current_avg_score": -26.09,
  "best_score": 0.0,
  "timestamp": "2026-04-17T23:36:51.052798",
  "map_number": 1,
  "robot_number": 3,
  "status": "training_in_progress"
}
```

**What to look for:**
- num_episodes increases (50, 100, 150, ...)
- best_score improves from 0.0 to positive values
- current_avg_score should trend upward
- Status changes from "training_in_progress" to "complete"

---

### 3️⃣ Checkpoint Weights Files

**Location:** `/home/aman/cs671_7/experiments/EXPERIMENT/TIMESTAMP/best/` and `last/`

**Created by:** Training script (saved every 50 episodes to `last/`, when best score improves to `best/`)  
**Update frequency:** Every 50 episodes (last/), on improvement (best/)  
**Size:** 5-25 MB per file

**Files created:**
- `actor_agent0.pt` - Actor network for robot 0
- `actor_agent1.pt` - Actor network for robot 1
- `critic_agent0.pt` - Critic network for robot 0
- `critic_agent1.pt` - Critic network for robot 1

**View checkpoint info:**
```bash
ls -lh /home/aman/cs671_7/experiments/baseline/*/best/

# Expected output:
# -rw-r--r-- actor_agent0.pt    (7.5M)
# -rw-r--r-- actor_agent1.pt    (7.5M)
# -rw-r--r-- critic_agent0.pt   (15M)
# -rw-r--r-- critic_agent1.pt   (15M)
# Total: ~45 MB
```

**Checkpoint file sizes increase** as training progresses:
- Episode 0-50: Files don't exist yet
- Episode 50: Files created (~45 MB)
- Episode 100-500: File size stable (same model, updated weights)
- Episode 5000: Final checkpoint saved

**Verify checkpoints are being saved:**
```bash
# Watch checkpoints get created
watch -n 10 'ls -lh /home/aman/cs671_7/experiments/baseline/*/best/'
```

---

### 4️⃣ Experiment Task Specifications

**Location:** `/home/aman/cs671_7/experiments/EXPERIMENT/TIMESTAMP/README.md`

**Created by:** main_train.py (auto-generated)  
**Update frequency:** Once (at experiment start)  
**Size:** 2-4 KB

**View it:**
```bash
cat /home/aman/cs671_7/experiments/baseline/*/README.md
```

**Content includes:**
- Task definition
- Expected score
- Modifications from original
- Reward structure
- How to reproduce
- Execution details

---

### 5️⃣ Training Logs

**Location:** `/home/aman/cs671_7/experiments/EXPERIMENT/TIMESTAMP/log/`

**Created by:** ROS 2 output (for local training) or training stdout  
**Update frequency:** Real-time (as training runs)  
**Size:** 1-100 MB (verbose logs)

**View logs (local ROS training):**
```bash
# ROS logs are saved to:
~/.ros/log/

# But also check the experiment log directory:
ls -la /home/aman/cs671_7/experiments/baseline/*/log/
cat /home/aman/cs671_7/experiments/baseline/*/log/*.log
```

**View recent log entries:**
```bash
# Tail the most recent log
tail -50 ~/.ros/log/latest/master.log
```

**Look for these messages in logs:**
- "Episode: 0, Average score: ..." - Training started
- "💾 Saved LAST checkpoint at episode 50" - Checkpoint saved
- "🏆 Saved BEST checkpoint at episode 200 with score..." - Best improved
- "✅ Training complete!" - Finished

---

### 6️⃣ Comparison Results

**Location:** `/home/aman/cs671_7/comparison_results.json` (or custom output name)

**Created by:** `python3 comparison.py`  
**Update frequency:** Once (after running comparison)  
**Size:** 5-20 KB

**View it:**
```bash
cat /home/aman/cs671_7/comparison_results.json | python3 -m json.tool
```

**Content example:**
```json
{
  "generated_at": "2026-04-18T16:30:00",
  "experiments": {
    "baseline": {
      "final_score": 62.4,
      "average_score": -15.2,
      "num_episodes": 5000,
      "convergence_episodes": 1250,
      "improvement_rate": 0.45,
      "status": "complete",
      "path": "/home/aman/cs671_7/experiments/baseline/20260418_140000"
    },
    "coverage_task": {
      "final_score": 85.6,
      "average_score": 5.3,
      "num_episodes": 2500,
      "convergence_episodes": 890,
      "improvement_rate": 0.68,
      "status": "complete"
    }
  }
}
```

---

### 7️⃣ Learning Curves Plot

**Location:** `/home/aman/cs671_7/comparison_learning_curves.png`

**Created by:** `python3 comparison.py` (automatic)  
**Update frequency:** Once (when comparison runs)  
**Size:** 100-500 KB

**View it:**
```bash
# Linux
display /home/aman/cs671_7/comparison_learning_curves.png
xdg-open /home/aman/cs671_7/comparison_learning_curves.png

# macOS
open /home/aman/cs671_7/comparison_learning_curves.png

# Or in VS Code
code /home/aman/cs671_7/comparison_learning_curves.png
```

**Contains 4 subplots:**
1. Episode scores over time (raw)
2. Smoothed curves (100-episode moving average)
3. Final scores comparison (bar chart)
4. Convergence speed (episodes to 80% of best)

---

## 🔄 MONITORING DURING TRAINING

### Real-time Score Monitoring

**While training is running:**

```bash
# Option 1: Watch every 5 seconds (recommended)
watch -n 5 'find /home/aman/cs671_7/experiments -name training_scores.json -exec tail -1 {} \; -print'

# Option 2: Continuous tail
while true; do
  clear
  python3 << 'EOF'
import json
from pathlib import Path
import time

exp_root = Path("/home/aman/cs671_7/experiments")
for exp_dir in sorted(exp_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        scores_file = run_dir / "training_scores.json"
        if scores_file.exists():
            with open(scores_file) as f:
                data = json.load(f)
            
            bar_len = 40
            progress = min(data['num_episodes'] / 5000, 1.0)
            filled = int(bar_len * progress)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f"{exp_dir.name:20} [{bar}] {data['num_episodes']:5d}/5000 | "
                  f"Avg: {data['current_avg_score']:7.2f} | "
                  f"Best: {data['best_score']:7.2f}")
EOF
  sleep 5
done

# Option 3: Simple one-liner
watch -n 10 'python3 -c "import json; from pathlib import Path; [print(f\"{d.name}: {json.load(open(r))[\"num_episodes\"]}/5000 - Best: {json.load(open(r))[\"best_score\"]:.1f}\") for d in Path(\"/home/aman/cs671_7/experiments\").iterdir() if d.is_dir() for r in [next((d.glob(\"*/training_scores.json\")).__next__(), None)] if r]"'
```

### Checkpoint Creation Monitoring

```bash
# Watch for checkpoint files being created/updated
watch -n 10 'echo "=== BEST CHECKPOINTS ===" && ls -lh /home/aman/cs671_7/experiments/*/*/best/*.pt 2>/dev/null | tail -5 && echo && echo "=== LAST CHECKPOINTS ===" && ls -lh /home/aman/cs671_7/experiments/*/*/last/*.pt 2>/dev/null | tail -5'
```

### Episode Count Monitoring

```bash
# Track which episode each experiment is at
while true; do
  clear
  echo "=== EPISODE PROGRESS ==="
  echo "Updated: $(date)"
  echo
  
  python3 << 'EOF'
import json
from pathlib import Path

exp_root = Path("/home/aman/cs671_7/experiments")
for exp_dir in sorted(exp_root.iterdir()):
    if not exp_dir.is_dir():
        continue
    
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        scores_file = run_dir / "training_scores.json"
        if scores_file.exists():
            with open(scores_file) as f:
                data = json.load(f)
            
            ep = data['num_episodes']
            pct = (ep / 5000) * 100
            print(f"  {exp_dir.name:20} {ep:5d} episodes ({pct:5.1f}%)")
EOF
  sleep 5
done
```

---

## 📊 ONE-BY-ONE LOG VERIFICATION

### After Running Baselines

```bash
# Step 1: Check baseline comparison file exists
ls -lh /home/aman/cs671_7/baseline_comparison_*.json
# Expected: 1 file, 5-10 KB

# Step 2: Verify file content
python3 << 'EOF'
import json
with open('/home/aman/cs671_7/baseline_comparison_20260418_*.json') as f:
    data = json.load(f)
print(f"Frontier: {data['frontier_based']['final_coverage']:.1%}")
print(f"Greedy:   {data['greedy']['final_coverage']:.1%}")
print(f"Potential: {data['potential_field']['final_coverage']:.1%}")
EOF
# Expected: ~62%, ~58%, ~61% respectively

# Step 3: Confirm no errors
cat /home/aman/cs671_7/baseline_comparison_*.json | python3 -c "import sys, json; json.load(sys.stdin); print('✅ JSON valid')"
```

### After First Training Experiment

```bash
# Step 1: Check experiment directory created
ls -la /home/aman/cs671_7/experiments/baseline/
# Expected: 1+ subdirectories with timestamps

# Step 2: Verify subdirectory structure
ls -la /home/aman/cs671_7/experiments/baseline/*/
# Expected: README.md, log/, best/, last/, training_scores.json

# Step 3: Check training_scores.json file
head -50 /home/aman/cs671_7/experiments/baseline/*/training_scores.json
# Expected: JSON with episode_scores array

# Step 4: Check checkpoint files
ls -lh /home/aman/cs671_7/experiments/baseline/*/best/
# Expected: 4 .pt files (actor/critic for agents 0-1)

# Step 5: Verify README was generated
cat /home/aman/cs671_7/experiments/baseline/*/README.md | head -20
# Expected: Task definition, expected score, etc.
```

### During Long-Running Training

```bash
# Every 30 minutes, check:

# 1. Episode count increasing
tail -1 /home/aman/cs671_7/experiments/baseline/*/training_scores.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'Episodes: {d[\"num_episodes\"]}')"

# 2. Best score improving
tail -1 /home/aman/cs671_7/experiments/baseline/*/training_scores.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f'Best: {d[\"best_score\"]:.2f}')"

# 3. Checkpoints being updated
stat /home/aman/cs671_7/experiments/baseline/*/best/actor_agent0.pt

# 4. No errors in logs
grep -i "error\|exception\|failed" ~/.ros/log/latest/*.log 2>/dev/null | wc -l
# Expected: 0 lines
```

### After All Training Complete

```bash
# Step 1: Run comparison analysis
cd /home/aman/cs671_7 && python3 comparison.py --root ./experiments

# Step 2: Verify comparison outputs
ls -lh comparison_results.json comparison_learning_curves.png
# Expected: 2 files created

# Step 3: Check final results
cat comparison_results.json | python3 -m json.tool | head -50

# Step 4: View final rankings
python3 << 'EOF'
import json
with open('comparison_results.json') as f:
    data = json.load(f)

sorted_exp = sorted(data['experiments'].items(),
                   key=lambda x: x[1]['final_score'], reverse=True)

print("🏆 FINAL RANKINGS")
print("-" * 50)
for rank, (name, metrics) in enumerate(sorted_exp, 1):
    print(f"{rank}. {name:20} Score: {metrics['final_score']:7.2f}")
EOF
```

---

## 🎯 QUICK LOG ACCESS COMMANDS

### Get everything at once

```bash
# Create a comprehensive report
python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

print("\n" + "="*60)
print("🔍 COMPLETE PROJECT STATUS REPORT")
print("="*60)
print(f"Generated: {datetime.now().isoformat()}\n")

# 1. Baselines
baseline_file = list(Path(".").glob("baseline_comparison_*.json"))
if baseline_file:
    with open(baseline_file[0]) as f:
        data = json.load(f)
    print("📊 CLASSICAL BASELINES:")
    for method, metrics in data.items():
        print(f"  {method:20} Coverage: {metrics['final_coverage']:.1%}")

# 2. RL Experiments
print("\n📈 RL EXPERIMENTS:")
exp_root = Path("experiments")
results = []
for exp_dir in exp_root.iterdir():
    if not exp_dir.is_dir():
        continue
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir():
            continue
        scores_file = run_dir / "training_scores.json"
        if scores_file.exists():
            with open(scores_file) as f:
                d = json.load(f)
            results.append((exp_dir.name, d))

for exp_name, data in sorted(results, key=lambda x: x[1].get('best_score', -999), reverse=True):
    status = "🟢" if data['status'] == 'complete' else "🟡"
    print(f"  {status} {exp_name:20} Episodes: {data['num_episodes']:4d}  Best: {data['best_score']:7.2f}")

print("\n" + "="*60 + "\n")
EOF
```

---

## 📌 QUICK REFERENCE TABLE

| Log Type | Location | Created By | Update | Size | View |
|----------|----------|------------|--------|------|------|
| Baselines | `baseline_comparison_*.json` | baselines.py | Once | 5KB | `cat baseline_comparison_*.json` |
| Scores | `experiments/*/*/training_scores.json` | Training script | Per 50 eps | 1MB | `tail -1 ...` |
| Checkpoints | `experiments/*/*/best/*.pt` | Training script | Per 50 eps | 45MB | `ls -lh ...` |
| Task Spec | `experiments/*/*/README.md` | main_train.py | Once | 2KB | `cat ...` |
| Training Logs | `experiments/*/*/log/` | ROS 2 | Real-time | 100MB | `tail -50 ...` |
| Analysis | `comparison_results.json` | comparison.py | Once | 10KB | `cat ...` |
| Plots | `comparison_learning_curves.png` | comparison.py | Once | 200KB | `display ...` |

---

Now you know exactly where everything is! 🎯
