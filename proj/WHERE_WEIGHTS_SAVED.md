# 📁 Where Are Weights Saving? Complete Guide

## 🎯 Quick Answer

```
WEIGHTS (Neural Networks):
📁 /home/aman/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/

Files saved:
├─ agent_0_actor        (Robot 0 policy network)
├─ agent_0_critic       (Robot 0 value network)
├─ agent_0_target_actor
├─ agent_0_target_critic
├─ agent_1_actor        (Robot 1)
├─ agent_1_critic
├─ agent_1_target_actor
├─ agent_1_target_critic
├─ agent_2_actor        (Robot 2)
├─ agent_2_critic
├─ agent_2_target_actor
├─ agent_2_target_critic
└─ training_scores.json ← ALL YOUR SCORES!

SCORES (JSON):
📄 Same location: training_scores.json
```

---

## 📊 What Each File Contains

### Weights Files (PyTorch .pt format)

```python
# Each file is a PyTorch state_dict
# Can be loaded like:

import torch
weights = torch.load('agent_0_actor')
# weights contains all neural network parameters
```

| File | Purpose | Size |
|------|---------|------|
| `agent_0_actor` | Robot 0 decision maker | ~5-20 MB |
| `agent_0_critic` | Robot 0 value evaluator | ~5-20 MB |
| `agent_0_target_actor` | Target network (copy) | ~5-20 MB |
| `agent_0_target_critic` | Target network (copy) | ~5-20 MB |
| Same for robots 1 & 2 | 3 robots × 4 networks | ~60-240 MB total |

### Scores File (JSON)

```json
{
  "episode_scores": [-37.6, -35.2, -34.1, ..., 72.3, 72.1],
  "num_episodes": 5000,
  "final_avg_score": 72.15,
  "best_score": 73.4,
  "timestamp": "2026-04-17T...",
  "map_number": 1,
  "robot_number": 3
}
```

---

## 🔍 Check Where Files Actually Are

### Method 1: Find all saved weights

```bash
# Find all saved weight files
find ~/cs671_7/github_repos/multi-robot-exploration-rl -name "agent_*" -type f 2>/dev/null

# Or look in the directory
ls -lah ~/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/ 2>/dev/null || echo "Directory not created yet"
```

### Method 2: While training is running

```bash
# Keep watching for new files
watch 'find ~/cs671_7/github_repos/multi-robot-exploration-rl -name "agent_*" -type f 2>/dev/null | wc -l'

# This shows how many weight files exist (will go from 0 → 12 when training saves)
```

### Method 3: Check scores specifically

```bash
# After training finishes, view scores
cat ~/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/training_scores.json | python3 -m json.tool

# Or just the important metrics
python3 << 'EOF'
import json
path = 'install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/training_scores.json'
with open(path) as f:
    data = json.load(f)
print(f"Final Score: {data['final_avg_score']:.2f}")
print(f"Best Score: {data['best_score']:.2f}")
print(f"Episodes: {data['num_episodes']}")
EOF
```

---

## 📍 Step-by-Step Path Breakdown

### From the Code

[maddpg_main.py line 39](maddpg_main.py#L39):
```python
chkpt_dir_var = os.path.join(
    get_package_share_directory('start_reinforcement_learning'),
    'start_reinforcement_learning',
    'deep_learning_weights',
    'maddpg'
)
```

This translates to:
```
get_package_share_directory('start_reinforcement_learning')
↓
/home/aman/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning

+ 'start_reinforcement_learning'
↓
/home/aman/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning

+ 'deep_learning_weights'
↓
/home/aman/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights

+ 'maddpg'
↓
/home/aman/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/
```

### Then Each Network Saves

[networks.py line 10](networks.py#L10):
```python
self.chkpt_file = os.path.join(chkpt_dir, name)
```

Where `name` is like `"agent_0_actor"`, so:
```
/maddpg/ + agent_0_actor
→ /maddpg/agent_0_actor
```

---

## 🔄 When Do Weights Save?

### From maddpg_main.py (lines 113-123):

```python
for i in range(N_GAMES):  # 5000 episodes
    # ... training ...
    avg_score = np.mean(score_history[-100:])
    if not evaluate:
        if avg_score > best_score:
            maddpg_agents.save_checkpoint()  # ← SAVES HERE!
            best_score = avg_score
```

**Weights save ONLY when average score improves!**

```
Episode 10: avg_score = -37.6 → Save (first time, best so far)
Episode 20: avg_score = -35.2 → Save (improved!)
Episode 30: avg_score = -36.1 → NO save (got worse)
Episode 100: avg_score = -15.3 → Save (improvement!)
...
Episode 5000: avg_score = 72.3 → FINAL weights saved!
```

### When Scores Save?

From modified `maddpg_main.py` (I added this):

```python
# After training loop completes (line ~130)
with open(scores_file, 'w') as f:
    json.dump(scores_data, f, indent=2)
```

**Scores save ONCE at the very end!**

---

## 📥 How to Use Saved Weights

### Load for Testing/Evaluation

```python
from start_reinforcement_learning.maddpg_algorithm.maddpg import MADDPG

# Create agents
maddpg_agents = MADDPG(...)

# Load best weights
maddpg_agents.load_checkpoint()

# Now use for testing
actions = maddpg_agents.choose_action(observation)
```

### Copy Weights to Another Machine

```bash
# On local machine, copy weights from remote
scp -r username@remote:/path/to/maddpg/ ./my_weights/

# Or if on remote SSH
rsync -avz username@remote:/path/to/maddpg/ ./my_weights/
```

---

## ✅ What You Should See

### Before Training
```
/home/aman/cs671_7/github_repos/multi-robot-exploration-rl/
├── install/
│   └── share/
│       └── start_reinforcement_learning/
│           └── start_reinforcement_learning/
│               └── deep_learning_weights/
│                   └── maddpg/  ← EMPTY DIR
```

### After Episode 1-10
```
/maddpg/
├── agent_0_actor      ← Created!
├── agent_0_critic
├── agent_0_target_actor
├── agent_0_target_critic
├── agent_1_actor
├── ... (12 files total)
└── training_scores.json ← Created at the end
```

### After Training Complete
```
/maddpg/
├── agent_0_actor      ← Best weights
├── agent_0_critic
├── ...
└── training_scores.json  ← All scores
```

---

## 🎯 Commands to Monitor

```bash
# Watch weights being created
watch -n 5 'ls -lah ~/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/ 2>/dev/null | tail -5'

# Count how many weight files exist
watch 'find ~/cs671_7 -path "*deep_learning_weights/maddpg*" -name "agent_*" 2>/dev/null | wc -l'

# Check final scores
cat ~/cs671_7/github_repos/multi-robot-exploration-rl/install/share/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/training_scores.json | python3 -m json.tool | head -20
```

---

## 🚀 Quick Summary

| Item | Location | Format | When |
|------|----------|--------|------|
| **Weights** | `install/share/.../deep_learning_weights/maddpg/agent_*` | PyTorch `.pt` | When score improves |
| **Scores** | `same/path/training_scores.json` | JSON | When training ends |
| **Logs** | Console output | Text | Every 10 episodes |

**Your baseline training will save files to:**
```
~/cs671_7/github_repos/multi-robot-exploration-rl/
   install/share/start_reinforcement_learning/
      start_reinforcement_learning/
         deep_learning_weights/
            maddpg/
               ├─ agent_0_actor
               ├─ agent_0_critic
               ├─ ... (12 files total)
               └─ training_scores.json
```

Total size: ~100-200 MB (manageable!)
