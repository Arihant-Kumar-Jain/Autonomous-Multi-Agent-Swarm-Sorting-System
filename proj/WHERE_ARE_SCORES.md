# 📊 Where Your Training Scores Are Saved

## Real-Time Location (While Training)

```
CONSOLE OUTPUT (happening right now):
Episode 50, Average score: -30.1
Episode 60, Average score: -31.8

Location: Terminal output only
Access: Watch the terminal running `python3 -m start_reinforcement_learning.maddpg_main`
```

## Permanent Location (After Training Completes)

```
📁 PRIMARY SCORES FILE:
/home/aman/cs671_7/github_repos/multi-robot-exploration-rl/src/start_reinforcement_learning/deep_learning_weights/maddpg/training_scores.json

This contains:
├─ episode_scores: [List of all scores, one per episode]
├─ num_episodes: Total episodes trained
├─ final_avg_score: Average of last 100 episodes
├─ best_score: Best score achieved
├─ timestamp: When training finished
├─ map_number: Which map (1 or 2)
└─ robot_number: Number of robots (3)

Example file after training:
{
  "episode_scores": [-37.6, -35.2, -34.1, ..., 72.3, 72.1],
  "num_episodes": 5000,
  "final_avg_score": 72.15,
  "best_score": 73.4,
  "timestamp": "2026-04-17T14:30:45.123456",
  "map_number": 1,
  "robot_number": 3
}
```

## Quick Access (While Training)

```bash
# Check progress in real-time
tail -f /tmp/maddpg_training.log

# Or view console from the running terminal
# (Just watch the output)
```

## After Training - Load & Analyze

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load scores
with open('~/cs671_7/github_repos/multi-robot-exploration-rl/src/start_reinforcement_learning/deep_learning_weights/maddpg/training_scores.json') as f:
    data = json.load(f)

scores = data['episode_scores']
print(f"Final score: {data['final_avg_score']:.2f}")
print(f"Best score: {data['best_score']:.2f}")

# Plot
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Training Progress')
plt.savefig('training_curve.png')
```

## Your Current Progress

```bash
# See what's been generated so far:
ls -la ~/cs671_7/github_repos/multi-robot-exploration-rl/src/start_reinforcement_learning/deep_learning_weights/maddpg/

# Expected to see (after enough episodes):
# - actor_1_checkpoint.pt (robot 1 weights)
# - critic_checkpoint.pt (critic network)
# - training_scores.json ← YOUR SCORES!
```

## Timeline

| Episode | Status | Score File |
|---------|--------|-----------|
| 10-60 (NOW) | Training | Not saved yet (runs in memory) |
| 100 | Ongoing | Still training |
| 1000 | Ongoing | Still training |
| 5000 | COMPLETE ✅ | `training_scores.json` created! |

## How to Check Right Now

### Option 1: Watch Real-Time Console
```bash
# In the terminal where training is running, you see:
Episode 50, Average score: -30.1
Episode 60, Average score: -31.8
Episode 70, Average score: -29.5
...

This IS the scores! Printing every 10 episodes.
```

### Option 2: Check Saved Models (Future)
```bash
# After training, check if directory exists:
ls -la ~/cs671_7/github_repos/multi-robot-exploration-rl/src/start_reinforcement_learning/deep_learning_weights/maddpg/

# Should show:
# training_scores.json ← Contains all scores!
```

### Option 3: Python Script (Future)
```python
# After training, run this to see scores:
python3 << 'EOF'
import json
with open('src/start_reinforcement_learning/deep_learning_weights/maddpg/training_scores.json') as f:
    data = json.load(f)
    print(f"Episodes trained: {data['num_episodes']}")
    print(f"Final average score: {data['final_avg_score']:.2f}")
    print(f"Best score: {data['best_score']:.2f}")
    print(f"\nAll scores: {data['episode_scores'][:10]} ... [plus {len(data['episode_scores'])-10} more]")
EOF
```

## Summary

🎯 **Your scores are EVERYWHERE:**

1. ✅ **Live**: Console output (right now, every 10 episodes)
2. ✅ **Saved**: `training_scores.json` (after training finishes)
3. ✅ **Models**: Actor/Critic weights (save best performance)

## Let Training Continue! 🚀

Your current training:
- Episode 60: -31.8
- Expected Episode 5000: ~72 ✓

Keep it running! In ~6-8 hours you'll have the full `training_scores.json` file with all the data!
