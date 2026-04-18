# Training Output Guide

## What You Should See Now

### Good Output (Training Working)
```
[INFO] [1776468512.848043184] [maddpg_node]: Map nuumber: 1
[INFO] [1776468512.848043184] [maddpg_node]: Robot number: 3
...
[INFO] [logger]: A robot has found the goal
...
[INFO] [maddpg_node]: Episode: 10, Average score: -34.6
[INFO] [maddpg_node]: Episode: 20, Average score: -20.3
[INFO] [maddpg_node]: Episode: 30, Average score: +5.2
[INFO] [maddpg_node]: Episode: 40, Average score: +15.1
```

### What Each Line Means

| Output | Meaning |
|--------|---------|
| `Map number: 1` | Using Map 1 environment |
| `Robot number: 3` | Training with 3 robots |
| `A robot has found the goal` | ✅ Success! Robot reached goal (+20 reward) |
| `Episode: 10, Average score: -34.6` | Average reward this episode (last 100 eps avg) |

---

## 📈 Score Progression (What to Expect)

### Poor Training (Gets Stuck)
```
Episode: 10:   -40
Episode: 100:  -40  ← NOT improving, stuck
Episode: 500:  -40  ← Still stuck = PROBLEM
```

### Good Training (Converging)
```
Episode: 10:   -40
Episode: 50:   -20  ← Getting better!
Episode: 100:  +10  ← Reached goals!
Episode: 500:  +70  ← Converged
```

**Your current score (-34.6 at episode 10) is NORMAL for beginning. Watch if it improves.**

---

## 🐛 Debug Info Removed

- ❌ Removed: `debug1` print statements (no more spam)
- ✅ Output cleaner now
- ✅ Can focus on actual metrics

---

## 💡 How to Monitor Training

### Real-time Checks

**1. Watch scores improving:**
```bash
# Get last 10 scores (copy this to terminal)
tail -20 ~/.ros/log/maddpg_node.log | grep "Episode"
```

**2. Check if robots are moving:**
```bash
# Show active robot topics
ros2 topic list | grep -E "cmd_vel|odom"
```

**3. Monitor GPU usage:**
```bash
# Watch GPU memory
watch -n 1 nvidia-smi
```

---

## ✨ Success Indicators (All Should Be True)

| Check | Expected | Your Status |
|-------|----------|-------------|
| Training starts | Episode 1, 2, 3... | ✅ Yes |
| Robots navigate | "robot has found goal" appears | ✅ Yes |
| Score improves | -40 → -20 → +20 → +70 | Need to wait |
| GPU used | 5+ GB on RTX 4050 | Need to verify |
| No crashes | Consistent output | Likely ✅ |

---

## ⏱️ Timeline Estimate

| Phase | Time | Episode Range |
|-------|------|---------------|
| **Initialization** | 1-2 mins | Episodes 0-1 |
| **Early Learning** | 30-45 mins | Episodes 1-50 |
| **Learning** | 1-2 hours | Episodes 50-500 |
| **Convergence** | 1-2 hours | Episodes 500-5000 |

---

## 🎯 Next Action

1. ✅ Let training run for **30 more episodes** (5-10 mins)
2. 📊 Check if average score is improving
3. ✨ If improving: Let it run to 500 episodes
4. ❌ If stuck (no improvement): Check troubleshooting below

---

## 🔧 Troubleshooting

### If Score Stuck at Negative After 100 Episodes
```bash
# Check robot status
ros2 topic echo /robot_0/odom | head -20  # Should show position changing

# Check if goal is reachable
ros2 topic echo /goal_box/pose  # Should see goal position
```

### If GPU Not Being Used
```bash
# Verify CUDA in use
nvidia-smi  # Should show python3 using memory

# Restart with explicit GPU
export CUDA_VISIBLE_DEVICES=0
python3 -m start_reinforcement_learning.maddpg_main
```

### If Training Crashes
```bash
# Check last error
tail -50 ~/.ros/log/latest/maddpg_node.log

# Restart from scratch
pkill -f maddpg_main
source install/setup.bash
python3 -m start_reinforcement_learning.maddpg_main
```

---

## 📌 Key Takeaway

**Your output shows:**
- ✅ Environment working
- ✅ Robots navigating
- ✅ Goals reachable
- ✅ Training initialized

**Keep running, watch score improve!** 🚀
