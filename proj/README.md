# Multi-Robot RL Training Project

Complete guide for training and evaluating multi-robot exploration with MADDPG.

---

## 📁 Project Structure

```
~/cs671_7/proj/
├── README.md                 ← You are here
├── guide.md                  ← START HERE! Full training guide
├── commands.md               ← Quick copy-paste commands
├── baseline_setup.md         ← Create comparison baselines
├── setup.sh                  ← Automated setup script
└── baselines/                ← (Create these later)
    ├── random_walk.py
    ├── frontier_based.py
    ├── lawnmower.py
    └── evaluate.py
```

---

## 🚀 Quick Start

### 1️⃣ Run Setup (One-time)
```bash
bash ~/cs671_7/proj/setup.sh
```

### 2️⃣ Read Training Guide
```bash
cat ~/cs671_7/proj/guide.md
```

### 3️⃣ Launch Training (3 terminals)

**Terminal 1 - Gazebo:**
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && \
source /opt/ros/humble/setup.bash && \
source install/setup.bash && \
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3
```

**Terminal 2 - Training:**
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && \
source /opt/ros/humble/setup.bash && \
source install/setup.bash && \
export CUDA_VISIBLE_DEVICES=0 && \
python3 -m start_reinforcement_learning.maddpg_main
```

**Terminal 3 - Monitor (Optional):**
```bash
watch -n 1 nvidia-smi
```

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **guide.md** | Complete step-by-step training instructions |
| **commands.md** | Quick reference for all commands |
| **baseline_setup.md** | Create 3 baseline algorithms for comparison |
| **setup.sh** | Automated setup verification |

---

## ⏱️ Timeline

| Phase | Duration |
|-------|----------|
| Setup & verification | 5 mins |
| Gazebo launch | 15 secs |
| MADDPG initialization | 60 secs |
| **Training** | **2-4 hours** |
| Baseline testing | 30 mins |
| Evaluation | 20 mins |
| **Total** | **~3-4.5 hours** |

---

## 📊 What You'll Get

After training completes:

1. **Trained Models** (MADDPG agents)
   - Location: `install/start_reinforcement_learning/.../deep_learning_weights/maddpg/`
   - Can be loaded and reused

2. **Training Logs**
   - Episodes, scores, rewards
   - Location: `~/.ros/log/`

3. **Comparison Results** (after running baselines)
   - Coverage comparison plots
   - Performance metrics
   - RL vs Non-RL analysis

---

## ✨ Features

- ✅ **3 Robots** exploring a Gazebo environment
- ✅ **MADDPG Algorithm** for multi-agent learning
- ✅ **GPU Acceleration** (RTX 4050 or better)
- ✅ **Easy Setup** with automated scripts
- ✅ **Baseline Comparisons** (Random Walk, Frontier-Based, Lawnmower)
- ✅ **Performance Metrics** (coverage, efficiency, time)

---

## 🎯 Success Indicators

✅ **Training is working if:**
- Terminal 1: Gazebo shows robot topics in `ros2 topic list`
- Terminal 2: Shows "Episode 10, Average score: XX.X"
- Terminal 3: GPU usage ~5.5-5.9 GB on RTX 4050

---

## ❌ Troubleshooting

See **guide.md** for detailed troubleshooting guide.

**Quick fixes:**
```bash
# Rebuild packages
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && \
rm -rf build install log && \
colcon build --symlink-install

# Kill stuck processes
pkill -f maddpg_main
pkill -f gzserver
killall -9 ros2
```

---

## 📝 Next Steps After Training

### Step 1: Create Baselines
```bash
mkdir -p ~/cs671_7/proj/baselines
# Copy code from baseline_setup.md into:
# - baselines/random_walk.py
# - baselines/frontier_based.py
# - baselines/lawnmower.py
# - baselines/evaluate.py
```

### Step 2: Run Evaluation
```bash
cd ~/cs671_7/proj/baselines
python3 evaluate.py
```

### Step 3: Generate Report
Compare results and create visualization showing RL > Baselines

---

## 📚 Reference Files

### Original Repository
- Location: `~/cs671_7/github_repos/multi-robot-exploration-rl/`
- Contains: MADDPG implementation, environment, launch files

### MADDPG Implementation
- File: `src/start_reinforcement_learning/start_reinforcement_learning/maddpg_main.py`
- Algorithm: Multi-Agent Deep Deterministic Policy Gradient

### Environment Interface
- File: `src/start_reinforcement_learning/start_reinforcement_learning/env_logic/logic.py`
- Handles: Robot control, reward calculation, collision detection

---

## 🔧 Configuration

### Change Number of Robots
```bash
ros2 launch start_rl_environment main.launch.py \
  map_number:=1 robot_number:=5  # 5 robots instead of 3
```

### Change Maps
```bash
ros2 launch start_rl_environment main.launch.py \
  map_number:=2 robot_number:=3  # Map 2 instead of Map 1
```

### Reduce Training Time
Edit `maddpg_main.py`:
```python
N_GAMES = 500  # Instead of 5000 (30 mins instead of 2-4 hours)
```

---

## 🐛 Known Issues

| Issue | Solution |
|-------|----------|
| Gazebo client crashes | Normal, server still runs. Check with `ros2 topic list` |
| GPU out of memory | Reduce batch size or number of robots |
| Training hangs | Gazebo may have stalled, restart both terminals |
| `ros2: command not found` | Source setup: `source /opt/ros/humble/setup.bash` |

---

## 💡 Tips

- ✨ Use **Terminal 1 for Gazebo, Terminal 2 for Training, Terminal 3 for monitoring**
- ✨ **Don't close** Terminal 1 while Terminal 2 is training
- ✨ **Watch GPU** in Terminal 3 to ensure CUDA is being used
- ✨ **Save models** appear after first 100 episodes

---

## 📞 Help

### Reading Output

**Good:**
```
Episode: 10, Average score: 45.1
Episode: 20, Average score: 51.2
Episode: 30, Average score: 58.3
```
→ Training is learning!

**Bad:**
```
Episode: 10, Average score: 10.0
Episode: 20, Average score: 9.8
Episode: 30, Average score: 8.2
```
→ Training crashed, restart.

### Checking GPU
```bash
nvidia-smi
# RTX 4050 should show 5.5+ GB memory used
```

### Checking ROS
```bash
ros2 topic list | grep robot
# Should show: /robot_0/cmd_vel, /robot_1/cmd_vel, /robot_2/cmd_vel
```

---

## 🎓 Learning Resources

- **MADDPG Paper:** https://arxiv.org/abs/1706.02275
- **ROS 2 Humble:** https://docs.ros.org/en/humble/
- **Gazebo Simulation:** https://gazebosim.org/

---

## 📄 License

This project uses ROS 2 Humble and Gazebo under their respective open-source licenses.

---

## 🚀 Ready to Start?

1. ✅ Run setup: `bash ~/cs671_7/proj/setup.sh`
2. ✅ Read guide: `cat ~/cs671_7/proj/guide.md`
3. ✅ Launch training using 3 terminals
4. ✅ Wait for training to complete
5. ✅ Create baselines and compare

**Good luck! Let me know how it goes!** 🤖

---

**Last Updated:** April 17, 2026  
**Project Location:** `~/cs671_7/proj/`  
**Original Repo:** `~/cs671_7/github_repos/multi-robot-exploration-rl/`
