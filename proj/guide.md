# Multi-Robot RL Training Guide

This guide will help you run the MADDPG training on Gazebo with 3 robots.

---

## Prerequisites

✅ Verify everything is ready:

```bash
# Check ROS 2 is installed
which ros2
echo $ROS_DISTRO  # Should show: humble

# Check Gazebo is installed
which gzserver

# Check CUDA/GPU
nvidia-smi  # Should show RTX 4050 with 6GB VRAM
```

---

## Step 1: Terminal 1 - Launch Gazebo Simulation

Open **Terminal 1** and run:

```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3
```

**Expected Output:**
```
[INFO] [launch]: All log files can be found below ...
[INFO] [gzserver-1]: process started with pid [...]
[INFO] [gzclient-2]: process started with pid [...]
[INFO] [robot_state_publisher-3]: process started with pid [...]
[INFO] [spawn_entity.py-4]: process started with pid [...]
...
[INFO] [spawn_entity.py-8]: Spawn status: Entity [my_bot2] already exists.
```

**Note:** Gazebo server process may show errors, but robots are spawned successfully. Check with:
```bash
ros2 topic list | grep robot
# Should show: /robot_0/cmd_vel, /robot_1/cmd_vel, etc.
```

✅ **Gazebo is ready** when you see robot topics listed.

---

## Step 2: Terminal 2 - Start MADDPG Training

Open **Terminal 2** and run:

```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source /opt/ros/humble/setup.bash
source install/setup.bash
export CUDA_VISIBLE_DEVICES=0

python3 -m start_reinforcement_learning.maddpg_main
```

**Expected Output (First 30 seconds):**
```
[INFO] [1776467882.736936628] [maddpg_node]: Map nuumber: 1
[INFO] [1776467882.737198257] [maddpg_node]: Robot number: 3
[INFO] [1776467883.831277822] [maddpg_node]: Map nuumber: 1
...
[INFO] [ss.Spawn_Entity]: %%%%%%%%%% Spawning Goal %%%%%%%%%%
debug1
```

⏳ **Wait 60-90 seconds** for initialization. The training is loading neural networks and setting up.

**Expected After ~90 seconds:**
```
[INFO] [maddpg_node]: Episode: 10, Average score: 45.3
[INFO] [maddpg_node]: Episode: 20, Average score: 48.2
```

---

## Step 3: Terminal 3 (Optional) - Monitor GPU Usage

Open **Terminal 3** to watch GPU memory:

```bash
watch -n 1 nvidia-smi
```

**Expected GPU Usage:**
- VRAM: ~5.5-5.9 GB / 6 GB
- GPU Util: 30-60%
- Process: `python3` showing ~142MB+ VRAM

---

## Step 4: Training Progress

The training will run for **5000 episodes**. 

**Typical Timeline:**
- **Episodes 0-100:** Initialization, random exploration
- **Episodes 100-500:** Learning starting, scores improving
- **Episodes 500+:** Convergence, high scores

**Print Every 10 Episodes:**
```
Episode: 10, Average score: 45.1
Episode: 20, Average score: 48.3
Episode: 30, Average score: 52.1
...
Episode: 100, Average score: 75.2
```

---

## Expected Duration

| Phase | Time |
|-------|------|
| Gazebo startup | 10-15 secs |
| MADDPG initialization | 30-60 secs |
| Training (5000 eps) | 2-4 hours |
| **Total** | **~2-4.5 hours** |

---

## Stop Training

### Graceful Shutdown:
- **Terminal 2:** Press `Ctrl+C` to stop MADDPG
- **Terminal 1:** Press `Ctrl+C` to stop Gazebo
- Models are saved to: `install/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/`

---

## Troubleshooting

### ❌ Problem: "No module named 'start_reinforcement_learning'"

**Solution:** Rebuild packages:
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

---

### ❌ Problem: "gzserver process has died"

**Solution:** This is normal. Gazebo client (GUI) may crash, but server is running. Verify with:
```bash
# In Terminal 2, check for robot topics
ros2 topic list | grep -E "robot_[0-9]"
```

Should show:
```
/robot_0/cmd_vel
/robot_0/odom
/robot_1/cmd_vel
/robot_1/odom
/robot_2/cmd_vel
/robot_2/odom
```

---

### ❌ Problem: Training slow / High CPU usage

**Solution:** Check GPU is being used:
```bash
nvidia-smi  # In Terminal 3
```

If GPU usage is 0%, kill and restart with:
```bash
pkill -f maddpg_main
python3 -m start_reinforcement_learning.maddpg_main
```

---

### ❌ Problem: "Waiting for service /spawn_entity"

**Solution:** Gazebo startup issue. Kill all processes and restart:
```bash
pkill -f gzserver
pkill -f ros2
sleep 2

# Then restart Terminal 1 and 2
```

---

## Advanced: Reduce Training Time

### Option 1: Fewer Episodes (30 mins)
Edit and reduce episodes in `maddpg_main.py`:
```python
N_GAMES = 500  # Instead of 5000
```

### Option 2: Smaller Batch Size
```python
batch_size=32  # Instead of 128
replay_buffer=30000  # Instead of 100k
```

### Option 3: Check if SSH has GPU available

If training too slow, try SSH with A6000 GPU (~30 GB VRAM):
```bash
ssh user@remote_server
# Install ROS 2 first if needed
sudo apt install ros-humble-desktop
```

---

## Output Files

After training completes:

| File | Location |
|------|----------|
| **Actor weights** | `install/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/actor_*.pth` |
| **Critic weights** | `install/start_reinforcement_learning/start_reinforcement_learning/deep_learning_weights/maddpg/critic_*.pth` |
| **Training logs** | `~/.ros/log/` |

---

## Next Steps: Create Baselines

After training completes, create comparison baselines:

```bash
# In proj/ folder, create:
# - random_walk.py
# - frontier_based.py
# - lawnmower.py
# - evaluate.py
```

See `baseline_setup.md` for instructions.

---

## Quick Reference

### Terminal 1 (Gazebo)
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && \
source /opt/ros/humble/setup.bash && \
source install/setup.bash && \
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3
```

### Terminal 2 (MADDPG Training)
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && \
source /opt/ros/humble/setup.bash && \
source install/setup.bash && \
export CUDA_VISIBLE_DEVICES=0 && \
python3 -m start_reinforcement_learning.maddpg_main
```

### Terminal 3 (GPU Monitor)
```bash
watch -n 1 nvidia-smi
```

---

## Questions?

- Check Terminal 2 output for error messages
- Monitor GPU with `nvidia-smi`
- Verify Gazebo with `ros2 topic list`

**Good luck! 🚀**
