# Quick Commands Reference

## One-Line Commands (Copy & Paste)

### Terminal 1: Launch Gazebo
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && source /opt/ros/humble/setup.bash && source install/setup.bash && ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3
```

### Terminal 2: Start MADDPG Training
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl && source /opt/ros/humble/setup.bash && source install/setup.bash && export CUDA_VISIBLE_DEVICES=0 && python3 -m start_reinforcement_learning.maddpg_main
```

### Terminal 3: Monitor GPU (Optional)
```bash
watch -n 1 nvidia-smi
```

---

## Multi-Command Scripts

### Gazebo Launch (save as `start_gazebo.sh`)
```bash
#!/bin/bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3
```

Run with:
```bash
bash start_gazebo.sh
```

### Training Launch (save as `start_training.sh`)
```bash
#!/bin/bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
source /opt/ros/humble/setup.bash
source install/setup.bash
export CUDA_VISIBLE_DEVICES=0
python3 -m start_reinforcement_learning.maddpg_main
```

Run with:
```bash
bash start_training.sh
```

---

## Alternative Map/Robot Configs

### Different Maps
```bash
# Map 2 instead of Map 1
ros2 launch start_rl_environment main.launch.py map_number:=2 robot_number:=3

# 2 robots instead of 3
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=2

# 5 robots (max supported: 7)
ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=5
```

---

## Cleanup Commands

### Kill Training
```bash
pkill -f maddpg_main
```

### Kill Gazebo
```bash
pkill -f gzserver
```

### Kill All ROS
```bash
pkill -f ros2
killall -9 gzserver gzclient
```

### Clean Logs
```bash
rm -rf ~/.ros/log/*
```

---

## Monitoring Commands

### Check Robot Topics
```bash
ros2 topic list | grep robot
```

### Monitor Robot Odometry
```bash
ros2 topic echo /robot_0/odom
```

### View GPU Real-Time
```bash
watch -n 0.5 nvidia-smi
```

### Check Training Logs
```bash
tail -f ~/.ros/log/latest/maddpg_node*
```

---

## Rebuild Commands

### Full Rebuild
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
rm -rf build install log
colcon build --symlink-install
```

### Quick Rebuild (no clean)
```bash
cd ~/cs671_7/github_repos/multi-robot-exploration-rl
colcon build --symlink-install
```

---

## Setup Commands

### Install Missing Dependencies
```bash
sudo apt-get update
sudo apt-get install -y ros-humble-simple-launch
```

### Verify Installations
```bash
which ros2
which gzserver
which colcon
nvidia-smi
```
