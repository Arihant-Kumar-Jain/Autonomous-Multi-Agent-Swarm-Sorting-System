#!/bin/bash

# Quick Setup Script for MADDPG Training
# Run this ONCE before starting training

set -e  # Exit on error

echo "🚀 Multi-Robot RL Training Setup"
echo "=================================="

# Step 1: Check prerequisites
echo ""
echo "📋 Checking prerequisites..."

if ! command -v ros2 &> /dev/null; then
    echo "❌ ROS 2 not found. Install it first."
    exit 1
fi
echo "✅ ROS 2 found"

if ! command -v gzserver &> /dev/null; then
    echo "❌ Gazebo not found. Install it first."
    exit 1
fi
echo "✅ Gazebo found"

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU tools not found."
    exit 1
fi
echo "✅ NVIDIA GPU found"

# Step 2: Check packages
echo ""
echo "📦 Checking ROS packages..."

REPO_PATH=~/cs671_7/github_repos/multi-robot-exploration-rl

if [ ! -d "$REPO_PATH" ]; then
    echo "❌ Repo not found at $REPO_PATH"
    exit 1
fi
echo "✅ Repo found"

# Step 3: Rebuild if needed
echo ""
echo "🔨 Rebuilding packages (this may take 1 min)..."

cd "$REPO_PATH"
if [ ! -d "install" ]; then
    echo "   No install directory found. Building..."
    colcon build --symlink-install > /dev/null 2>&1
fi

if [ ! -f "install/setup.bash" ]; then
    echo "   Rebuilding..."
    rm -rf build install log
    colcon build --symlink-install > /dev/null 2>&1
fi

echo "✅ Packages ready"

# Step 4: Summary
echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo ""
echo "Next, open 2-3 terminals and run:"
echo ""
echo "📌 Terminal 1 (Gazebo):"
echo "   cd ~/cs671_7/github_repos/multi-robot-exploration-rl"
echo "   source /opt/ros/humble/setup.bash"
echo "   source install/setup.bash"
echo "   ros2 launch start_rl_environment main.launch.py map_number:=1 robot_number:=3"
echo ""
echo "📌 Terminal 2 (Training):"
echo "   cd ~/cs671_7/github_repos/multi-robot-exploration-rl"
echo "   source /opt/ros/humble/setup.bash"
echo "   source install/setup.bash"
echo "   export CUDA_VISIBLE_DEVICES=0"
echo "   python3 -m start_reinforcement_learning.maddpg_main"
echo ""
echo "📌 Terminal 3 (Optional - GPU Monitor):"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "See ~/cs671_7/proj/guide.md for full instructions"
echo "=================================="
