# Quick Start: SSH A6000 GPU Training

## Step 1: Access SSH Server

```bash
# From your local machine
ssh username@your_a6000_server_ip

# Example:
ssh aman@192.168.1.100
```

## Step 2: Check GPU

```bash
nvidia-smi

# Should show:
# A6000 with 24GB or 48GB VRAM
```

## Step 3: Setup (One time only)

```bash
# Navigate to experiments
cd ~/cs671_7/experiments

# Install minimal dependencies (ROS not needed for training)
pip install torch numpy matplotlib tensorboard

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

## Step 4: Run Variant Training

### Variant 2: Modified Reward (Fastest)
```bash
cd exp2_modified_reward
python3 train_modified_reward.py --gpu 0 --episodes 1000

# Expected: ~2-3 hours for 1000 episodes
```

### Variant 3: Coverage Task
```bash
cd exp3_coverage_task
python3 train_coverage.py --gpu 0 --episodes 1000

# Expected: ~2-3 hours
```

### Variant 4: Formation Control
```bash
cd exp4_formation_control
python3 train_formation.py --gpu 0 --episodes 1000

# Expected: ~2-3 hours
```

## Step 5: Run All Sequentially

```bash
# Run all variants back-to-back
bash run_all_variants.sh

# This will:
# - Run exp2 (2-3 hrs)
# - Then exp3 (2-3 hrs)
# - Then exp4 (2-3 hrs)
# Total: ~8 hours

# You can disconnect and check later!
```

## Step 6: Monitor Training

```bash
# In another SSH terminal
cd ~/cs671_7/experiments

# Watch TensorBoard logs (if saving)
tensorboard --logdir=logs/

# Or check specific experiment
tail -f exp2_modified_reward/training.log

# Or quick status check
ls -lah exp2_modified_reward/results.json
```

## Step 7: Copy Results Back

```bash
# After training completes, copy to local machine
# From local machine:

scp -r username@a6000_server:~/cs671_7/experiments/exp2_modified_reward/results/ ./results_exp2/

# Or all at once:
rsync -avz username@a6000_server:~/cs671_7/experiments/ ./experiments_results/
```

---

## ✨ Advantages of SSH A6000

| Metric | RTX 4050 | A6000 |
|--------|----------|-------|
| VRAM | 6 GB | 24-48 GB |
| Speed | 1x | 5-10x faster |
| Training time | 2-4 hrs | 20-40 mins |
| Cost | Yours | FREE (school) |
| Multi-GPU | No | Possible |

---

## 🔄 Disconnect & Reconnect

```bash
# Start training, then disconnect
nohup python3 train_modified_reward.py > training.log &

# Close SSH connection (Ctrl+D)

# Next day, reconnect and check
ssh username@a6000_server
tail -f ~/cs671_7/experiments/exp2_modified_reward/training.log

# Results already saved!
```

---

## 💾 Estimated Storage

```
Per experiment:
- Training logs: ~50 MB
- Model weights: ~100-500 MB
- Results JSON: ~1 MB

6 experiments total:
- ~1-3 GB total

This fits easily on any server!
```

---

## 🆘 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size in config
export BATCH_SIZE=64  # Instead of 128

python3 train_modified_reward.py
```

### Connection Timeout
```bash
# SSH with keep-alive
ssh -o ServerAliveInterval=60 username@server
```

### Already Running?
```bash
# Check if training already running
ps aux | grep python3 | grep train

# Kill if stuck
pkill -f train_modified_reward.py
```

---

## 📊 Next: Compare Results

After all training complete:
```bash
python3 comparison.py  # generates plots
```

---

## 🚀 You're Ready!

Your A6000 can run 5x faster than local RTX 4050.
This means all 6 variants done in ~20-30 hours total instead of ~100+ hours! ⚡
