"""Quick smoke-test: run 30 episodes and print results."""
import sys
sys.path.insert(0, '.')
from config import cfg
cfg.LOG_INTERVAL = 5
cfg.SAVE_INTERVAL = 9999
cfg.UPDATE_INTERVAL = 512   # faster updates for test

from main import train
import argparse

args = argparse.Namespace(episodes=30, render=False, demo=False, eval=False, eval_episodes=10)
train(args)
