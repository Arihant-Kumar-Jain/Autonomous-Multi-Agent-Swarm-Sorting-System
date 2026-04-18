import sys
sys.path.insert(0, '.')
from config import cfg
from env.environment import GridWorld
import numpy as np

# Verify obs size
env = GridWorld(num_obstacles=0)
obs = env.reset()
gs  = env.get_global_state()

print(f'OBS_SIZE in config     : {cfg.OBS_SIZE}')
print(f'Actual obs[0] shape    : {obs[0].shape[0]}')
print(f'GLOBAL_STATE_SIZE cfg  : {cfg.GLOBAL_STATE_SIZE}')
print(f'Actual global_state    : {gs.shape[0]}')

assert obs[0].shape[0] == cfg.OBS_SIZE, f'OBS MISMATCH: {obs[0].shape[0]} vs {cfg.OBS_SIZE}'
assert gs.shape[0] == cfg.GLOBAL_STATE_SIZE, f'GS MISMATCH: {gs.shape[0]} vs {cfg.GLOBAL_STATE_SIZE}'

# Run a few steps
env2 = GridWorld(num_obstacles=5)
obs2 = env2.reset()
actions = [0, 1, 5]
next_obs, rewards, done, info = env2.step(actions)
print(f'Step OK - rewards: {[f"{r:.2f}" for r in rewards]}  info: {info}')
print('ALL ASSERTIONS PASSED')
