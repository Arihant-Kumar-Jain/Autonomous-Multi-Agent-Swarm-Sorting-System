"""Configuration for Multi-Agent Warehouse Simulation."""

# ─── Grid ───────────────────────────────────────────────────────────
GRID_ROWS = 15
GRID_COLS = 15

# Cell types
EMPTY = 0
WALL = 1
DROP_ZONE = 2
OBJECT = 3

# ─── Warehouse layout (15×15) ──────────────────────────────────────
# Realistic warehouse with aisles, shelving, open staging area
WAREHOUSE_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# ─── Robots ─────────────────────────────────────────────────────────
NUM_ROBOTS = 3
ROBOT_SPAWN_POSITIONS = [(1, 1), (1, 7), (1, 13)]
ROBOT_COLORS_RGB = [
    (46, 204, 113),   # Green  — Robot 0
    (52, 152, 219),   # Blue   — Robot 1
    (231, 76, 60),    # Red    — Robot 2
]
ROBOT_NAMES = ["Atlas", "Bolt", "Claw"]

# ─── Objects (collectible items) ───────────────────────────────────
OBJECT_POSITIONS = [(2, 4), (5, 10), (8, 4), (4, 7), (10, 1)]
NUM_OBJECTS = len(OBJECT_POSITIONS)

# ─── Drop zone center ──────────────────────────────────────────────
DROP_ZONE_CENTER = (13, 7)

# ─── Actions ────────────────────────────────────────────────────────
ACTIONS = {
    0: (-1, 0),   # UP
    1: (1, 0),    # DOWN
    2: (0, -1),   # LEFT
    3: (0, 1),    # RIGHT
    4: (0, 0),    # WAIT
}
NUM_ACTIONS = len(ACTIONS)
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]

# ─── RL Hyperparameters ────────────────────────────────────────────
STATE_SIZE = 15       # 12 base + 3 robot_id one-hot (17 with congestion)
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 300000
TARGET_UPDATE = 50    # episodes
TRAIN_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 300

# ─── PPO Hyperparameters (A5000 optimized) ─────────────────────────
PPO_LR = 3e-4
PPO_BATCH_SIZE = 256       # larger batches = better GPU utilization
PPO_EPOCHS = 10            # update epochs per rollout
ROLLOUT_LENGTH = 1024      # steps before PPO update
GAE_LAMBDA = 0.95          # GAE lambda
CLIP_EPS = 0.2             # PPO clipping
ENTROPY_COEF = 0.05        # exploration bonus (prevents entropy collapse)
VALUE_COEF = 0.5           # value loss weight
PPO_EPISODES = 10000       # overnight training on A5000

# ─── Rewards ────────────────────────────────────────────────────────
REWARD_GOAL = 20.0
REWARD_PICKUP = 10.0
REWARD_COLLISION = -10.0
REWARD_CLOSER = 0.5
REWARD_FARTHER = -0.3
REWARD_STEP = -0.05
REWARD_WAIT = -0.1
REWARD_WALL = -2.0
REWARD_CONGESTION_PENALTY = -1.0   # extra penalty on collision (improved RL only)
REWARD_PROXIMITY_PENALTY = -0.3    # per-step penalty per nearby robot (improved RL only)

# ─── Congestion ─────────────────────────────────────────────────────
CONGESTION_RADIUS = 3     # cells
CONGESTION_WEIGHT = 0.5   # in task allocation cost

# ─── MAPPO Hyperparameters ──────────────────────────────────────────
MAPPO_LR_ACTOR = 3e-4
MAPPO_LR_CRITIC = 1e-3       # critic can learn faster
MAPPO_BATCH_SIZE = 256
MAPPO_EPOCHS = 10
MAPPO_ROLLOUT_LENGTH = 1024
MAPPO_EPISODES = 10000

# ─── Collision avoidance ────────────────────────────────────────────
COLLISION_THRESHOLD = 1   # grid distance
PRIORITY_ORDER = [0, 1, 2]  # robot 0 has highest priority

# ─── Visualization ──────────────────────────────────────────────────
CELL_SIZE = 50
FPS = 6                  # slow enough to see coordination
WINDOW_PADDING = 20

# Colors
COLOR_BG = (30, 30, 40)
COLOR_WALL = (80, 80, 100)
COLOR_FLOOR = (50, 50, 65)
COLOR_DROP_ZONE = (155, 89, 182)
COLOR_OBJECT = (241, 196, 15)
COLOR_TEXT = (220, 220, 220)
COLOR_PATH = (100, 100, 120)
COLOR_GRID_LINE = (45, 45, 58)
