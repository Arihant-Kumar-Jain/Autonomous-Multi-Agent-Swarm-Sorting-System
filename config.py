"""Configuration for Multi-Agent Warehouse Simulation."""

# ─── Grid ───────────────────────────────────────────────────────────
GRID_ROWS = 30
GRID_COLS = 30

# Cell types
EMPTY = 0
WALL = 1
DROP_ZONE = 2
OBJECT = 3

# ─── Warehouse layout (30×30) ──────────────────────────────────────
# Large warehouse: 6 shelf zones, narrow aisles, bottleneck corridors
# W=wall, .=empty, D=drop zone
WAREHOUSE_MAP = [
    #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 1  spawn row
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 2
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],  # 3  shelf zone A
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],  # 4
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],  # 5
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 6  aisle
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 7  bottleneck
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],  # 8  shelf zone B
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],  # 9
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 10 aisle
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 11
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],  # 12 shelf zone C
    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],  # 13
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14 aisle
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],  # 15 partial walls
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 16
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # 17 shelf zone D
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # 18
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 19 aisle
    [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],  # 20 shelf zone E
    [1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],  # 21
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 22 aisle
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],  # 23 shelf zone F
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 24
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 25 staging area
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 26
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 27 drop zone
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 28
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 29
]

# ─── Robots ─────────────────────────────────────────────────────────
NUM_ROBOTS = 3
ROBOT_SPAWN_POSITIONS = [(1, 1), (1, 15), (1, 28)]
ROBOT_COLORS_RGB = [
    (46, 204, 113),   # Green  — Robot 0
    (52, 152, 219),   # Blue   — Robot 1
    (231, 76, 60),    # Red    — Robot 2
]
ROBOT_NAMES = ["Atlas", "Bolt", "Claw"]

# ─── Objects (collectible items) ───────────────────────────────────
OBJECT_POSITIONS = [(3, 5), (5, 14), (8, 4), (9, 25), (12, 11), (14, 1), (17, 9), (20, 5), (23, 14), (23, 27)]
NUM_OBJECTS = len(OBJECT_POSITIONS)
RANDOMIZE_OBJECTS = True    # spawn objects at random walkable cells each episode

# ─── Drop zone center ──────────────────────────────────────────────
DROP_ZONE_CENTER = (28, 15)

# ─── Partial Observability ─────────────────────────────────────────
SENSOR_RANGE = 3            # obstacle sensor range (cells). 1 = adjacent only
SENSOR_NOISE = 0.1          # probability of false sensor reading
VISIBILITY_RADIUS = 999     # full visibility — robots share positions via ROS2 comms

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
STATE_SIZE = 16       # 13 base + 3 robot_id one-hot (18 with congestion)
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 300000
TARGET_UPDATE = 50    # episodes
TRAIN_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 800   # more steps for 30x30 map

# ─── PPO Hyperparameters (A5000 optimized) ─────────────────────────
PPO_LR = 3e-4
PPO_BATCH_SIZE = 256
PPO_EPOCHS = 10
ROLLOUT_LENGTH = 1024
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.05        # exploration bonus (prevents entropy collapse)
VALUE_COEF = 0.5
PPO_EPISODES = 10000

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
REWARD_DISCOVERY = 5.0             # bonus for discovering an object (exploration)

# ─── Congestion ─────────────────────────────────────────────────────
CONGESTION_RADIUS = 3     # cells
CONGESTION_WEIGHT = 0.5   # in task allocation cost

# ─── MAPPO Hyperparameters ──────────────────────────────────────────
MAPPO_LR_ACTOR = 3e-4
MAPPO_LR_CRITIC = 1e-3
MAPPO_BATCH_SIZE = 256
MAPPO_EPOCHS = 10
MAPPO_ROLLOUT_LENGTH = 1024
MAPPO_EPISODES = 10000

# ─── Collision avoidance ────────────────────────────────────────────
COLLISION_THRESHOLD = 1
PRIORITY_ORDER = [0, 1, 2]

# ─── Visualization ──────────────────────────────────────────────────
CELL_SIZE = 26            # small cells for 30x30 map
FPS = 10
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
