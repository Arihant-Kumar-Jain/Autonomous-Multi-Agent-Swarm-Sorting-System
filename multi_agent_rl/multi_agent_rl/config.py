# config.py — All tunable parameters for the Multi-Agent RL simulation
# Centralizing config here makes it easy to run experiments by changing one file.

# ─────────────────────────────────────────────
# GRID / WORLD
# ─────────────────────────────────────────────
GRID_ROWS        = 20          # Number of rows in the grid
GRID_COLS        = 20          # Number of columns in the grid
NUM_BALLS        = 10          # Total collectible balls per episode
NUM_AGENTS       = 3           # Fixed at 3 for this simulation
NUM_OBSTACLES    = 10          # Static obstacle count (set 0 to disable)
MAX_STEPS        = 500         # Max steps per episode before forced reset

# ─────────────────────────────────────────────
# PYGAME / RENDERING
# ─────────────────────────────────────────────
CELL_SIZE        = 36          # Pixels per grid cell
HUD_HEIGHT       = 90          # Pixel height of the HUD panel at bottom
FPS              = 15          # Frames per second (lower = slower demo)
WINDOW_TITLE     = "Multi-Agent Collaborative RL"

# Derived window dimensions (do not edit these directly)
WINDOW_W         = GRID_COLS * CELL_SIZE
WINDOW_H         = GRID_ROWS * CELL_SIZE + HUD_HEIGHT

# ─────────────────────────────────────────────
# COLORS (R, G, B)
# ─────────────────────────────────────────────
COLOR_BG         = (18,  18,  30)   # Dark background
COLOR_GRID_LINE  = (40,  40,  60)   # Subtle grid lines
COLOR_BALL       = (255, 215,   0)  # Gold balls
COLOR_OBSTACLE   = (80,  80,  100)  # Dark slate obstacles
COLOR_HUD_BG     = (10,  10,  20)   # HUD background
COLOR_TEXT       = (220, 220, 255)  # HUD text
COLOR_TEXT_DIM   = (120, 120, 160)  # Secondary HUD text

# Agent colors — visually distinct
AGENT_COLORS = [
    (0,   200, 255),   # Cyan   — Agent 0
    (255,  80, 120),   # Red    — Agent 1
    (80,  255, 160),   # Green  — Agent 2
]

# ─────────────────────────────────────────────
# REINFORCEMENT LEARNING — Q-LEARNING
# ─────────────────────────────────────────────
ALPHA            = 0.15        # Learning rate (how fast Q-values update)
GAMMA            = 0.95        # Discount factor (future reward importance)
EPSILON_START    = 1.0         # Initial exploration rate
EPSILON_MIN      = 0.05        # Minimum exploration rate (never fully greedy)
EPSILON_DECAY    = 0.997       # Multiplicative decay per episode

# ─────────────────────────────────────────────
# REWARD SHAPING
# ─────────────────────────────────────────────
REWARD_BALL_COLLECTED    =  20.0   # Agent touches a new ball
REWARD_ALL_COLLECTED     =  50.0   # Bonus when ALL balls are cleared
REWARD_STEP_PENALTY      =  -0.5   # Time penalty — encourages efficiency
REWARD_COLLISION         =  -5.0   # Agent tries to enter an occupied cell
REWARD_ANTI_CLUSTER      =  -3.0   # Two agents within 2 cells of same ball
REWARD_DISTANCE_SCALE    =   2.0   # Multiplier for distance-shaping term
REWARD_REVISIT_PENALTY   =  -1.0   # Agent re-visits a recently visited cell

# ─────────────────────────────────────────────
# STUCK DETECTION
# ─────────────────────────────────────────────
STUCK_WINDOW     = 15          # Steps to look back for "no progress"
REVISIT_WINDOW   = 8           # Steps to check for position revisits

# ─────────────────────────────────────────────
# OBSERVATION / STATE
# ─────────────────────────────────────────────
OBS_RADIUS       = 4           # Local observation window radius (cells)
# State encodes: coarse pos + nearest ball dirs + nearest agent dirs
COARSE_BINS      = 5           # Grid divided into N×N coarse bins

# ─────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────
MODEL_DIR        = "models"
SAVE_EVERY       = 50          # Save Q-tables every N episodes
LOG_PATH         = "logs/training_log.csv"
