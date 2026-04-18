# config.py — Complete configuration for the multi-agent RL system

import os

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for saving/loading models
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Directory for logging
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Grid dimensions
GRID_ROWS = 30
GRID_COLS = 30

# Number of agents (2-4 recommended)
NUM_AGENTS = 2

# Maximum number of balls to collect per episode
MAX_BALLS = 10

# Obstacle configuration
NUM_OBSTACLES = 8  # Number of static obstacles

# Observation radius for local observations (used in environment.py)
OBS_RADIUS = 2  # Radius for local observation around agent


# ──────────────────────────────────────────────────────────────────────────────
# REWARD CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Reward values
REWARD_COLLECT_BALL = 10.0      # Reward for collecting a ball
REWARD_COLLISION = -2.0          # Penalty for colliding with another agent
REWARD_STEP = -0.01              # Small penalty per step to encourage efficiency
REWARD_BONUS_ALL_BALLS = 5.0     # Bonus when all balls are collected
REWARD_SHAPING_FACTOR = 0.1      # Factor for distance-based reward shaping


# ──────────────────────────────────────────────────────────────────────────────
# TABULAR Q-LEARNING PARAMETERS (Legacy / Fallback)
# ──────────────────────────────────────────────────────────────────────────────

# Learning parameters
ALPHA = 0.1          # Learning rate
GAMMA = 0.95         # Discount factor

# Exploration parameters
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_MIN = 0.01   # Minimum exploration rate
EPSILON_DECAY = 0.995 # Decay factor per episode


# ──────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK PARAMETERS (PPO with Communication)
# ──────────────────────────────────────────────────────────────────────────────

# Neural network architecture
LEARNING_RATE = 0.003          # Learning rate for policy and value networks
HIDDEN_DIMS = [128, 128]        # Hidden layer dimensions
COMM_DIM = 64                   # Communication message dimension
NUM_ATTENTION_HEADS = 4         # Number of attention heads for communication

# PPO specific parameters
PPO_CLIP_EPSILON = 0.2          # PPO clipping parameter
PPO_EPOCHS = 4                  # Number of PPO update epochs per batch
PPO_BATCH_SIZE = 64             # Batch size for PPO updates
GAE_LAMBDA = 0.95               # GAE lambda parameter for advantage estimation

# Experience replay
BUFFER_CAPACITY = 10000         # Capacity of replay buffer

# Training parameters
TRAIN_ITERATIONS = 100          # Number of iterations per update


# ──────────────────────────────────────────────────────────────────────────────
# RENDERING CONFIGURATION (Pygame)
# ──────────────────────────────────────────────────────────────────────────────

# Display settings
CELL_SIZE = 48                  # Pixels per grid cell
WINDOW_W = GRID_COLS * CELL_SIZE
WINDOW_H = GRID_ROWS * CELL_SIZE + 100  # Extra space for HUD
HUD_HEIGHT = 100
FPS = 30
WINDOW_TITLE = "Multi-Agent RL - Ball Collection"

# Colors (RGB)
COLOR_BG = (20, 20, 35)               # Dark background
COLOR_GRID_LINE = (50, 50, 70)        # Subtle grid lines
COLOR_OBSTACLE = (100, 100, 120)      # Gray obstacles
COLOR_BALL = (255, 200, 50)           # Golden balls
COLOR_HUD_BG = (30, 30, 45)           # HUD background
COLOR_TEXT = (220, 220, 240)          # Light text
COLOR_TEXT_DIM = (150, 150, 180)      # Dim text

# Agent colors (distinct for each agent)
AGENT_COLORS = [
    (100, 200, 255),  # Agent 0: Light blue
    (255, 100, 100),  # Agent 1: Coral red
    (100, 255, 100),  # Agent 2: Light green
    (255, 200, 100),  # Agent 3: Orange
]


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Save/Load settings
SAVE_EVERY = 100                  # Save model every N episodes
PRINT_EVERY = 10                  # Print stats every N episodes

# Training modes
MODE_RL = "rl"                    # Reinforcement learning mode
MODE_RULE = "rule"                # Rule-based baseline mode

# Evaluation settings
EVAL_EPISODES = 100               # Number of episodes for evaluation
EVAL_INTERVAL = 500               # Evaluate every N episodes


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_agent_color(agent_id: int):
    """Get the color for an agent based on its ID."""
    return AGENT_COLORS[agent_id % len(AGENT_COLORS)]


def get_state_dimension():
    """
    Calculate the dimension of the state vector for neural network input.
    
    State features:
    - Position one-hot: GRID_ROWS * GRID_COLS
    - Normalized distance to nearest ball: 1
    - Normalized distance to nearest agent: 1
    - Direction to nearest ball (dx, dy): 2
    - Local obstacle map (5x5): 25
    - Agent ID one-hot: NUM_AGENTS
    - Number of balls remaining: 1
    
    Total: (GRID_ROWS * GRID_COLS) + 1 + 1 + 2 + 25 + NUM_AGENTS + 1
    """
    return (GRID_ROWS * GRID_COLS) + 1 + 1 + 2 + 25 + NUM_AGENTS + 1


# ──────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def verify_config():
    """Verify that the configuration is valid."""
    assert GRID_ROWS > 0 and GRID_COLS > 0, "Grid dimensions must be positive"
    assert NUM_AGENTS >= 1, "Must have at least 1 agent"
    assert MAX_BALLS >= 1, "Must have at least 1 ball"
    assert NUM_OBSTACLES < GRID_ROWS * GRID_COLS, "Too many obstacles"
    assert 0 <= EPSILON_MIN <= 1, "EPSILON_MIN must be between 0 and 1"
    assert 0 <= EPSILON_START <= 1, "EPSILON_START must be between 0 and 1"
    assert 0 < ALPHA <= 1, "ALPHA must be between 0 and 1"
    assert 0 < GAMMA <= 1, "GAMMA must be between 0 and 1"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert PPO_BATCH_SIZE > 0, "PPO_BATCH_SIZE must be positive"
    assert BUFFER_CAPACITY > 0, "BUFFER_CAPACITY must be positive"
    
    print("✓ Configuration verified")
    print(f"  Grid: {GRID_ROWS}x{GRID_COLS}")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  Max balls: {MAX_BALLS}")
    print(f"  Obstacles: {NUM_OBSTACLES}")
    print(f"  State dimension: {get_state_dimension()}")
    print(f"  Model directory: {MODEL_DIR}")
    print(f"  Log directory: {LOG_DIR}")


if __name__ == "__main__":
    # Test the configuration
    verify_config()