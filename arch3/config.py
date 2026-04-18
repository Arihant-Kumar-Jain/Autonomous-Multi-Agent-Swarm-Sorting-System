"""
config.py - Central configuration for MAPPO Multi-Agent System
All hyperparameters and environment settings in one place.
"""
import torch

class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    GRID_SIZE       = 30
    NUM_AGENTS      = 3
    NUM_BALLS       = 10
    MAX_STEPS       = 500          # steps per episode
    OBS_RADIUS      = 5            # local observation window half-size

    # ── Obstacle curriculum phases ────────────────────────────────────────
    # Phase 0: no obstacles, Phase 1: sparse, Phase 2: full
    CURRICULUM_PHASES = {
        0: {"num_obstacles": 0,   "episodes": 300},
        1: {"num_obstacles": 15,  "episodes": 300},
        2: {"num_obstacles": 30,  "episodes": 9999},
    }

    # ── Action Space ─────────────────────────────────────────────────────────
    # 0:UP  1:DOWN  2:LEFT  3:RIGHT  4:STAY  5:PICK  6:DROP
    NUM_ACTIONS = 7

    # ── Observation sizes (computed below) ───────────────────────────────────
    # agent pos(2) + carrying(1) + nearest_ball_rel(2) + box_rel(2)
    # + local_obstacle_window((2*OBS_RADIUS+1)^2) + nearby_agents_rel(2*(N-1))
    LOCAL_WINDOW   = (2 * OBS_RADIUS + 1) ** 2          # 121
    OBS_SIZE       = 2 + 1 + 2 + 2 + LOCAL_WINDOW + 2 * (NUM_AGENTS - 1)  # 132

    # Global state: agent_pos(2*N) + ball_pos(2*10) + box_pos(2) + obstacles(GRID^2)
    GLOBAL_STATE_SIZE = 2 * NUM_AGENTS + 2 * NUM_BALLS + 2 + GRID_SIZE ** 2  # 914

    # ── Network architecture ──────────────────────────────────────────────────
    HIDDEN_DIM      = 256
    COMM_DIM        = 64           # communication embedding dimension
    NUM_HEADS       = 4            # attention heads
    ACTOR_LR        = 3e-4
    CRITIC_LR       = 1e-3

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    CLIP_EPS        = 0.2
    GAMMA           = 0.99
    GAE_LAMBDA      = 0.95
    ENTROPY_COEF    = 0.01
    VALUE_COEF      = 0.5
    MAX_GRAD_NORM   = 0.5
    PPO_EPOCHS      = 4            # epochs per update
    MINI_BATCH_SIZE = 256
    UPDATE_INTERVAL = 2048         # steps collected before each PPO update

    # ── Reward shaping ────────────────────────────────────────────────────────
    R_DELIVERY      = 25.0
    R_PICKUP        = 5.0
    R_ALL_DONE      = 50.0
    R_STEP          = -0.2
    R_COLLISION     = -5.0
    R_INVALID_ACT   = -2.0
    R_OSCILLATION   = -1.0
    R_DIST_SCALE    = 1.0          # distance-reduction reward scale

    # ── Training ─────────────────────────────────────────────────────────────
    TOTAL_EPISODES      = 5000
    SAVE_INTERVAL       = 100
    LOG_INTERVAL        = 10
    MODEL_DIR           = "models"
    LOG_DIR             = "logs"
    USE_COMMUNICATION   = True     # toggle communication module

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Rendering ────────────────────────────────────────────────────────────
    CELL_SIZE       = 24
    FPS             = 30


cfg = Config()