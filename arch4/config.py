"""
config.py - Central configuration for arch4 MAPPO

Key differences from arch3:
  • Compact global state (42 dims vs 914) - drops raw 900-cell obstacle map
  • Richer local obs (87 dims) - ALL balls visible, enables implicit coordination
  • Rebalanced rewards - softer step penalty, stronger delivery, coordination bonus
  • Performance-based curriculum - advances only when agents show actual progress
  • LR scheduling + entropy annealing
"""

import torch


class Config:
    # ── Environment ──────────────────────────────────────────────────────────
    GRID_SIZE   = 30
    NUM_AGENTS  = 3
    NUM_BALLS   = 10
    MAX_STEPS   = 500        # steps per episode
    OBS_RADIUS  = 5          # local obstacle window half-size (5×5 -> 11×11 window? No - we use 5x5 centered)

    # ── Curriculum (PERFORMANCE-BASED, not episode-count-based) ──────────────
    # Phase advances when avg balls_delivered >= threshold over last window
    CURRICULUM_PHASES = {
        0: {"num_obstacles": 0,  "advance_threshold": 3.0, "window": 50},
        1: {"num_obstacles": 15, "advance_threshold": 5.0, "window": 50},
        2: {"num_obstacles": 30, "advance_threshold": 9999},  # final phase
    }

    # ── Action Space ─────────────────────────────────────────────────────────
    # 0:UP  1:DOWN  2:LEFT  3:RIGHT  4:STAY  5:PICK  6:DROP
    NUM_ACTIONS = 7

    # ── Observation sizes ────────────────────────────────────────────────────
    # Agent-local obs:
    #   pos(2) + carry(1) + box_rel(2)                      =  5
    #   all ball rel positions (10×2)                        = 20
    #   all ball status: available/transit/delivered (10×1)  = 10
    #   other agents rel pos+carry (2 agents × 3)           =  6
    #   local 5×5 obstacle window                            = 25
    #   step fraction (1)                                    =  1
    #   TOTAL                                                = 67
    OBS_SIZE = 67

    # Compact global state (for centralized critic):
    #   agent pos (3×2=6) + agent carry (3) + all ball pos (10×2=20)
    #   + ball status (10) + box pos (2) + step fraction (1)
    #   TOTAL = 42
    GLOBAL_STATE_SIZE = 42

    # ── Network architecture ──────────────────────────────────────────────────
    HIDDEN_DIM  = 256
    ACTOR_LR    = 3e-4
    CRITIC_LR   = 1e-3

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    CLIP_EPS        = 0.2
    GAMMA           = 0.99
    GAE_LAMBDA      = 0.95
    ENTROPY_COEF    = 0.05    # start higher (was 0.01), annealed down
    ENTROPY_COEF_MIN = 0.005
    VALUE_COEF      = 0.5
    MAX_GRAD_NORM   = 0.5
    PPO_EPOCHS      = 8       # more epochs per update (was 4)
    MINI_BATCH_SIZE = 128
    UPDATE_INTERVAL = 1024    # steps between updates (was 2048; shorter = more responsive)

    # ── Reward shaping ────────────────────────────────────────────────────────
    # Key philosophy: delivery is primary, step penalty is light, distance shaping
    # is the main dense signal. Collision penalties are mild vs delivery rewards.
    R_DELIVERY      = 5.0     # was 50.0 - strong primary signal
    R_PICKUP        = 1.0     # was 10.0
    R_ALL_DONE      = 10.0    # was 100.0
    R_STEP          = -0.005  # was -0.05 - very soft; don't punish for trying
    R_WALL_HIT      = -0.05   # was -0.5 - mild; don't terrify agent from exploring
    R_AGENT_COLL    = -0.1    # agent-agent collision (separate from wall)
    R_INVALID_ACT   = -0.02   # was -0.2 - very mild
    R_DIST_SCALE    = 0.3     # was 3.0 - dominant dense signal
    R_CLAIM_BONUS   = 0.2     # was 2.0 - NEW: reward for being the closest agent to an unclaimed ball
                               #       encourages division of labor

    # ── Training ─────────────────────────────────────────────────────────────
    TOTAL_EPISODES  = 5000
    SAVE_INTERVAL   = 100
    LOG_INTERVAL    = 10
    MODEL_DIR       = "models"
    LOG_DIR         = "logs"

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Rendering ────────────────────────────────────────────────────────────
    CELL_SIZE = 24
    FPS       = 30


cfg = Config()
