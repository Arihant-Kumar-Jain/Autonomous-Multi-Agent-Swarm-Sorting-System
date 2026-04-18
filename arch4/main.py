"""
main.py - Entry point for arch4 MAPPO Multi-Agent Ball Delivery

Usage
-----
  python main.py                  # train (no render, fast)
  python main.py --demo           # demo with pygame (loads saved model)
  python main.py --eval           # evaluation mode (greedy, no exploration)
  python main.py --render         # train with live pygame render
  python main.py --episodes 2000  # custom episode count

Architecture summary (arch4)
-----------------------------
  SharedActor        - 3-layer MLP + residual + Tanh, shared across agents
  CentralizedCritic  - compact 42-dim global state, used only at training
  MAPPO              - PPO + GAE + cosine LR decay + entropy annealing
  PerformanceCurriculum - advances phases on actual delivery progress
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from config import cfg
from env.environment import GridWorld
from agents.agent    import Agent
from marl.mappo      import MAPPO
from utils.helpers   import PerformanceCurriculum, EpisodeLogger, pretty_metrics

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

# Optional Pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ── Pygame renderer ───────────────────────────────────────────────────────────

COLORS = {
    "bg":       (15,  15,  25),
    "grid":     (30,  30,  45),
    "obstacle": (90,  90, 110),
    "ball":     (255, 210,  50),
    "box":      (60,  210,  90),
    "agent":    [(80, 140, 255), (255, 100, 110), (100, 230, 200)],
    "carry":    (255, 255, 120),
    "text":     (220, 225, 235),
    "hud_bg":   (8,    8,  18),
    "phase_0":  (50,  80, 140),
    "phase_1":  (130, 80,  50),
    "phase_2":  (130, 50,  80),
}

CELL  = cfg.CELL_SIZE
HUD_H = 90


def init_pygame():
    pygame.init()
    w = cfg.GRID_SIZE * CELL
    h = cfg.GRID_SIZE * CELL + HUD_H
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("arch4 - MAPPO Multi-Agent Ball Delivery")
    font_big   = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)
    clock      = pygame.time.Clock()
    return screen, font_big, font_small, clock


def render_frame(screen, font_big, font_small, env: GridWorld,
                 episode: int, step: int, total_reward: float, phase: int,
                 delivered: int):
    screen.fill(COLORS["bg"])
    G = cfg.GRID_SIZE

    # Grid lines
    for r in range(G + 1):
        pygame.draw.line(screen, COLORS["grid"], (0, r * CELL), (G * CELL, r * CELL))
    for c in range(G + 1):
        pygame.draw.line(screen, COLORS["grid"], (c * CELL, 0), (c * CELL, G * CELL))

    # Obstacles
    for r in range(G):
        for c in range(G):
            if env.obstacle_map[r, c]:
                pygame.draw.rect(screen, COLORS["obstacle"],
                                 (c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2))

    # Drop-off box
    br, bc = env.box_pos
    pygame.draw.rect(screen, COLORS["box"],
                     (bc * CELL + 2, br * CELL + 2, CELL - 4, CELL - 4),
                     border_radius=5)
    lbl = font_small.render("BOX", True, (10, 20, 10))
    screen.blit(lbl, (bc * CELL + 4, br * CELL + 6))

    # Balls (only available ones)
    from env.environment import BALL_AVAILABLE
    for b in range(env.n_balls):
        if env.ball_status[b] == BALL_AVAILABLE:
            br2, bc2 = env.ball_pos[b]
            cx = bc2 * CELL + CELL // 2
            cy = br2 * CELL + CELL // 2
            pygame.draw.circle(screen, COLORS["ball"], (cx, cy), CELL // 3)

    # Agents
    for i in range(env.n_agents):
        ar, ac   = env.agent_pos[i]
        color    = COLORS["agent"][i % len(COLORS["agent"])]
        cx = ac * CELL + CELL // 2
        cy = ar * CELL + CELL // 2
        pygame.draw.circle(screen, color, (cx, cy), CELL // 2 - 2)
        if env.agent_carry[i]:
            # Yellow dot on top -> carrying a ball
            pygame.draw.circle(screen, COLORS["carry"],
                               (cx, cy - CELL // 3), 5)
        id_lbl = font_small.render(str(i), True, (255, 255, 255))
        screen.blit(id_lbl, (cx - 4, cy - 6))

    # HUD
    hud_y    = G * CELL
    phase_bg = COLORS.get(f"phase_{phase}", COLORS["hud_bg"])
    pygame.draw.rect(screen, COLORS["hud_bg"], (0, hud_y, G * CELL, HUD_H))
    pygame.draw.rect(screen, phase_bg, (0, hud_y, 8, HUD_H))  # phase color strip

    balls_left = int(np.sum(env.ball_status != 2))   # not delivered
    carried    = int(np.sum(env.agent_carry))

    hud1 = font_big.render(
        f"Ep:{episode:5d}  Step:{step:4d}  Phase:{phase}  "
        f"Delivered:{delivered:2d}/10  Carrying:{carried}  R:{total_reward:8.2f}",
        True, COLORS["text"]
    )
    screen.blit(hud1, (15, hud_y + 8))
    hud2 = font_small.render(
        "arch4 | MAPPO | Shared Actor + Centralized Critic | Performance Curriculum",
        True, (100, 110, 130)
    )
    screen.blit(hud2, (15, hud_y + 38))
    hud3 = font_small.render(
        f"Balls on floor: {balls_left - carried}  |  Device: {cfg.DEVICE}",
        True, (80, 90, 110)
    )
    screen.blit(hud3, (15, hud_y + 58))
    pygame.display.flip()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    print("=" * 70)
    print("  arch4 - MAPPO Multi-Agent Ball Delivery  |  Performance Curriculum")
    print("=" * 70)
    print(f"  Device      : {cfg.DEVICE}")
    print(f"  Grid        : {cfg.GRID_SIZE}×{cfg.GRID_SIZE}")
    print(f"  Agents      : {cfg.NUM_AGENTS}")
    print(f"  Balls       : {cfg.NUM_BALLS}")
    print(f"  Obs size    : {cfg.OBS_SIZE}")
    print(f"  Global state: {cfg.GLOBAL_STATE_SIZE}")
    print("=" * 70)

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,   exist_ok=True)

    writer     = SummaryWriter(cfg.LOG_DIR) if TB_AVAILABLE else None
    curriculum = PerformanceCurriculum()
    env        = GridWorld(num_obstacles=curriculum.num_obstacles)
    mappo      = MAPPO(writer=writer)
    logger     = EpisodeLogger(cfg.LOG_DIR, writer)
    agents     = [Agent(i) for i in range(cfg.NUM_AGENTS)]

    # Load existing checkpoint if available
    model_prefix = os.path.join(cfg.MODEL_DIR, "mappo")
    mappo.load(model_prefix)

    screen = font_b = font_s = clock = None
    if args.render and PYGAME_AVAILABLE:
        screen, font_b, font_s, clock = init_pygame()

    last_losses  = {}
    step_counter = 0
    t_start      = time.time()

    for episode in range(1, args.episodes + 1):
        env.set_num_obstacles(curriculum.num_obstacles)
        obs = env.reset()

        for a in agents:
            a.reset_stats()

        ep_reward    = 0.0
        ep_steps     = 0
        delivered    = 0
        done         = False

        while not done:
            # Pygame quit
            if screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        mappo.save(model_prefix)
                        pygame.quit()
                        sys.exit()

            global_state = env.get_global_state()
            value        = mappo.get_value(global_state)
            actions, lp  = mappo.select_actions(obs)

            next_obs, rewards, done, info = env.step(actions)

            mappo.store(obs, global_state, actions, lp, rewards, value, done)

            for i in range(cfg.NUM_AGENTS):
                agents[i].update_from_env(env.agent_pos[i], env.agent_carry[i])

            ep_reward  += sum(rewards)
            ep_steps   += 1
            delivered   = info["balls_delivered"]
            obs         = next_obs
            step_counter += 1

            # PPO update when buffer is full
            if mappo.buffer_ready:
                last_losses = mappo.update(obs, env.get_global_state())

            if screen is not None:
                render_frame(screen, font_b, font_s, env,
                             episode, ep_steps, ep_reward,
                             curriculum.phase, delivered)
                clock.tick(cfg.FPS)

        # Episode end
        success = delivered == cfg.NUM_BALLS
        phase   = curriculum.record(delivered)       # performance-based advance

        # Check if curriculum just advanced
        announcement = curriculum.pop_announcement()
        if announcement:
            print(announcement)

        if episode % cfg.LOG_INTERVAL == 0:
            fps = step_counter / max(time.time() - t_start, 1e-6)
            print(pretty_metrics(episode, phase, ep_steps, ep_reward,
                                  delivered, last_losses, fps))

        logger.log(episode, phase, ep_steps, delivered, ep_reward, success)

        if episode % cfg.SAVE_INTERVAL == 0:
            mappo.save(model_prefix)

    # Training complete
    mappo.save(model_prefix)
    logger.close()
    if writer:
        writer.close()
    if screen is not None:
        pygame.quit()

    print("\nTraining complete!")
    _plot_rewards(logger, cfg.LOG_DIR)


# ── Demo mode ─────────────────────────────────────────────────────────────────

def demo(args):
    if not PYGAME_AVAILABLE:
        print("ERROR: pygame not installed.  pip install pygame")
        return

    env   = GridWorld(num_obstacles=cfg.CURRICULUM_PHASES[2]["num_obstacles"])
    mappo = MAPPO()
    ok    = mappo.load(os.path.join(cfg.MODEL_DIR, "mappo"))
    if not ok:
        print("No saved model found. Train first with: python main.py")
        return

    screen, font_b, font_s, clock = init_pygame()
    episode = 0

    while True:
        episode += 1
        obs  = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0
        delivered = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    done = True   # manual reset with R key

            actions, _ = mappo.select_actions(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions)
            ep_reward += sum(rewards)
            ep_steps  += 1
            delivered  = info["balls_delivered"]

            render_frame(screen, font_b, font_s, env,
                         episode, ep_steps, ep_reward, 2, delivered)
            clock.tick(cfg.FPS)

        time.sleep(0.5)


# ── Evaluation mode ───────────────────────────────────────────────────────────

def evaluate(args):
    n_eval = args.eval_episodes
    env    = GridWorld(num_obstacles=cfg.CURRICULUM_PHASES[2]["num_obstacles"])
    mappo  = MAPPO()
    ok     = mappo.load(os.path.join(cfg.MODEL_DIR, "mappo"))
    if not ok:
        print("No saved model. Train first.")
        return

    successes, steps_list, rewards_list, deliveries_list = [], [], [], []

    for ep in range(n_eval):
        obs  = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0
        delivered = 0

        while not done:
            actions, _ = mappo.select_actions(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions)
            ep_reward += sum(rewards)
            ep_steps  += 1
            delivered  = info["balls_delivered"]

        success = delivered == cfg.NUM_BALLS
        successes.append(int(success))
        steps_list.append(ep_steps)
        rewards_list.append(ep_reward)
        deliveries_list.append(delivered)

        if (ep + 1) % 10 == 0:
            print(f"  Eval ep {ep+1:4d}/{n_eval}  "
                  f"success={np.mean(successes[-10:]):.2f}  "
                  f"avg_delivered={np.mean(deliveries_list[-10:]):.1f}  "
                  f"avg_steps={np.mean(steps_list[-10:]):.1f}")

    print("\n── Evaluation Summary ────────────────────────────────")
    print(f"  Episodes        : {n_eval}")
    print(f"  Success rate    : {np.mean(successes)*100:.1f}%")
    print(f"  Avg delivered   : {np.mean(deliveries_list):.2f} / {cfg.NUM_BALLS}")
    print(f"  Avg steps       : {np.mean(steps_list):.1f}")
    print(f"  Avg reward      : {np.mean(rewards_list):.2f}")
    print("──────────────────────────────────────────────────────")


# ── Reward plot ───────────────────────────────────────────────────────────────

def _plot_rewards(logger, log_dir: str):
    """Plot training curves from CSV log."""
    try:
        import matplotlib.pyplot as plt
        import csv as _csv

        csv_path = os.path.join(log_dir, "training_log.csv")
        if not os.path.exists(csv_path):
            return

        episodes, rewards, delivered = [], [], []
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                episodes.append(int(row["episode"]))
                rewards.append(float(row["reward"]))
                delivered.append(float(row["balls_delivered"]))

        window = 50
        def smooth(arr):
            return np.convolve(arr, np.ones(window)/window, mode='valid')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Reward subplot
        ax1.plot(episodes, rewards, alpha=0.25, color="#5588ff", label="Raw reward")
        if len(rewards) >= window:
            ax1.plot(episodes[window-1:], smooth(rewards),
                     color="#5588ff", linewidth=2, label=f"Smoothed ({window}ep)")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("arch4 - MAPPO Training Curve")
        ax1.legend()
        ax1.grid(alpha=0.2)

        # Deliveries subplot
        ax2.plot(episodes, delivered, alpha=0.25, color="#55cc88", label="Balls delivered")
        if len(delivered) >= window:
            ax2.plot(episodes[window-1:], smooth(delivered),
                     color="#55cc88", linewidth=2, label=f"Smoothed ({window}ep)")
        ax2.axhline(y=cfg.NUM_BALLS, color="orange", linestyle="--",
                    linewidth=1.5, label=f"Target ({cfg.NUM_BALLS} balls)")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Balls Delivered")
        ax2.legend()
        ax2.grid(alpha=0.2)

        plt.tight_layout()
        path = os.path.join(log_dir, "training_curve.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Training curve saved -> {path}")
    except ImportError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="arch4 - MAPPO Multi-Agent Ball Delivery")
    p.add_argument("--demo",          action="store_true",
                   help="Demo mode (pygame, loads saved model)")
    p.add_argument("--eval",          action="store_true",
                   help="Evaluation mode (greedy, no exploration)")
    p.add_argument("--render",        action="store_true",
                   help="Enable pygame rendering during training")
    p.add_argument("--episodes",      type=int, default=cfg.TOTAL_EPISODES,
                   help="Number of training episodes")
    p.add_argument("--eval-episodes", type=int, default=100,
                   help="Number of evaluation episodes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        demo(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)
