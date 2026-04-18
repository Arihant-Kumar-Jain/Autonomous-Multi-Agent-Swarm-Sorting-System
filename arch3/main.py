"""
main.py – Entry point for MAPPO Multi-Agent Ball Delivery

Usage
-----
  python main.py                  # train (no render)
  python main.py --demo           # demo mode (pygame render, loads saved model)
  python main.py --eval           # evaluation mode (no exploration)
  python main.py --no-comm        # disable communication module (ablation)
  python main.py --episodes 2000  # custom episode count

Architecture summary
--------------------
  SharedActor      – one policy for all agents, uses multi-head attention comm
  CentralizedCritic – V(global_state), used only during training (CTDE)
  MAPPO            – PPO with GAE, entropy bonus, gradient clipping
  CurriculumScheduler – 3-phase obstacle curriculum
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

from config import cfg
from env.environment   import GridWorld
from agents.agent      import Agent
from marl.mappo        import MAPPO
from utils.helpers     import (EpisodeLogger, CurriculumScheduler,
                                pretty_metrics)

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
    "bg":       (20,  20,  30),
    "grid":     (35,  35,  50),
    "obstacle": (100, 100, 115),
    "ball":     (255, 200,  50),
    "box":      (80,  200,  80),
    "agent":    [(70, 130, 255), (255, 100, 100), (100, 220, 200)],
    "carry":    (255, 255, 100),
    "text":     (220, 220, 230),
    "hud_bg":   (10,  10,  20),
}

CELL = cfg.CELL_SIZE
HUD_H = 80


def init_pygame():
    pygame.init()
    w = cfg.GRID_SIZE * CELL
    h = cfg.GRID_SIZE * CELL + HUD_H
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("MAPPO – Multi-Agent Ball Delivery")
    font_big   = pygame.font.SysFont("monospace", 16, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)
    clock = pygame.time.Clock()
    return screen, font_big, font_small, clock


def render_frame(screen, font_big, font_small, env: GridWorld,
                 episode: int, step: int, total_reward: float, phase: int):
    screen.fill(COLORS["bg"])
    G = cfg.GRID_SIZE

    # Grid lines
    for r in range(G + 1):
        pygame.draw.line(screen, COLORS["grid"], (0, r*CELL), (G*CELL, r*CELL))
    for c in range(G + 1):
        pygame.draw.line(screen, COLORS["grid"], (c*CELL, 0), (c*CELL, G*CELL))

    # Obstacles
    for r in range(G):
        for c in range(G):
            if env.obstacle_map[r, c]:
                pygame.draw.rect(screen, COLORS["obstacle"],
                                 (c*CELL+1, r*CELL+1, CELL-2, CELL-2))

    # Box
    br, bc = env.box_pos
    pygame.draw.rect(screen, COLORS["box"],
                     (bc*CELL+2, br*CELL+2, CELL-4, CELL-4), border_radius=4)
    lbl = font_small.render("BOX", True, (20, 20, 20))
    screen.blit(lbl, (bc*CELL+4, br*CELL+6))

    # Balls
    for b in range(env.n_balls):
        if env.ball_pos[b, 0] >= 0:
            br2, bc2 = env.ball_pos[b]
            cx = bc2*CELL + CELL//2
            cy = br2*CELL + CELL//2
            pygame.draw.circle(screen, COLORS["ball"], (cx, cy), CELL//3)

    # Agents
    for i in range(env.n_agents):
        ar, ac = env.agent_pos[i]
        color  = COLORS["agent"][i % len(COLORS["agent"])]
        cx = ac*CELL + CELL//2
        cy = ar*CELL + CELL//2
        pygame.draw.circle(screen, color, (cx, cy), CELL//2 - 2)
        # Carrying indicator
        if env.agent_carry[i]:
            pygame.draw.circle(screen, COLORS["carry"], (cx, cy - CELL//3), 4)
        # Agent ID
        id_lbl = font_small.render(str(i), True, (255, 255, 255))
        screen.blit(id_lbl, (cx - 4, cy - 6))

    # HUD
    hud_y = G * CELL
    pygame.draw.rect(screen, COLORS["hud_bg"],
                     (0, hud_y, G*CELL, HUD_H))
    balls_left = int(np.sum(env.ball_pos[:, 0] >= 0))
    carried    = int(np.sum(env.agent_carry))
    hud1 = font_big.render(
        f"Ep:{episode:5d}  Step:{step:4d}  Phase:{phase}  "
        f"Balls:{balls_left}  Carried:{carried}  Reward:{total_reward:8.2f}",
        True, COLORS["text"]
    )
    screen.blit(hud1, (10, hud_y + 10))
    hud2 = font_small.render(
        "MAPPO | Shared Actor + Centralized Critic + Attention Comm | CTDE",
        True, (120, 120, 140)
    )
    screen.blit(hud2, (10, hud_y + 45))
    pygame.display.flip()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    print("=" * 70)
    print("  MAPPO – Multi-Agent Ball Delivery  |  CTDE  |  Attention Comm")
    print("=" * 70)
    print(f"  Device      : {cfg.DEVICE}")
    print(f"  Grid        : {cfg.GRID_SIZE}×{cfg.GRID_SIZE}")
    print(f"  Agents      : {cfg.NUM_AGENTS}")
    print(f"  Communication: {'ON' if args.use_comm else 'OFF'}")
    print("=" * 70)

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,   exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(cfg.LOG_DIR) if TB_AVAILABLE else None

    # Core components
    curriculum = CurriculumScheduler()
    env        = GridWorld(num_obstacles=0)
    mappo      = MAPPO(use_comm=args.use_comm, writer=writer)
    logger     = EpisodeLogger(cfg.LOG_DIR, writer)
    agents     = [Agent(i) for i in range(cfg.NUM_AGENTS)]

    # Load saved model if available
    model_prefix = os.path.join(cfg.MODEL_DIR, "mappo")
    mappo.load(model_prefix)

    # Pygame setup (demo during training if requested)
    screen = font_b = font_s = clock = None
    if args.render and PYGAME_AVAILABLE:
        screen, font_b, font_s, clock = init_pygame()

    # ── Episode loop ──────────────────────────────────────────────────────────
    last_losses  = {}
    step_counter = 0
    t_start      = time.time()
    ep_rewards   = []

    for episode in range(1, args.episodes + 1):
        phase, n_obs = curriculum.step()
        env.set_num_obstacles(n_obs)
        obs = env.reset()

        for a in agents:
            a.reset_stats()

        ep_reward    = 0.0
        ep_steps     = 0
        balls_left   = cfg.NUM_BALLS
        done         = False

        while not done:
            # ── Pygame events ─────────────────────────────────────────────
            if screen is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        mappo.save(model_prefix)
                        pygame.quit()
                        sys.exit()

            # ── Policy step ───────────────────────────────────────────────
            global_state = env.get_global_state()
            value        = mappo.get_value(global_state)
            actions, lp  = mappo.select_actions(obs)

            next_obs, rewards, done, info = env.step(actions)

            mappo.store(obs, global_state, actions, lp, rewards, value, done)

            # Update agent wrappers (for stats)
            for i in range(cfg.NUM_AGENTS):
                agents[i].update_from_env(env.agent_pos[i], env.agent_carry[i])

            ep_reward  += sum(rewards)
            ep_steps   += 1
            balls_left  = info["balls_remaining"]
            obs         = next_obs
            step_counter += 1

            # ── PPO update ────────────────────────────────────────────────
            if mappo.buffer_ready:
                last_losses = mappo.update(obs, env.get_global_state())

            # ── Render ────────────────────────────────────────────────────
            if screen is not None:
                render_frame(screen, font_b, font_s, env,
                             episode, ep_steps, ep_reward, phase)
                clock.tick(cfg.FPS)

        # ── Episode end ───────────────────────────────────────────────────
        success = balls_left == 0
        ep_rewards.append(ep_reward)

        if episode % cfg.LOG_INTERVAL == 0:
            fps = step_counter / max(time.time() - t_start, 1e-6)
            print(pretty_metrics(episode, phase, ep_steps, ep_reward,
                                 balls_left, last_losses, fps))

        logger.log(episode, phase, ep_steps, balls_left, ep_reward, success)

        if episode % cfg.SAVE_INTERVAL == 0:
            mappo.save(model_prefix)

    # ── End of training ───────────────────────────────────────────────────────
    mappo.save(model_prefix)
    logger.close()
    if writer:
        writer.close()
    if screen is not None:
        pygame.quit()

    print("\nTraining complete!")
    _plot_rewards(ep_rewards, cfg.LOG_DIR)


# ── Demo / Evaluation modes ───────────────────────────────────────────────────

def demo(args):
    """Visualize trained agents with Pygame."""
    if not PYGAME_AVAILABLE:
        print("ERROR: pygame not installed.  pip install pygame")
        return

    phase   = 2   # full env for demo
    n_obs   = cfg.CURRICULUM_PHASES[phase]["num_obstacles"]
    env     = GridWorld(num_obstacles=n_obs)
    mappo   = MAPPO(use_comm=args.use_comm)
    ok      = mappo.load(os.path.join(cfg.MODEL_DIR, "mappo"))
    if not ok:
        print("No saved model found.  Train first with: python main.py")
        return

    screen, font_b, font_s, clock = init_pygame()
    episode = 0

    while True:
        episode += 1
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    done = True  # manual reset

            actions, _ = mappo.select_actions(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions)
            ep_reward += sum(rewards)
            ep_steps  += 1

            render_frame(screen, font_b, font_s, env,
                         episode, ep_steps, ep_reward, phase)
            clock.tick(cfg.FPS)

        time.sleep(0.5)  # brief pause between episodes


def evaluate(args):
    """Run N evaluation episodes, report success rate and avg steps."""
    n_eval  = args.eval_episodes
    phase   = 2
    n_obs   = cfg.CURRICULUM_PHASES[phase]["num_obstacles"]
    env     = GridWorld(num_obstacles=n_obs)
    mappo   = MAPPO(use_comm=args.use_comm)
    ok      = mappo.load(os.path.join(cfg.MODEL_DIR, "mappo"))
    if not ok:
        print("No saved model.  Train first.")
        return

    successes, steps_list, rewards_list = [], [], []

    for ep in range(n_eval):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            actions, _ = mappo.select_actions(obs, deterministic=True)
            obs, rewards, done, info = env.step(actions)
            ep_reward += sum(rewards)
            ep_steps  += 1

        success = info["balls_remaining"] == 0
        successes.append(int(success))
        steps_list.append(ep_steps)
        rewards_list.append(ep_reward)

        if (ep + 1) % 10 == 0:
            print(f"  Eval ep {ep+1:4d}/{n_eval}  "
                  f"success={np.mean(successes[-10:]):.2f}  "
                  f"avg_steps={np.mean(steps_list[-10:]):.1f}")

    print("\n── Evaluation Summary ──────────────────────────────")
    print(f"  Episodes     : {n_eval}")
    print(f"  Success rate : {np.mean(successes)*100:.1f}%")
    print(f"  Avg steps    : {np.mean(steps_list):.1f}")
    print(f"  Avg reward   : {np.mean(rewards_list):.2f}")
    print("────────────────────────────────────────────────────")


# ── Reward plot ───────────────────────────────────────────────────────────────

def _plot_rewards(ep_rewards: list[float], log_dir: str):
    try:
        import matplotlib.pyplot as plt
        window = 50
        rewards = np.array(ep_rewards)
        smoothed = np.convolve(rewards,
                               np.ones(window)/window, mode='valid')
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rewards, alpha=0.3, color='steelblue', label='Raw reward')
        ax.plot(range(window-1, len(rewards)), smoothed,
                color='steelblue', linewidth=2, label=f'Smoothed ({window}ep)')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("MAPPO Training Curve – Multi-Agent Ball Delivery")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(log_dir, "training_curve.png")
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[Plot] Training curve saved → {path}")
    except ImportError:
        pass


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MAPPO Multi-Agent Ball Delivery")
    p.add_argument("--demo",           action="store_true",
                   help="Demo mode (requires pygame, loads saved model)")
    p.add_argument("--eval",           action="store_true",
                   help="Evaluation mode (no exploration)")
    p.add_argument("--render",         action="store_true",
                   help="Enable pygame rendering during training")
    p.add_argument("--no-comm",        action="store_true",
                   help="Disable communication module (ablation study)")
    p.add_argument("--episodes",       type=int, default=cfg.TOTAL_EPISODES,
                   help="Number of training episodes")
    p.add_argument("--eval-episodes",  type=int, default=100,
                   help="Number of evaluation episodes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.use_comm = not args.no_comm

    if args.demo:
        demo(args)
    elif args.eval:
        evaluate(args)
    else:
        train(args)