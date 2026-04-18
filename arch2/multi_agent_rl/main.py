# main.py — Entry point: simulation loop + Pygame rendering
# Run with: python main.py

import sys
import os
import pygame
import random

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C
from env.environment  import Environment
from agents.agent     import Agent
from utils.utils      import log_episode

# ── Keyboard→Action mapping ───────────────────────────────────────────────────
pygame.init()


# ═════════════════════════════════════════════════════════════════════════════
# RENDERER — all Pygame drawing lives here
# ═════════════════════════════════════════════════════════════════════════════

class Renderer:
    """Handles all Pygame rendering: grid, entities, HUD."""

    def __init__(self):
        self.screen = pygame.display.set_mode((C.WINDOW_W, C.WINDOW_H))
        pygame.display.set_caption(C.WINDOW_TITLE)
        self.clock  = pygame.time.Clock()

        # Fonts
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 22, bold=True)

    # ──────────────────────────────────────────
    def draw_frame(self, env: Environment, agents: list, episode: int,
                   mode: str, total_collisions: int):
        """Render one complete frame."""
        self.screen.fill(C.COLOR_BG)
        self._draw_grid()
        self._draw_obstacles(env)
        self._draw_balls(env)
        self._draw_agents(agents)
        self._draw_hud(env, agents, episode, mode, total_collisions)
        pygame.display.flip()
        self.clock.tick(C.FPS)

    # ──────────────────────────────────────────
    def _draw_grid(self):
        for r in range(C.GRID_ROWS + 1):
            y = r * C.CELL_SIZE
            pygame.draw.line(self.screen, C.COLOR_GRID_LINE,
                             (0, y), (C.WINDOW_W, y), 1)
        for c in range(C.GRID_COLS + 1):
            x = c * C.CELL_SIZE
            pygame.draw.line(self.screen, C.COLOR_GRID_LINE,
                             (x, 0), (x, C.GRID_ROWS * C.CELL_SIZE), 1)

    # ──────────────────────────────────────────
    def _draw_obstacles(self, env: Environment):
        for (r, c) in env.obstacles:
            rect = pygame.Rect(
                c * C.CELL_SIZE + 2, r * C.CELL_SIZE + 2,
                C.CELL_SIZE - 4,    C.CELL_SIZE - 4
            )
            pygame.draw.rect(self.screen, C.COLOR_OBSTACLE, rect, border_radius=4)

    # ──────────────────────────────────────────
    def _draw_balls(self, env: Environment):
        radius = C.CELL_SIZE // 2 - 4
        for (r, c) in env.balls:
            cx = c * C.CELL_SIZE + C.CELL_SIZE // 2
            cy = r * C.CELL_SIZE + C.CELL_SIZE // 2
            # Outer glow
            pygame.draw.circle(self.screen, (180, 140, 0), (cx, cy), radius + 3)
            # Main ball
            pygame.draw.circle(self.screen, C.COLOR_BALL, (cx, cy), radius)
            # Shine spot
            pygame.draw.circle(self.screen, (255, 255, 200),
                               (cx - radius // 3, cy - radius // 3), radius // 4)

    # ──────────────────────────────────────────
    def _draw_agents(self, agents: list):
        for agent in agents:
            r, c   = agent.pos
            cx     = c * C.CELL_SIZE + C.CELL_SIZE // 2
            cy     = r * C.CELL_SIZE + C.CELL_SIZE // 2
            radius = C.CELL_SIZE // 2 - 3

            # Shadow
            pygame.draw.circle(self.screen, (0, 0, 0),
                               (cx + 2, cy + 2), radius)
            # Body
            pygame.draw.circle(self.screen, agent.color, (cx, cy), radius)
            # Border
            pygame.draw.circle(self.screen, (255, 255, 255),
                               (cx, cy), radius, 2)
            # ID label
            label = self.font_sm.render(str(agent.agent_id), True, (0, 0, 0))
            self.screen.blit(label, (cx - label.get_width() // 2,
                                      cy - label.get_height() // 2))

    # ──────────────────────────────────────────
    def _draw_hud(self, env: Environment, agents: list,
                  episode: int, mode: str, total_collisions: int):
        """Render the HUD panel below the grid."""
        hud_top = C.GRID_ROWS * C.CELL_SIZE
        pygame.draw.rect(self.screen, C.COLOR_HUD_BG,
                         (0, hud_top, C.WINDOW_W, C.HUD_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 90),
                         (0, hud_top), (C.WINDOW_W, hud_top), 2)

        mode_color = (0, 220, 255) if mode == "rl" else (255, 180, 0)
        mode_label = f"MODE: {'RL Q-LEARNING' if mode == 'rl' else 'RULE-BASED BFS'}"

        # Row 1 — title + mode
        title = self.font_title.render(
            f"Multi-Agent RL   Ep {episode}", True, C.COLOR_TEXT
        )
        mode_surf = self.font_lg.render(mode_label, True, mode_color)
        self.screen.blit(title,     (10,  hud_top + 8))
        self.screen.blit(mode_surf, (C.WINDOW_W - mode_surf.get_width() - 10,
                                      hud_top + 8))

        # Row 2 — stats
        stats = (
            f"Steps: {env.step_count:>4}   "
            f"Balls Left: {len(env.balls):>2}   "
            f"Collisions: {total_collisions:>4}   "
            f"ε: {agents[0].epsilon:.3f}   "
            f"Q-states: {agents[0].q_table_size:>6}"
        )
        stats_surf = self.font_sm.render(stats, True, C.COLOR_TEXT_DIM)
        self.screen.blit(stats_surf, (10, hud_top + 36))

        # Row 3 — per-agent info
        x = 10
        for ag in agents:
            col_swatch = pygame.Rect(x, hud_top + 62, 12, 12)
            pygame.draw.rect(self.screen, ag.color, col_swatch)
            info = self.font_sm.render(
                f" A{ag.agent_id}: {ag.balls_collected}balls  "
                f"r={ag.episode_reward:+.0f}",
                True, C.COLOR_TEXT_DIM
            )
            self.screen.blit(info, (x + 14, hud_top + 60))
            x += info.get_width() + 28

        # Toggle hint
        hint = self.font_sm.render(
            "  [T] Toggle mode   [S] Save   [L] Load   [Q] Quit",
            True, (70, 70, 100)
        )
        self.screen.blit(hint, (C.WINDOW_W - hint.get_width() - 8,
                                 hud_top + 60))


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run():
    """Main training + rendering loop."""
    renderer         = Renderer()
    env              = Environment()
    mode             = "rl"   # "rl" or "rule"

    # Instantiate agents
    agents = [Agent(i, mode=mode) for i in range(C.NUM_AGENTS)]

    # Try loading saved models
    for ag in agents:
        ag.load()

    episode          = 1
    total_collisions = 0

    # ── Episode start ─────────────────────────
    start_positions = env.reset()
    for i, ag in enumerate(agents):
        ag.reset(start_positions[i])
        ag.mode = mode
        # Initialise distance-to-ball for shaping
        from utils.utils import nearest_ball as _nb
        d, _ = _nb(ag.pos, env.balls)
        ag.prev_dist_to_ball = d if d != float("inf") else C.GRID_ROWS + C.GRID_COLS

    running = True
    while running:

        # ── Events ────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_t:
                    # Toggle between RL and rule-based
                    mode = "rule" if mode == "rl" else "rl"
                    for ag in agents:
                        ag.mode = mode
                    print(f"[Mode] Switched to: {mode.upper()}")

                elif event.key == pygame.K_s:
                    for ag in agents:
                        ag.save()
                    print("[Save] Q-tables saved.")

                elif event.key == pygame.K_l:
                    for ag in agents:
                        ag.load()
                    print("[Load] Q-tables loaded.")

        # ── Collect actions ───────────────────
        proposed_positions = []
        proposed_actions   = []

        other_positions = [ag.pos for ag in agents]

        for ag in agents:
            others = [p for j, p in enumerate(other_positions)
                      if j != ag.agent_id]
            pos, action = ag.select_action(env.balls, others, env.obstacles)
            proposed_positions.append(pos)
            proposed_actions.append(action)

        # ── Environment step ──────────────────
        rewards, new_positions, done, info = env.step(proposed_positions)
        total_collisions += info["collisions_step"]

        # ── Update agent positions & learn ────
        for i, ag in enumerate(agents):
            ag.pos = new_positions[i]
            if info["collected"][i]:
                ag.balls_collected += 1

            others_new = [new_positions[j] for j in range(C.NUM_AGENTS) if j != i]
            ag.learn(rewards[i], env.balls, others_new, done)

        # ── Render ────────────────────────────
        renderer.draw_frame(env, agents, episode, mode, total_collisions)

        # ── Episode end ───────────────────────
        if done:
            print(
                f"Ep {episode:>5} | Steps {env.step_count:>4} | "
                f"Balls {env.balls_collected:>2}/10 | "
                f"Collisions {env.collision_count:>3} | "
                f"ε {agents[0].epsilon:.4f} | "
                f"Mode {mode}"
            )

            # Log stats
            log_episode(
                episode, env.step_count, env.collision_count,
                env.balls_collected, agents[0].epsilon
            )

            # Save Q-tables periodically
            if episode % C.SAVE_EVERY == 0:
                for ag in agents:
                    ag.save()
                print(f"[Save] Q-tables saved at episode {episode}.")

            # End-of-episode bookkeeping per agent
            for ag in agents:
                ag.end_episode()

            episode += 1

            # ── Reset for next episode ────────
            start_positions = env.reset()
            for i, ag in enumerate(agents):
                ag.reset(start_positions[i])
                ag.mode = mode
                from utils.utils import nearest_ball as _nb
                d, _ = _nb(ag.pos, env.balls)
                ag.prev_dist_to_ball = d if d != float("inf") else C.GRID_ROWS + C.GRID_COLS

    # ── Shutdown ──────────────────────────────
    print("Saving models before exit...")
    for ag in agents:
        ag.save()
    pygame.quit()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()
