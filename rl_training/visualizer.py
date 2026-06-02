"""
Pygame visualizer for the multi-agent warehouse simulation.

Features:
  - Real-time grid rendering with warehouse aesthetics
  - Robot movement with color trails
  - Object/pickup/delivery indicators
  - Live metrics overlay
  - Step-by-step or continuous playback
"""

import pygame
import sys
import config as cfg

# Initialize fonts after pygame init
_font = None
_small_font = None
_title_font = None


def _init_fonts():
    global _font, _small_font, _title_font
    if _font is None:
        pygame.font.init()
        _font = pygame.font.SysFont("Menlo", 14)
        _small_font = pygame.font.SysFont("Menlo", 11)
        _title_font = pygame.font.SysFont("Menlo", 18, bold=True)


class WarehouseVisualizer:
    """Real-time Pygame renderer for the warehouse environment."""

    def __init__(self, env, title="Multi-Agent Warehouse", mode_label="BFS"):
        pygame.init()
        _init_fonts()

        self.env = env
        self.mode_label = mode_label
        self.cell = cfg.CELL_SIZE
        self.pad = cfg.WINDOW_PADDING

        # Window size
        grid_w = cfg.GRID_COLS * self.cell
        grid_h = cfg.GRID_ROWS * self.cell
        panel_w = 260  # side panel for metrics
        self.win_w = grid_w + panel_w + self.pad * 3
        self.win_h = grid_h + self.pad * 2

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        # Robot trail history
        self.trails = {i: [] for i in range(cfg.NUM_ROBOTS)}

    def draw(self, metrics=None, step_info=None):
        """Draw one frame."""
        self.screen.fill(cfg.COLOR_BG)
        self._draw_grid()
        self._draw_objects()
        self._draw_drop_zone()
        self._draw_trails()
        self._draw_robots()
        self._draw_panel(metrics, step_info)
        pygame.display.flip()
        self.clock.tick(cfg.FPS)

    def _draw_grid(self):
        """Draw warehouse grid."""
        for r in range(cfg.GRID_ROWS):
            for c in range(cfg.GRID_COLS):
                x = self.pad + c * self.cell
                y = self.pad + r * self.cell
                cell_type = self.env.grid[r][c]

                if cell_type == cfg.WALL:
                    color = cfg.COLOR_WALL
                elif cell_type == cfg.DROP_ZONE:
                    color = cfg.COLOR_DROP_ZONE
                else:
                    color = cfg.COLOR_FLOOR

                pygame.draw.rect(self.screen, color, (x, y, self.cell, self.cell))
                pygame.draw.rect(self.screen, cfg.COLOR_GRID_LINE,
                                 (x, y, self.cell, self.cell), 1)

    def _draw_objects(self):
        """Draw collectible objects."""
        for idx, obj_pos in enumerate(self.env.objects):
            if self.env.objects_collected[idx]:
                continue
            r, c = obj_pos
            x = self.pad + c * self.cell + self.cell // 2
            y = self.pad + r * self.cell + self.cell // 2
            # Golden diamond shape
            size = self.cell // 3
            points = [
                (x, y - size),
                (x + size, y),
                (x, y + size),
                (x - size, y),
            ]
            pygame.draw.polygon(self.screen, cfg.COLOR_OBJECT, points)
            pygame.draw.polygon(self.screen, (200, 160, 0), points, 2)
            # Label
            label = _small_font.render(f"O{idx}", True, (0, 0, 0))
            self.screen.blit(label, (x - 7, y - 5))

    def _draw_drop_zone(self):
        """Draw drop zone label."""
        r, c = cfg.DROP_ZONE_CENTER
        x = self.pad + c * self.cell + self.cell // 2
        y = self.pad + r * self.cell + self.cell // 2
        label = _font.render("DROP", True, (255, 255, 255))
        self.screen.blit(label, (x - 16, y - 7))

    def _draw_trails(self):
        """Draw movement trails for each robot."""
        for rid in range(cfg.NUM_ROBOTS):
            trail = self.trails[rid]
            if len(trail) < 2:
                continue
            color = cfg.ROBOT_COLORS_RGB[rid]
            faded = (color[0] // 3, color[1] // 3, color[2] // 3)
            for i in range(len(trail) - 1):
                r1, c1 = trail[i]
                r2, c2 = trail[i + 1]
                x1 = self.pad + c1 * self.cell + self.cell // 2
                y1 = self.pad + r1 * self.cell + self.cell // 2
                x2 = self.pad + c2 * self.cell + self.cell // 2
                y2 = self.pad + r2 * self.cell + self.cell // 2
                pygame.draw.line(self.screen, faded, (x1, y1), (x2, y2), 2)

    def _draw_robots(self):
        """Draw robots as colored circles with labels."""
        for rid in range(cfg.NUM_ROBOTS):
            r, c = self.env.robot_positions[rid]
            x = self.pad + c * self.cell + self.cell // 2
            y = self.pad + r * self.cell + self.cell // 2
            color = cfg.ROBOT_COLORS_RGB[rid]

            # Update trail
            if not self.trails[rid] or self.trails[rid][-1] != (r, c):
                self.trails[rid].append((r, c))
                # Keep last 30 positions
                if len(self.trails[rid]) > 30:
                    self.trails[rid] = self.trails[rid][-30:]

            radius = self.cell // 3

            if self.env.robot_failed[rid]:
                # X mark for failed robots
                pygame.draw.line(self.screen, (200, 0, 0),
                                 (x - radius, y - radius), (x + radius, y + radius), 3)
                pygame.draw.line(self.screen, (200, 0, 0),
                                 (x + radius, y - radius), (x - radius, y + radius), 3)
            elif self.env.robot_done[rid]:
                # Checkmark for done robots
                pygame.draw.circle(self.screen, (100, 100, 100), (x, y), radius)
                label = _small_font.render("✓", True, (0, 255, 0))
                self.screen.blit(label, (x - 5, y - 6))
            else:
                # Normal robot
                pygame.draw.circle(self.screen, color, (x, y), radius)
                if self.env.robot_carrying[rid]:
                    # Inner golden dot = carrying
                    pygame.draw.circle(self.screen, cfg.COLOR_OBJECT, (x, y), radius // 2)

                # Robot ID
                label = _small_font.render(f"R{rid}", True, (0, 0, 0))
                self.screen.blit(label, (x - 7, y - 5))

                # Direction indicator (target)
                target = self.env._get_current_target(rid)
                if target:
                    tr, tc = target
                    tx = self.pad + tc * self.cell + self.cell // 2
                    ty = self.pad + tr * self.cell + self.cell // 2
                    pygame.draw.line(self.screen, (*color[:3],),
                                     (x, y), (tx, ty), 1)

    def _draw_panel(self, metrics=None, step_info=None):
        """Draw side panel with metrics."""
        panel_x = self.pad * 2 + cfg.GRID_COLS * self.cell
        y = self.pad

        # Title
        title = _title_font.render(f"Mode: {self.mode_label}", True, (255, 255, 255))
        self.screen.blit(title, (panel_x, y))
        y += 30

        # Separator
        pygame.draw.line(self.screen, (80, 80, 100),
                         (panel_x, y), (panel_x + 230, y), 1)
        y += 15

        if metrics:
            items = [
                ("Steps", str(metrics.get("steps", 0))),
                ("Collisions", str(metrics.get("collisions", 0))),
                ("Pickups", str(metrics.get("pickups", 0))),
                ("Deliveries", str(metrics.get("deliveries", 0))),
                ("Completion", f"{metrics.get('completion', 0):.0%}"),
            ]
            for label, val in items:
                text = _font.render(f"{label}: {val}", True, cfg.COLOR_TEXT)
                self.screen.blit(text, (panel_x, y))
                y += 22

        y += 15
        pygame.draw.line(self.screen, (80, 80, 100),
                         (panel_x, y), (panel_x + 230, y), 1)
        y += 15

        # Robot status
        header = _font.render("Robot Status:", True, (255, 255, 255))
        self.screen.blit(header, (panel_x, y))
        y += 22

        for rid in range(cfg.NUM_ROBOTS):
            color = cfg.ROBOT_COLORS_RGB[rid]
            name = cfg.ROBOT_NAMES[rid]
            pos = self.env.robot_positions[rid]

            if self.env.robot_failed[rid]:
                status = "FAILED"
                scolor = (231, 76, 60)
            elif self.env.robot_done[rid]:
                status = "DONE"
                scolor = (46, 204, 113)
            elif self.env.robot_carrying[rid]:
                status = "CARRYING"
                scolor = cfg.COLOR_OBJECT
            elif rid in self.env.assignments:
                obj_idx = self.env.assignments[rid]
                status = f"→ O{obj_idx}"
                scolor = color
            else:
                status = "IDLE"
                scolor = (150, 150, 150)

            pygame.draw.circle(self.screen, color, (panel_x + 8, y + 8), 6)
            text = _font.render(f"{name} {pos} {status}", True, scolor)
            self.screen.blit(text, (panel_x + 22, y))
            y += 22

        # Legend
        y += 20
        pygame.draw.line(self.screen, (80, 80, 100),
                         (panel_x, y), (panel_x + 230, y), 1)
        y += 10
        legend_items = [
            (cfg.COLOR_WALL, "Shelving/Wall"),
            (cfg.COLOR_DROP_ZONE, "Drop Zone"),
            (cfg.COLOR_OBJECT, "Object"),
            (cfg.COLOR_FLOOR, "Aisle"),
        ]
        for color, label in legend_items:
            pygame.draw.rect(self.screen, color, (panel_x, y, 14, 14))
            text = _small_font.render(label, True, cfg.COLOR_TEXT)
            self.screen.blit(text, (panel_x + 20, y + 1))
            y += 18

        # Controls
        y += 15
        controls = _small_font.render("[SPACE] pause  [Q] quit", True, (120, 120, 140))
        self.screen.blit(controls, (panel_x, y))

    def handle_events(self):
        """Handle Pygame events. Returns False if quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                if event.key == pygame.K_SPACE:
                    # Pause
                    self._pause()
        return True

    def _pause(self):
        """Pause until space pressed again."""
        paused = True
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

    def close(self):
        """Close pygame window."""
        pygame.quit()
