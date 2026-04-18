"""
env/environment.py

30x30 grid-world for cooperative multi-agent ball delivery.

Coordinate convention: (row, col) with row 0 at top.

Action map:
  0:UP(-row)  1:DOWN(+row)  2:LEFT(-col)  3:RIGHT(+col)
  4:STAY      5:PICK        6:DROP
"""

import numpy as np
from config import cfg


# ── Tile IDs (for rendering) ──────────────────────────────────────────────────
EMPTY    = 0
OBSTACLE = 1
BALL     = 2
BOX      = 3
AGENT    = 4

# Action deltas (row, col)
ACTION_DELTA = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
    4: ( 0,  0),   # STAY
}


class GridWorld:
    """
    Cooperative multi-agent grid world.

    Key design decisions
    --------------------
    * Compact local observations   → actor stays lightweight
    * Full global state            → centralized critic sees everything
    * Shaped rewards               → encourage efficient cooperative behavior
    * Curriculum-controlled obstacles via `set_phase()`
    """

    def __init__(self, num_obstacles: int = 0, seed: int | None = None):
        self.rng           = np.random.default_rng(seed)
        self.num_obstacles = num_obstacles
        self.grid_size     = cfg.GRID_SIZE
        self.n_agents      = cfg.NUM_AGENTS
        self.n_balls       = cfg.NUM_BALLS
        self.obs_radius    = cfg.OBS_RADIUS
        self.max_steps     = cfg.MAX_STEPS

        # Fixed drop-off box position (center-ish)
        self.box_pos = np.array([self.grid_size // 2, self.grid_size // 2])

        # State placeholders (filled on reset)
        self.agent_pos    = None   # (N, 2) int
        self.agent_carry  = None   # (N,)   bool
        self.ball_pos     = None   # (B, 2) int  ;  (-1,-1) = delivered
        self.obstacle_map = None   # (G, G) bool
        self.step_count   = 0
        self.episode_reward = np.zeros(self.n_agents)

        # Track previous positions for oscillation penalty
        self._prev_pos    = None
        self._prev_prev_pos = None

        # Cache flat obstacle array for global state
        self._obs_flat    = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> list[np.ndarray]:
        """Reset environment, return list of local observations."""
        self.step_count   = 0
        self.episode_reward = np.zeros(self.n_agents)
        self._place_entities()
        self._obs_flat = self.obstacle_map.flatten().astype(np.float32)
        self._prev_pos = self.agent_pos.copy()
        self._prev_prev_pos = self.agent_pos.copy()
        return self._get_observations()

    def step(self, actions: list[int]):
        """
        Execute one environment step.

        Returns
        -------
        observations : list[np.ndarray]
        rewards      : list[float]
        done         : bool
        info         : dict
        """
        rewards = np.zeros(self.n_agents)
        self.step_count += 1

        # ── 1. Movement phase ────────────────────────────────────────────────
        new_positions = self.agent_pos.copy()
        for i, a in enumerate(actions):
            if a in ACTION_DELTA:
                dr, dc = ACTION_DELTA[a]
                nr = self.agent_pos[i, 0] + dr
                nc = self.agent_pos[i, 1] + dc
                # Boundary + obstacle check
                if (0 <= nr < self.grid_size and
                    0 <= nc < self.grid_size and
                    not self.obstacle_map[nr, nc]):
                    new_positions[i] = [nr, nc]
                else:
                    rewards[i] += cfg.R_COLLISION  # hit wall/obstacle

        # ── 2. Agent-agent collision resolution ─────────────────────────────
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(new_positions[i], new_positions[j]):
                    # Revert both; penalize
                    new_positions[i] = self.agent_pos[i]
                    new_positions[j] = self.agent_pos[j]
                    rewards[i] += cfg.R_COLLISION
                    rewards[j] += cfg.R_COLLISION

        # ── 3. Oscillation penalty (back-and-forth) ───────────────────────
        for i in range(self.n_agents):
            if np.array_equal(new_positions[i], self._prev_prev_pos[i]):
                rewards[i] += cfg.R_OSCILLATION

        self._prev_prev_pos = self._prev_pos.copy()
        self._prev_pos      = self.agent_pos.copy()
        self.agent_pos      = new_positions

        # ── 4. PICK / DROP actions ───────────────────────────────────────────
        for i, a in enumerate(actions):
            if a == 5:   # PICK
                rewards[i] += self._try_pick(i)
            elif a == 6: # DROP
                rewards[i] += self._try_drop(i)

        # ── 5. Distance-shaping reward ───────────────────────────────────────
        for i in range(self.n_agents):
            rewards[i] += self._distance_reward(i)

        # ── 6. Per-step penalty ───────────────────────────────────────────────
        rewards += cfg.R_STEP

        # ── 7. Terminal conditions ────────────────────────────────────────────
        balls_left = np.sum(self.ball_pos[:, 0] >= 0)
        done = False
        if balls_left == 0:
            rewards += cfg.R_ALL_DONE / self.n_agents
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        self.episode_reward += rewards
        info = {
            "balls_remaining": int(balls_left),
            "step": self.step_count,
            "episode_reward": self.episode_reward.copy(),
        }
        return self._get_observations(), rewards.tolist(), done, info

    def get_global_state(self) -> np.ndarray:
        """
        Centralized critic input: flat concatenation of all state components.
        Shape: (GLOBAL_STATE_SIZE,)
        """
        parts = []
        # Normalized agent positions
        parts.append((self.agent_pos.flatten() / self.grid_size).astype(np.float32))
        # Normalized ball positions (delivered balls → 0-vector)
        ball_flat = self.ball_pos.copy().astype(np.float32)
        ball_flat[ball_flat < 0] = 0.0
        ball_flat /= self.grid_size
        parts.append(ball_flat.flatten())
        # Box position (normalized)
        parts.append((self.box_pos / self.grid_size).astype(np.float32))
        # Obstacle map (binary, flat)
        parts.append(self._obs_flat)
        return np.concatenate(parts)

    def set_num_obstacles(self, n: int):
        self.num_obstacles = n

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _place_entities(self):
        """Randomly place agents, balls, obstacles (avoid overlaps)."""
        occupied = set()

        # Box is fixed
        occupied.add(tuple(self.box_pos))

        # Obstacle map
        self.obstacle_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for _ in range(self.num_obstacles):
            pos = self._random_empty(occupied)
            if pos is None:
                break
            self.obstacle_map[pos] = True
            occupied.add(pos)

        # Agents
        self.agent_pos   = np.zeros((self.n_agents, 2), dtype=int)
        self.agent_carry = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            pos = self._random_empty(occupied)
            self.agent_pos[i] = pos
            occupied.add(pos)

        # Balls
        self.ball_pos = np.zeros((self.n_balls, 2), dtype=int)
        for b in range(self.n_balls):
            pos = self._random_empty(occupied)
            self.ball_pos[b] = pos
            occupied.add(pos)

    def _random_empty(self, occupied: set) -> tuple | None:
        """Sample a random unoccupied cell (max 1000 attempts)."""
        for _ in range(1000):
            r = int(self.rng.integers(0, self.grid_size))
            c = int(self.rng.integers(0, self.grid_size))
            if (r, c) not in occupied:
                return (r, c)
        return None

    def _try_pick(self, agent_idx: int) -> float:
        """Agent attempts to pick up a ball at its current position."""
        if self.agent_carry[agent_idx]:
            return cfg.R_INVALID_ACT  # already carrying
        pos = tuple(self.agent_pos[agent_idx])
        for b in range(self.n_balls):
            if self.ball_pos[b, 0] >= 0 and tuple(self.ball_pos[b]) == pos:
                self.agent_carry[agent_idx] = True
                self.ball_pos[b] = [-2, -2]  # mark as "in transit"
                return cfg.R_PICKUP
        return cfg.R_INVALID_ACT  # no ball here

    def _try_drop(self, agent_idx: int) -> float:
        """Agent attempts to drop ball at box location."""
        if not self.agent_carry[agent_idx]:
            return cfg.R_INVALID_ACT  # not carrying
        if np.array_equal(self.agent_pos[agent_idx], self.box_pos):
            self.agent_carry[agent_idx] = False
            # Mark first in-transit ball as delivered
            for b in range(self.n_balls):
                if self.ball_pos[b, 0] == -2:
                    self.ball_pos[b] = [-1, -1]  # delivered
                    break
            return cfg.R_DELIVERY
        return cfg.R_INVALID_ACT  # not at box

    def _distance_reward(self, agent_idx: int) -> float:
        """
        Dense shaping: reward proportional to moving closer to the target.
        Target = nearest ball if not carrying, else the box.
        """
        pos = self.agent_pos[agent_idx].astype(float)
        if not self.agent_carry[agent_idx]:
            # Find nearest available ball
            available = [b for b in range(self.n_balls) if self.ball_pos[b, 0] >= 0]
            if not available:
                return 0.0
            dists = [np.linalg.norm(pos - self.ball_pos[b]) for b in available]
            # Previous distance
            prev = self._prev_pos[agent_idx].astype(float)
            prev_dists = [np.linalg.norm(prev - self.ball_pos[b]) for b in available]
            delta = min(prev_dists) - min(dists)
        else:
            delta = (np.linalg.norm(self._prev_pos[agent_idx].astype(float) - self.box_pos) -
                     np.linalg.norm(pos - self.box_pos))
        return cfg.R_DIST_SCALE * delta

    def _get_observations(self) -> list[np.ndarray]:
        """Build compact local observations for all agents."""
        return [self._obs_for_agent(i) for i in range(self.n_agents)]

    def _obs_for_agent(self, idx: int) -> np.ndarray:
        """
        Local observation for agent `idx`.
        Components (all normalized to [-1, 1] or [0, 1]):
          [0:2]   agent position (normalized)
          [2]     carrying flag
          [3:5]   relative position to nearest available ball
          [5:7]   relative position to drop-off box
          [7:7+W] local obstacle window (flattened, W=(2r+1)^2)
          [7+W:]  relative positions of other agents
        """
        G = self.grid_size
        r = self.obs_radius
        W = (2 * r + 1) ** 2

        parts = []

        # Agent position (normalized)
        parts.append(self.agent_pos[idx] / G)

        # Carrying flag
        parts.append(np.array([float(self.agent_carry[idx])]))

        # Relative position to nearest available ball
        available = [b for b in range(self.n_balls) if self.ball_pos[b, 0] >= 0]
        if available:
            dists = [np.linalg.norm(self.agent_pos[idx] - self.ball_pos[b]) for b in available]
            nearest = available[int(np.argmin(dists))]
            rel_ball = (self.ball_pos[nearest] - self.agent_pos[idx]) / G
        else:
            rel_ball = np.zeros(2)
        parts.append(rel_ball.astype(np.float32))

        # Relative position to box
        rel_box = (self.box_pos - self.agent_pos[idx]) / G
        parts.append(rel_box.astype(np.float32))

        # Local obstacle window
        window = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
        pr, pc = self.agent_pos[idx]
        for dr in range(-r, r+1):
            for dc in range(-r, r+1):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < G and 0 <= nc < G:
                    window[dr+r, dc+r] = float(self.obstacle_map[nr, nc])
                else:
                    window[dr+r, dc+r] = 1.0  # treat out-of-bounds as obstacle
        parts.append(window.flatten())

        # Relative positions of other agents
        for j in range(self.n_agents):
            if j != idx:
                rel = (self.agent_pos[j] - self.agent_pos[idx]) / G
                parts.append(rel.astype(np.float32))

        return np.concatenate(parts).astype(np.float32)