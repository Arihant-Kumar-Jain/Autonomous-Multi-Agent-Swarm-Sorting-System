"""
env/environment.py - 30×30 cooperative ball delivery world (arch4)

Key improvements over arch3:
  • ALL ball positions in local observation (not just nearest)
    -> agents can implicitly reason about what teammates are targeting
  • Compact global state (42 dims, no raw 900-cell obstacle map)
  • Rebalanced rewards: soft step penalty, strong distance shaping,
    coordination claim bonus, milder collision penalties
  • Ball "claimed" tracking to enable coordination bonus
  • Separate wall-hit vs agent-collision penalties

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

# Ball status codes (stored in ball_status array)
BALL_AVAILABLE  = 0   # on floor, can be picked up
BALL_IN_TRANSIT = 1   # carried by an agent
BALL_DELIVERED  = 2   # dropped at box


class GridWorld:
    """
    Cooperative multi-agent grid world (arch4).

    Design decisions
    ----------------
    * ALL balls in local obs  -> agents can see what teammates are chasing
    * Compact global state    -> critic learns efficiently (42 dims vs 914)
    * Claim bonus             -> emergent division-of-labor reward
    * Soft penalties          -> exploration is not too costly
    * Performance curriculum  -> advances only when agents actually improve
    """

    def __init__(self, num_obstacles: int = 0, seed: int | None = None):
        self.rng           = np.random.default_rng(seed)
        self.num_obstacles = num_obstacles
        self.grid_size     = cfg.GRID_SIZE
        self.n_agents      = cfg.NUM_AGENTS
        self.n_balls       = cfg.NUM_BALLS
        self.obs_radius    = 2          # 5×5 local obstacle window (was 5 -> 11×11=121)
        self.max_steps     = cfg.MAX_STEPS

        # Fixed drop-off box position (center-ish)
        self.box_pos = np.array([self.grid_size // 2, self.grid_size // 2])

        # State placeholders
        self.agent_pos    = None    # (N, 2) int
        self.agent_carry  = None    # (N,)   bool
        self.ball_pos     = None    # (B, 2) int  (may be [-1,-1] once delivered)
        self.ball_status  = None    # (B,)   int  0=available 1=transit 2=delivered
        self.obstacle_map = None    # (G, G) bool
        self.step_count   = 0
        self.episode_reward = np.zeros(self.n_agents)

        # Previous positions (for distance reward)
        self._prev_pos = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> list[np.ndarray]:
        """Reset environment, return list of local observations."""
        self.step_count     = 0
        self.episode_reward = np.zeros(self.n_agents)
        self._place_entities()
        self._prev_pos = self.agent_pos.copy()
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

        # ── 1. Movement phase ─────────────────────────────────────────────────
        new_positions = self.agent_pos.copy()
        for i, a in enumerate(actions):
            if a in ACTION_DELTA:
                dr, dc = ACTION_DELTA[a]
                nr = self.agent_pos[i, 0] + dr
                nc = self.agent_pos[i, 1] + dc
                if (0 <= nr < self.grid_size and
                    0 <= nc < self.grid_size and
                    not self.obstacle_map[nr, nc]):
                    new_positions[i] = [nr, nc]
                else:
                    rewards[i] += cfg.R_WALL_HIT   # mild wall/obstacle penalty

        # ── 2. Agent-agent collision resolution ──────────────────────────────
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(new_positions[i], new_positions[j]):
                    # Revert both to old positions; penalize
                    new_positions[i] = self.agent_pos[i]
                    new_positions[j] = self.agent_pos[j]
                    rewards[i] += cfg.R_AGENT_COLL
                    rewards[j] += cfg.R_AGENT_COLL

        self._prev_pos = self.agent_pos.copy()
        self.agent_pos = new_positions

        # ── 3. PICK / DROP actions ────────────────────────────────────────────
        for i, a in enumerate(actions):
            if a == 5:   # PICK
                rewards[i] += self._try_pick(i)
            elif a == 6: # DROP
                rewards[i] += self._try_drop(i)

        # ── 4. Distance-shaping reward ────────────────────────────────────────
        for i in range(self.n_agents):
            rewards[i] += self._distance_reward(i)

        # ── 5. Coordination claim bonus ───────────────────────────────────────
        # Reward agents for being the closest to an unclaimed ball.
        # This incentivizes different agents to head for different balls.
        rewards += self._claim_bonuses()

        # ── 6. Per-step penalty ───────────────────────────────────────────────
        rewards += cfg.R_STEP

        # ── 7. Terminal conditions ────────────────────────────────────────────
        balls_left = int(np.sum(self.ball_status == BALL_AVAILABLE) +
                         np.sum(self.ball_status == BALL_IN_TRANSIT))
        done = False
        if np.all(self.ball_status == BALL_DELIVERED):
            rewards += cfg.R_ALL_DONE / self.n_agents
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        self.episode_reward += rewards
        info = {
            "balls_remaining": int(np.sum(self.ball_status != BALL_DELIVERED)),
            "balls_delivered": int(np.sum(self.ball_status == BALL_DELIVERED)),
            "step": self.step_count,
            "episode_reward": self.episode_reward.copy(),
        }
        return self._get_observations(), rewards.tolist(), done, info

    def get_global_state(self) -> np.ndarray:
        """
        Compact centralized critic input (42 dims).

        Components:
          agent_pos (3×2=6) + agent_carry (3) + all_ball_pos (10×2=20 normalized)
          + ball_status_normalized (10) + box_pos (2) + step_fraction (1)
        """
        G = self.grid_size
        parts = []
        # Agent positions (normalized)
        parts.append((self.agent_pos.flatten() / G).astype(np.float32))
        # Agent carry flags
        parts.append(self.agent_carry.astype(np.float32))
        # All ball positions (use center of box as proxy for delivered/transit)
        ball_pos_norm = np.zeros((self.n_balls, 2), dtype=np.float32)
        for b in range(self.n_balls):
            if self.ball_status[b] == BALL_AVAILABLE:
                ball_pos_norm[b] = self.ball_pos[b] / G
            else:
                ball_pos_norm[b] = self.box_pos / G  # logically at/toward box
        parts.append(ball_pos_norm.flatten())
        # Ball status (0=available, 0.5=transit, 1=delivered) - normalized
        status_norm = self.ball_status.astype(np.float32) / 2.0
        parts.append(status_norm)
        # Box position (normalized)
        parts.append((self.box_pos / G).astype(np.float32))
        # Step fraction
        parts.append(np.array([self.step_count / self.max_steps], dtype=np.float32))

        state = np.concatenate(parts)
        assert state.shape[0] == cfg.GLOBAL_STATE_SIZE, \
            f"Global state size mismatch: {state.shape[0]} vs {cfg.GLOBAL_STATE_SIZE}"
        return state

    def set_num_obstacles(self, n: int):
        self.num_obstacles = n

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _place_entities(self):
        """Randomly place agents, balls, and obstacles (no overlaps)."""
        occupied = set()

        # Box is fixed
        occupied.add(tuple(self.box_pos))

        # Obstacles
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
        self.ball_pos    = np.zeros((self.n_balls, 2), dtype=int)
        self.ball_status = np.zeros(self.n_balls, dtype=int)  # all AVAILABLE
        for b in range(self.n_balls):
            pos = self._random_empty(occupied)
            self.ball_pos[b] = pos
            occupied.add(pos)

    def _random_empty(self, occupied: set) -> tuple | None:
        for _ in range(1000):
            r = int(self.rng.integers(0, self.grid_size))
            c = int(self.rng.integers(0, self.grid_size))
            if (r, c) not in occupied:
                return (r, c)
        return None

    def _try_pick(self, agent_idx: int) -> float:
        """Agent attempts to pick up a ball at its current position."""
        if self.agent_carry[agent_idx]:
            return cfg.R_INVALID_ACT   # already carrying

        pos = tuple(self.agent_pos[agent_idx])
        for b in range(self.n_balls):
            if self.ball_status[b] == BALL_AVAILABLE and tuple(self.ball_pos[b]) == pos:
                self.agent_carry[agent_idx] = True
                self.ball_status[b] = BALL_IN_TRANSIT
                return cfg.R_PICKUP
        return cfg.R_INVALID_ACT   # no ball here

    def _try_drop(self, agent_idx: int) -> float:
        """Agent attempts to drop ball at box location."""
        if not self.agent_carry[agent_idx]:
            return cfg.R_INVALID_ACT   # not carrying

        if np.array_equal(self.agent_pos[agent_idx], self.box_pos):
            self.agent_carry[agent_idx] = False
            # Mark the first in-transit ball as delivered
            for b in range(self.n_balls):
                if self.ball_status[b] == BALL_IN_TRANSIT:
                    self.ball_status[b] = BALL_DELIVERED
                    break
            return cfg.R_DELIVERY
        return cfg.R_INVALID_ACT   # not at box

    def _distance_reward(self, agent_idx: int) -> float:
        """
        Dense shaping: reward ∝ reduction in distance to next target.
        If not carrying -> target = nearest AVAILABLE ball
        If carrying     -> target = box
        """
        pos  = self.agent_pos[agent_idx].astype(float)
        prev = self._prev_pos[agent_idx].astype(float)

        if not self.agent_carry[agent_idx]:
            available = np.where(self.ball_status == BALL_AVAILABLE)[0]
            if len(available) == 0:
                return 0.0
            dists      = [np.linalg.norm(pos  - self.ball_pos[b]) for b in available]
            prev_dists = [np.linalg.norm(prev - self.ball_pos[b]) for b in available]
            delta = min(prev_dists) - min(dists)
        else:
            delta = (np.linalg.norm(prev - self.box_pos) -
                     np.linalg.norm(pos  - self.box_pos))

        return cfg.R_DIST_SCALE * delta

    def _claim_bonuses(self) -> np.ndarray:
        """
        Coordination claim bonus:
        For each available (unclaimed) ball, find the closest agent.
        That agent gets a small bonus. This encourages agents to spread out
        and cover different balls (division of labor).
        """
        bonuses = np.zeros(self.n_agents)
        available = np.where(self.ball_status == BALL_AVAILABLE)[0]
        for b in available:
            dists = [np.linalg.norm(self.agent_pos[i].astype(float) -
                                    self.ball_pos[b].astype(float))
                     for i in range(self.n_agents)]
            closest = int(np.argmin(dists))
            # Bonus inversely proportional to distance (closer -> bigger bonus)
            d = max(dists[closest], 1.0)
            bonuses[closest] += cfg.R_CLAIM_BONUS / d
        return bonuses

    def _get_observations(self) -> list[np.ndarray]:
        return [self._obs_for_agent(i) for i in range(self.n_agents)]

    def _obs_for_agent(self, idx: int) -> np.ndarray:
        """
        Local observation for agent `idx` (67 dims total).

        Layout:
          [0:2]     agent position (normalized)
          [2]       carrying flag
          [3:5]     relative position to box
          [5:25]    all ball relative positions (10×2, available balls have real pos,
                    transit/delivered balls get (0,0) proxy)
          [25:35]   ball status one-hot-like (0=avail, 0.5=transit, 1=delivered)
          [35:41]   other agents relative pos (2 × 2) + carry (2 × 1)  -> 6 dims
          [41:66]   local 5×5 obstacle window (25 dims)
          [66]      step fraction
        """
        G = self.grid_size
        r = self.obs_radius   # 2 -> 5×5 window
        parts = []

        # Agent position (normalized to [0,1])
        parts.append((self.agent_pos[idx] / G).astype(np.float32))       # 2

        # Carrying flag
        parts.append(np.array([float(self.agent_carry[idx])], dtype=np.float32))  # 1

        # Relative position to box
        rel_box = (self.box_pos - self.agent_pos[idx]) / G
        parts.append(rel_box.astype(np.float32))                          # 2

        # All ball relative positions + status
        ball_rel = np.zeros((self.n_balls, 2), dtype=np.float32)
        for b in range(self.n_balls):
            if self.ball_status[b] == BALL_AVAILABLE:
                ball_rel[b] = (self.ball_pos[b] - self.agent_pos[idx]) / G
            # transit/delivered -> stays 0,0 (invisible to this agent)
        parts.append(ball_rel.flatten())                                   # 20

        # Ball status (normalized: 0, 0.5, 1.0)
        ball_stat = (self.ball_status.astype(np.float32)) / 2.0
        parts.append(ball_stat)                                            # 10

        # Other agents: relative position + carry
        for j in range(self.n_agents):
            if j != idx:
                rel = (self.agent_pos[j] - self.agent_pos[idx]) / G
                parts.append(rel.astype(np.float32))                      # 2
                parts.append(np.array([float(self.agent_carry[j])],
                                      dtype=np.float32))                   # 1
        # -> 2 other agents × 3 = 6 dims

        # Local 5×5 obstacle window (r=2 -> 5×5=25 cells)
        window = np.zeros((2*r+1, 2*r+1), dtype=np.float32)
        pr, pc = self.agent_pos[idx]
        for dr in range(-r, r+1):
            for dc in range(-r, r+1):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < G and 0 <= nc < G:
                    window[dr+r, dc+r] = float(self.obstacle_map[nr, nc])
                else:
                    window[dr+r, dc+r] = 1.0   # out-of-bounds = virtual obstacle
        parts.append(window.flatten())                                     # 25

        # Step fraction
        parts.append(np.array([self.step_count / self.max_steps],
                               dtype=np.float32))                          # 1

        obs = np.concatenate(parts).astype(np.float32)
        # Total: 2+1+2+20+10+6+25+1 = 67
        return obs
