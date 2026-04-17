# env/environment.py — Grid world, ball placement, collision handling, reward computation
# This is the "ground truth" of the simulation; agents observe and act on this.

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from utils.utils import manhattan_distance, nearest_ball


class Environment:
    """
    Centralised 2-D grid environment shared by all agents.

    Responsibilities:
    • Place balls and obstacles at episode start
    • Track ball collection
    • Resolve movement conflicts (no two agents on same cell)
    • Compute per-agent rewards after each step
    • Detect episode completion
    """

    def __init__(self):
        self.rows      = C.GRID_ROWS
        self.cols      = C.GRID_COLS
        self.reset()

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────

    def reset(self):
        """
        Randomise ball positions, obstacle positions, and agent start positions.
        Called at the beginning of every episode.
        Returns initial positions for agents.
        """
        # Place static obstacles (fixed set each episode for simplicity;
        # change to random per-episode if desired)
        self.obstacles: set = self._place_obstacles()

        # Scatter balls (non-overlapping with obstacles and each other)
        self.balls: set = self._place_items(
            count=C.NUM_BALLS,
            exclude=self.obstacles
        )

        # Assign agent start positions
        exclude = self.obstacles | self.balls
        self.agent_positions: list = []
        for _ in range(C.NUM_AGENTS):
            pos = self._random_empty(exclude)
            self.agent_positions.append(pos)
            exclude.add(pos)

        # Tracking
        self.balls_collected  = 0
        self.step_count       = 0
        self.collision_count  = 0
        self.done             = False

        return list(self.agent_positions)

    # ─────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────

    def step(self, proposed_moves: list):
        """
        Apply one simultaneous step for all agents.

        proposed_moves: list of (new_row, new_col) — one per agent, in order.

        Returns:
            rewards         : list[float]   — per-agent reward this step
            new_positions   : list[tuple]   — resolved positions
            done            : bool          — True when all balls collected or max steps
            info            : dict          — extra diagnostics
        """
        self.step_count += 1
        old_positions = list(self.agent_positions)
        resolved      = self._resolve_collisions(old_positions, proposed_moves)

        rewards   = [0.0] * C.NUM_AGENTS
        collected = [False] * C.NUM_AGENTS

        # ── Reward: step penalty ──────────────────
        # Discourages agents from dawdling.
        for i in range(C.NUM_AGENTS):
            rewards[i] += C.REWARD_STEP_PENALTY

        # ── Reward: collision penalty ─────────────
        # Agent is penalised if its proposed move was blocked (collision).
        collisions_this_step = 0
        for i in range(C.NUM_AGENTS):
            if resolved[i] != proposed_moves[i]:
                # Movement was blocked → collision occurred
                rewards[i] += C.REWARD_COLLISION
                collisions_this_step += 1

        self.collision_count += collisions_this_step

        # ── Apply resolved positions ──────────────
        self.agent_positions = resolved

        # ── Reward: ball collection ───────────────
        # Most important reward signal — direct task completion.
        for i in range(C.NUM_AGENTS):
            pos = self.agent_positions[i]
            if pos in self.balls:
                self.balls.remove(pos)
                self.balls_collected += 1
                rewards[i] += C.REWARD_BALL_COLLECTED
                collected[i] = True

        # ── Reward: all balls collected bonus ─────
        # Large one-time bonus when every ball is gone.
        # Encourages agents to finish the task, not just collect some balls.
        if not self.balls:
            for i in range(C.NUM_AGENTS):
                rewards[i] += C.REWARD_ALL_COLLECTED
            self.done = True

        # ── Reward: anti-clustering ───────────────
        # If two or more agents are within 2 cells of the same nearest ball,
        # penalise each of them. This pushes agents to spread out and target
        # different balls, emerging as natural role specialisation.
        if self.balls and not self.done:
            ball_claimants: dict = {}
            for i in range(C.NUM_AGENTS):
                d, nb = nearest_ball(self.agent_positions[i], self.balls)
                if d <= 2:
                    ball_claimants.setdefault(nb, []).append(i)
            for ball_pos, claimants in ball_claimants.items():
                if len(claimants) > 1:
                    for i in claimants:
                        rewards[i] += C.REWARD_ANTI_CLUSTER

        # ── Done: max steps exceeded ──────────────
        if self.step_count >= C.MAX_STEPS:
            self.done = True

        info = {
            "collisions_step": collisions_this_step,
            "balls_remaining": len(self.balls),
            "collected":       collected,
        }
        return rewards, list(self.agent_positions), self.done, info

    # ─────────────────────────────────────────
    # COLLISION RESOLUTION
    # ─────────────────────────────────────────

    def _resolve_collisions(self, old_positions, proposed_moves):
        """
        Deterministically resolve movement conflicts so that no two agents
        share the same cell after a step.

        Strategy (priority-based):
        1. Clamp moves to grid boundaries.
        2. Block moves into obstacle cells.
        3. Agents are numbered 0..N-1; lower index has priority.
        4. If two agents propose the same target cell, only the lower-index
           agent moves; the other stays in place.
        5. An agent cannot move into a cell that another agent currently
           occupies AND is staying (edge-swap prevention).
        """
        n = len(old_positions)
        resolved = list(proposed_moves)

        # Step 1 & 2: boundary + obstacle clamping
        for i in range(n):
            r, c = resolved[i]
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                resolved[i] = old_positions[i]   # Out of bounds → stay
            elif resolved[i] in self.obstacles:
                resolved[i] = old_positions[i]   # Obstacle → stay

        # Step 3 & 4: resolve target-cell conflicts (lower index wins)
        claimed = {}   # target_cell → first agent that claimed it
        for i in range(n):
            target = resolved[i]
            if target in claimed:
                # Conflict — higher-priority agent already claimed this cell
                resolved[i] = old_positions[i]   # Fallback: stay
            else:
                claimed[target] = i

        # Step 5: prevent edge-swap (A→B and B→A simultaneously)
        for i in range(n):
            for j in range(i + 1, n):
                if resolved[i] == old_positions[j] and resolved[j] == old_positions[i]:
                    # Swap detected — agent with lower priority (j) stays
                    resolved[j] = old_positions[j]

        return resolved

    # ─────────────────────────────────────────
    # PLACEMENT HELPERS
    # ─────────────────────────────────────────

    def _place_obstacles(self) -> set:
        """Place NUM_OBSTACLES randomly, leaving borders clear."""
        obstacles = set()
        attempts  = 0
        while len(obstacles) < C.NUM_OBSTACLES and attempts < 10_000:
            r = random.randint(1, self.rows - 2)
            c = random.randint(1, self.cols - 2)
            obstacles.add((r, c))
            attempts += 1
        return obstacles

    def _place_items(self, count: int, exclude: set) -> set:
        """Place `count` items randomly, not overlapping with `exclude`."""
        items    = set()
        attempts = 0
        while len(items) < count and attempts < 100_000:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            pos = (r, c)
            if pos not in exclude and pos not in items:
                items.add(pos)
            attempts += 1
        return items

    def _random_empty(self, exclude: set):
        """Return a random cell not in `exclude`."""
        while True:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            if (r, c) not in exclude:
                return (r, c)

    # ─────────────────────────────────────────
    # OBSERVATION HELPER
    # ─────────────────────────────────────────

    def get_local_obs(self, agent_pos, radius: int = C.OBS_RADIUS) -> dict:
        """
        Return a dict describing what is visible within `radius` cells of agent.
        Used for building the RL state (see utils.encode_state).
        """
        r, c     = agent_pos
        visible_balls     = []
        visible_obstacles = []
        visible_agents    = []

        for pos in self.balls:
            if manhattan_distance(agent_pos, pos) <= radius:
                visible_balls.append(pos)

        for pos in self.obstacles:
            if manhattan_distance(agent_pos, pos) <= radius:
                visible_obstacles.append(pos)

        for i, pos in enumerate(self.agent_positions):
            if pos != agent_pos and manhattan_distance(agent_pos, pos) <= radius:
                visible_agents.append(pos)

        return {
            "balls":     visible_balls,
            "obstacles": visible_obstacles,
            "agents":    visible_agents,
        }
