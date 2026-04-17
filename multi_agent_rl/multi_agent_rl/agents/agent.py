# agents/agent.py — Agent logic: action selection (RL + rule-based), stuck detection
# Each Agent wraps a QLearner and handles its own position, history, and state encoding.

import random
import sys
import os
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C
from rl.rl_model import QLearner, ACTION_DELTAS, ACTIONS, ACTION_STAY
from utils.utils  import (
    encode_state, nearest_ball, manhattan_distance,
    bfs_next_step, direction_sign
)


class Agent:
    """
    One agent in the multi-agent system.

    Modes:
    • "rl"   — Tabular Q-learning (learning, epsilon-greedy)
    • "rule" — Greedy BFS baseline (always goes to nearest ball)

    The Agent does NOT interact with the Environment directly —
    it receives its position and the global ball set from the simulation loop,
    computes an action, and hands it back.
    """

    def __init__(self, agent_id: int, mode: str = "rl"):
        self.agent_id    = agent_id
        self.mode        = mode  # "rl" or "rule"
        self.color       = C.AGENT_COLORS[agent_id]

        # RL model (owns its own Q-table)
        self.learner     = QLearner(agent_id)

        # Current position — set by environment.reset()
        self.pos: tuple  = (0, 0)

        # History for stuck detection & revisit penalty
        self.pos_history: deque = deque(maxlen=C.STUCK_WINDOW)
        self.recent_pos:  deque = deque(maxlen=C.REVISIT_WINDOW)

        # Previous state/action for the Q-learning update
        self._prev_state  = None
        self._prev_action = None

        # Per-episode bookkeeping
        self.balls_collected = 0
        self.episode_reward  = 0.0

        # Distance to nearest ball last step (for shaping reward)
        self.prev_dist_to_ball: float = float("inf")

    # ─────────────────────────────────────────
    # EPISODE LIFECYCLE
    # ─────────────────────────────────────────

    def reset(self, start_pos: tuple):
        """Called at the start of each episode."""
        self.pos              = start_pos
        self.pos_history.clear()
        self.recent_pos.clear()
        self._prev_state      = None
        self._prev_action     = None
        self.balls_collected  = 0
        self.episode_reward   = 0.0
        self.prev_dist_to_ball = float("inf")

    # ─────────────────────────────────────────
    # ACTION SELECTION
    # ─────────────────────────────────────────

    def select_action(self, balls: set, other_positions: list,
                      obstacles: set) -> tuple:
        """
        Choose an action and return the proposed next cell.

        Returns:
            proposed_pos  : (row, col)
            action_index  : int (for Q-learning update)
        """
        if self._is_stuck():
            # Force random exploration if agent hasn't progressed
            action = random.choice(ACTIONS)
        elif self.mode == "rule":
            action = self._rule_based_action(balls, obstacles)
        else:
            # RL: encode state → epsilon-greedy selection
            other_pos_set = [p for p in other_positions if p != self.pos]
            state  = encode_state(self.pos, balls, other_pos_set)
            action = self.learner.select_action(state)
            self._prev_state  = state
            self._prev_action = action

        # Compute proposed next position
        dr, dc       = ACTION_DELTAS[action]
        proposed_pos = (self.pos[0] + dr, self.pos[1] + dc)
        return proposed_pos, action

    # ─────────────────────────────────────────
    # Q-LEARNING UPDATE
    # ─────────────────────────────────────────

    def learn(self, reward: float, balls: set, other_positions: list, done: bool):
        """
        Called after the environment resolves the step.
        Computes the augmented reward (distance shaping + revisit penalty)
        and triggers the Q-table update.
        """
        if self.mode != "rl" or self._prev_state is None:
            return

        # ── Distance shaping reward ───────────────────────────────────────
        # Reward the agent for moving closer to the nearest ball.
        # This dense signal dramatically speeds up early learning.
        dist, _ = nearest_ball(self.pos, balls)
        dist     = dist if dist != float("inf") else C.GRID_ROWS + C.GRID_COLS

        delta_dist = self.prev_dist_to_ball - dist
        shaped_reward = reward + C.REWARD_DISTANCE_SCALE * delta_dist
        self.prev_dist_to_ball = dist

        # ── Revisit penalty ───────────────────────────────────────────────
        # Discourage agents from oscillating between the same 2-3 cells.
        if list(self.recent_pos).count(self.pos) >= 2:
            shaped_reward += C.REWARD_REVISIT_PENALTY

        # Record position in history
        self.recent_pos.append(self.pos)
        self.pos_history.append(self.pos)

        # ── Q-learning update ─────────────────────────────────────────────
        other_pos_set = [p for p in other_positions if p != self.pos]
        next_state    = encode_state(self.pos, balls, other_pos_set)

        self.learner.update(
            self._prev_state,
            self._prev_action,
            shaped_reward,
            next_state,
            done
        )
        self.episode_reward += shaped_reward

    # ─────────────────────────────────────────
    # RULE-BASED GREEDY BASELINE
    # ─────────────────────────────────────────

    def _rule_based_action(self, balls: set, obstacles: set) -> int:
        """
        BFS toward the nearest ball.
        Provides a deterministic, non-learning baseline for comparison.
        """
        if not balls:
            return ACTION_STAY

        _, target = nearest_ball(self.pos, balls)
        if target is None:
            return ACTION_STAY

        next_pos = bfs_next_step(
            self.pos, target, obstacles,
            C.GRID_ROWS, C.GRID_COLS
        )
        if next_pos is None or next_pos == self.pos:
            return ACTION_STAY

        # Map next_pos → action
        dr = next_pos[0] - self.pos[0]
        dc = next_pos[1] - self.pos[1]
        delta_to_action = {(-1,0): 0, (1,0): 1, (0,-1): 2, (0,1): 3, (0,0): 4}
        return delta_to_action.get((dr, dc), ACTION_STAY)

    # ─────────────────────────────────────────
    # STUCK DETECTION
    # ─────────────────────────────────────────

    def _is_stuck(self) -> bool:
        """
        Returns True if the agent has not left a small radius during the
        last STUCK_WINDOW steps — indicating it is trapped or oscillating.
        Only triggers exploration override; does NOT apply a penalty
        (that comes from the revisit penalty in learn()).
        """
        if len(self.pos_history) < C.STUCK_WINDOW:
            return False
        # If all recorded positions are within 1 cell of the current pos,
        # the agent hasn't made meaningful progress.
        for old_pos in self.pos_history:
            if manhattan_distance(self.pos, old_pos) > 1:
                return False
        return True

    # ─────────────────────────────────────────
    # EPISODE END
    # ─────────────────────────────────────────

    def end_episode(self):
        """Decay epsilon and record episode stats."""
        self.learner.decay_epsilon()

    # ─────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────

    def save(self):
        self.learner.save()

    def load(self):
        self.learner.load()

    # ─────────────────────────────────────────
    # PROPERTIES
    # ─────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return self.learner.epsilon

    @property
    def q_table_size(self) -> int:
        return self.learner.q_table_size()
