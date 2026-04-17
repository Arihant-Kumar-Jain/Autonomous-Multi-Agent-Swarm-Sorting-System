# rl/rl_model.py — Tabular Q-Learning implementation
# Each agent owns one QLearner instance whose Q-table persists across episodes.

import random
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C

# Action constants — used everywhere in the project
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_STAY  = 4
NUM_ACTIONS  = 5

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY]
ACTION_DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
    ACTION_STAY:  ( 0,  0),
}


class QLearner:
    """
    Tabular Q-Learning agent.

    Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]

    State → hashable tuple (see utils.encode_state)
    Action → integer 0..4

    Key design choices:
    • Q-table is a plain dict: state → [q0, q1, q2, q3, q4]
      Avoids pre-allocating a huge array for an unknown state space.
    • Epsilon-greedy with exponential decay across episodes.
    • Q-table persists across episodes (no reset) so agents improve over time.
    """

    def __init__(self, agent_id: int, epsilon: float = C.EPSILON_START):
        self.agent_id = agent_id
        self.epsilon  = epsilon

        # Q-table: dict[state_tuple → list of Q-values per action]
        self.q_table: dict = {}

        # Training statistics
        self.total_updates = 0
        self.episode_rewards = []

    # ──────────────────────────────────────────
    # Q-VALUE ACCESS
    # ──────────────────────────────────────────

    def _get_q(self, state) -> list:
        """Return Q-values for a state, initialising to zero if unseen."""
        if state not in self.q_table:
            # Initialise with small random values to break ties symmetrically
            self.q_table[state] = [random.uniform(-0.01, 0.01) for _ in range(NUM_ACTIONS)]
        return self.q_table[state]

    def best_action(self, state) -> int:
        """Greedy action: argmax_a Q(s, a)."""
        q_vals = self._get_q(state)
        max_q  = max(q_vals)
        # Break ties randomly among actions sharing the maximum value
        best   = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(best)

    def select_action(self, state) -> int:
        """
        Epsilon-greedy action selection.
        With probability ε → random exploration.
        Otherwise          → greedy exploitation.
        """
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return self.best_action(state)

    # ──────────────────────────────────────────
    # LEARNING UPDATE
    # ──────────────────────────────────────────

    def update(self, state, action: int, reward: float, next_state, done: bool):
        """
        Standard Q-learning (off-policy TD update).

        done=True means the episode ended after this transition,
        so there is no future reward (next state value = 0).
        """
        q_vals     = self._get_q(state)
        next_q_max = 0.0 if done else max(self._get_q(next_state))

        # Bellman target
        td_target  = reward + C.GAMMA * next_q_max
        td_error   = td_target - q_vals[action]

        # In-place update
        q_vals[action] += C.ALPHA * td_error
        self.total_updates += 1

    # ──────────────────────────────────────────
    # EPSILON DECAY
    # ──────────────────────────────────────────

    def decay_epsilon(self):
        """
        Multiply epsilon by decay factor at the end of each episode.
        Clamps to EPSILON_MIN so agents never stop exploring completely.
        """
        self.epsilon = max(C.EPSILON_MIN, self.epsilon * C.EPSILON_DECAY)

    # ──────────────────────────────────────────
    # PERSISTENCE — SAVE / LOAD Q-TABLE
    # ──────────────────────────────────────────

    def save(self, directory: str = C.MODEL_DIR):
        """Pickle the Q-table to disk."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"agent_{self.agent_id}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "total_updates": self.total_updates,
            }, f)

    def load(self, directory: str = C.MODEL_DIR):
        """Load Q-table from disk if it exists (safe no-op otherwise)."""
        path = os.path.join(directory, f"agent_{self.agent_id}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.q_table      = data.get("q_table", {})
            self.epsilon      = data.get("epsilon", self.epsilon)
            self.total_updates = data.get("total_updates", 0)
            print(f"[QLearner {self.agent_id}] Loaded Q-table "
                  f"({len(self.q_table)} states, ε={self.epsilon:.3f})")
        else:
            print(f"[QLearner {self.agent_id}] No saved model found — starting fresh.")

    def q_table_size(self) -> int:
        """Number of unique states seen so far."""
        return len(self.q_table)
