"""
agents/agent.py

Thin Agent wrapper.

At execution time, each agent:
  1. Receives its local observation from the environment
  2. Passes it through the shared actor (+ comm module)
  3. Samples an action

The agent itself holds NO weights — it delegates everything to
the shared MAPPO trainer.  This separation keeps execution logic
clean and extensible (e.g., for heterogeneous agents in future).
"""

import numpy as np
from config import cfg


class Agent:
    """
    Lightweight agent interface.  Does NOT own any neural network weights.
    All policy computation is delegated to MAPPO.
    """

    def __init__(self, agent_id: int):
        self.id          = agent_id
        self.pos         = np.zeros(2, dtype=int)
        self.carrying    = False

        # Statistic counters (per episode)
        self.pickups     = 0
        self.deliveries  = 0
        self.collisions  = 0
        self.steps_taken = 0

    def reset_stats(self):
        self.pickups    = 0
        self.deliveries = 0
        self.collisions = 0
        self.steps_taken = 0

    def update_from_env(self, pos: np.ndarray, carrying: bool):
        """Sync agent state from environment after step."""
        self.pos      = pos.copy()
        self.carrying = carrying
        self.steps_taken += 1

    def __repr__(self):
        return (f"Agent(id={self.id}, pos={self.pos.tolist()}, "
                f"carrying={self.carrying})")