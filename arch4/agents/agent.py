"""
agents/agent.py - Thin agent wrapper (arch4)

The Agent class is a lightweight bookkeeper: it wraps the per-agent index
and tracks per-episode statistics (steps, deliveries, collisions).
All learning is handled centrally by MAPPO.
"""

import numpy as np


class Agent:
    """Bookkeeping wrapper for a single agent."""

    def __init__(self, agent_id: int):
        self.agent_id   = agent_id
        self.pos        = None
        self.carrying   = False
        self.deliveries = 0
        self.collisions = 0
        self.steps      = 0

    def reset_stats(self):
        self.deliveries = 0
        self.collisions = 0
        self.steps      = 0

    def update_from_env(self, pos: np.ndarray, carrying: bool):
        self.pos      = pos.copy()
        self.carrying = carrying
        self.steps   += 1

    def __repr__(self) -> str:
        return (f"Agent({self.agent_id}) "
                f"pos={self.pos} carry={self.carrying} "
                f"deliveries={self.deliveries}")
