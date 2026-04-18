"""
marl/critic.py

Centralized Critic V(s) – sees the FULL global state.

This solves the non-stationarity problem in cooperative MARL:
from any single agent's perspective, other agents are part of a
non-stationary environment.  By conditioning on the global state
during training, the critic can produce stable value estimates.

At EXECUTION time, the critic is NOT used — only the actor runs.
This is the key insight of CTDE (Centralized Training, Decentralized Execution).
"""

import torch
import torch.nn as nn
from config import cfg


class CentralizedCritic(nn.Module):
    """
    V(s) where s = full global state.

    Input  : (B, global_state_dim)
    Output : (B, 1)  scalar value estimate
    """

    def __init__(
        self,
        global_state_dim: int = cfg.GLOBAL_STATE_SIZE,
        hidden_dim:       int = cfg.HIDDEN_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Last layer: small init for stable early training
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        global_state : (B, global_state_dim)

        Returns
        -------
        value : (B, 1)
        """
        return self.net(global_state)