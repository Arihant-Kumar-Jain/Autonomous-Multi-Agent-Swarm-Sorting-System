"""
marl/critic.py - Centralized Critic V(s) (arch4)

Key improvement over arch3:
  • Input is 42 dims (compact global state) instead of 914 dims
    (arch3 included the raw 900-cell obstacle map in global state -
     the critic was drowning in sparse binary noise)
  • Slightly deeper network: 42 -> 256 -> 256 -> 128 -> 1
  • Proper orthogonal init (output head only gets small gain)

CTDE note: this critic is used ONLY during training. At execution time,
only the actor network runs. The critic provides stable value estimates
by seeing the full global state (all agent positions, all ball states).
"""

import torch
import torch.nn as nn

from config import cfg


class CentralizedCritic(nn.Module):
    """
    V(s) where s = compact global state (42 dims).

    Input  : (B, GLOBAL_STATE_SIZE)
    Output : (B, 1)  scalar value estimate
    """

    def __init__(
        self,
        global_state_dim: int = cfg.GLOBAL_STATE_SIZE,
        hidden_dim:       int = cfg.HIDDEN_DIM,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=(2.0 ** 0.5))
                nn.init.zeros_(m.bias)
        # Last layer: small init for stable early value estimates
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        global_state : (B, GLOBAL_STATE_SIZE)

        Returns
        -------
        value : (B, 1)
        """
        return self.net(global_state)
