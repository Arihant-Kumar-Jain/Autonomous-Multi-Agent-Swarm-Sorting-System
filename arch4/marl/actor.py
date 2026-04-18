"""
marl/actor.py - Shared policy network (arch4)

Architecture
------------
  obs (67) -> FC(256) + LayerNorm + Tanh
           -> FC(256) + LayerNorm + Tanh  [+ residual from layer 1]
           -> FC(num_actions)             [small init -> uniform start]

Design decisions
----------------
  • Tanh activations: smoother gradients than ReLU in early training,
    prevents "dead neuron" problem that was slowing arch3
  • Residual connection: helps gradient flow in deeper network
  • Proper orthogonal init: gain=√2 for hidden layers, 0.01 for output
    (arch3 used 0.01 EVERYWHERE - essentially zero-initializing the encoder)
  • No communication module: coordination emerges implicitly from the
    richer observation space (all balls visible, other agents visible)
  • CTDE: this actor is used in BOTH training and execution (decentralized)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import cfg


class SharedActor(nn.Module):
    """
    Shared policy network for all N agents.

    Input  : local observation (B, obs_dim) or (N, obs_dim) at execution
    Output : action logits + entropy
    """

    def __init__(
        self,
        obs_dim:     int = cfg.OBS_SIZE,
        hidden_dim:  int = cfg.HIDDEN_DIM,
        num_actions: int = cfg.NUM_ACTIONS,
    ):
        super().__init__()
        self.obs_dim    = obs_dim
        self.hidden_dim = hidden_dim

        # Layer 1: obs -> hidden
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Layer 2: hidden -> hidden  (+ residual from layer 1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Layer 3: refinement
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

        # Action head: small init -> approximately uniform policy at start
        self.action_head = nn.Linear(hidden_dim // 2, num_actions)

        self._init_weights()

    def _init_weights(self):
        # Hidden layers: orthogonal with gain=√2 (recommended for Tanh)
        gain = (2.0 ** 0.5)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        # Output head: tiny init -> near-uniform logits at start
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to feature vector."""
        # Layer 1
        h1 = torch.tanh(self.ln1(self.fc1(obs)))
        # Layer 2 (with residual from h1)
        h2 = torch.tanh(self.ln2(self.fc2(h1))) + h1
        # Layer 3
        h3 = torch.tanh(self.ln3(self.fc3(h2)))
        return h3

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits  : (B, num_actions)
        entropy : (B,)
        """
        features = self._encode(obs)
        logits   = self.action_head(features)
        dist     = Categorical(logits=logits)
        return logits, dist.entropy()

    @torch.no_grad()
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action for each agent (execution time).

        Parameters
        ----------
        obs : (N, obs_dim)

        Returns
        -------
        actions   : (N,)
        log_probs : (N,)
        entropy   : (N,)
        """
        logits, entropy = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        obs:     torch.Tensor,   # (B, obs_dim)
        actions: torch.Tensor,   # (B,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probs and entropy for PPO update."""
        logits, entropy = self.forward(obs)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return log_probs, entropy
