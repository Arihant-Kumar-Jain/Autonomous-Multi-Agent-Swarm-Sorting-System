"""
marl/actor.py

Shared Actor (Policy Network) used by ALL agents.

Architecture
------------
  obs → FC encoder → hidden_state
                         ↓
                  CommunicationModule
                         ↓
              hidden + comm_embedding → action head → Categorical dist

CTDE note: During execution, each agent runs this independently with its
own observation. No global state is needed at execution time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import cfg
from marl.communication import CommunicationModule


class SharedActor(nn.Module):
    """
    Single policy network shared across all N agents.

    Input  : local observation  (B*N, obs_dim)
    Output : action logits      (B*N, num_actions)
             (+ Categorical distribution for sampling)
    """

    def __init__(
        self,
        obs_dim:     int = cfg.OBS_SIZE,
        hidden_dim:  int = cfg.HIDDEN_DIM,
        comm_dim:    int = cfg.COMM_DIM,
        num_actions: int = cfg.NUM_ACTIONS,
        n_agents:    int = cfg.NUM_AGENTS,
        use_comm:    bool = True,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.use_comm = use_comm
        self.hidden_dim = hidden_dim

        # ── Observation encoder ──────────────────────────────────────────────
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Communication module ─────────────────────────────────────────────
        if use_comm:
            self.comm_module = CommunicationModule(
                input_dim = hidden_dim,
                comm_dim  = comm_dim,
                num_heads = cfg.NUM_HEADS,
                n_agents  = n_agents,
            )
            fused_dim = hidden_dim + comm_dim
        else:
            self.comm_module = None
            fused_dim = hidden_dim

        # ── Action head ──────────────────────────────────────────────────────
        self.action_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,     # (B*N, obs_dim)  or (N, obs_dim) at execution
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action_logits : (B*N, num_actions)
        dist_entropy  : (B*N,)  – per-sample entropy for regularization
        """
        # Encode observation
        hidden = self.obs_encoder(obs)                    # (B*N, hidden_dim)

        # Communication-enhanced embedding
        if self.use_comm and self.comm_module is not None:
            comm_emb = self.comm_module(hidden, batch_size)  # (B*N, comm_dim)
            fused    = torch.cat([hidden, comm_emb], dim=-1) # (B*N, hidden+comm)
        else:
            fused = hidden                                 # (B*N, hidden_dim)

        logits = self.action_head(fused)                  # (B*N, num_actions)
        dist   = Categorical(logits=logits)
        return logits, dist.entropy()

    def get_action(
        self,
        obs: torch.Tensor,     # (N, obs_dim)   – single step, no batch dim
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or argmax) action for each agent at execution time.

        Returns
        -------
        actions   : (N,)   int64
        log_probs : (N,)   float32
        entropy   : (N,)   float32
        """
        # batch_size=1 for single-step execution
        logits, entropy = self.forward(obs, batch_size=1)  # (N, num_actions)
        dist = Categorical(logits=logits)

        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = dist.sample()                       # stochastic ✓

        log_probs = dist.log_prob(actions)
        return actions, log_probs, entropy

    def evaluate_actions(
        self,
        obs:     torch.Tensor,    # (B*N, obs_dim)
        actions: torch.Tensor,    # (B*N,)
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log-probs and entropy for given (obs, actions) during PPO update.

        Returns
        -------
        log_probs : (B*N,)
        entropy   : (B*N,)
        """
        logits, entropy = self.forward(obs, batch_size)
        dist      = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        return log_probs, entropy