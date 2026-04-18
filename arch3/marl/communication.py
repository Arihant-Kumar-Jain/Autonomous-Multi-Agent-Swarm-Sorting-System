"""
marl/communication.py

Differentiable attention-based communication module.

Architecture
------------
Each agent i:
  1. Encodes its local observation → message embedding  m_i ∈ R^{comm_dim}
  2. Queries other agents' messages via Multi-Head Attention
  3. Outputs a communication-enhanced embedding  c_i ∈ R^{comm_dim}

This is fully differentiable and trained end-to-end with the actor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class CommunicationModule(nn.Module):
    """
    Multi-Head Attention communication between N agents.

    Parameters
    ----------
    input_dim : int   – size of each agent's encoded hidden state
    comm_dim  : int   – message / query / key / value dimension
    num_heads : int   – number of attention heads
    n_agents  : int   – number of agents (fixed)
    """

    def __init__(
        self,
        input_dim: int = cfg.HIDDEN_DIM,
        comm_dim:  int = cfg.COMM_DIM,
        num_heads: int = cfg.NUM_HEADS,
        n_agents:  int = cfg.NUM_AGENTS,
    ):
        super().__init__()
        self.comm_dim  = comm_dim
        self.n_agents  = n_agents
        self.num_heads = num_heads

        # Project agent hidden state → message embedding
        self.msg_encoder = nn.Sequential(
            nn.Linear(input_dim, comm_dim),
            nn.LayerNorm(comm_dim),
            nn.ReLU(),
        )

        # Multi-Head Attention: each agent attends to others' messages
        # embed_dim must be divisible by num_heads
        assert comm_dim % num_heads == 0, \
            f"comm_dim ({comm_dim}) must be divisible by num_heads ({num_heads})"
        self.attn = nn.MultiheadAttention(
            embed_dim   = comm_dim,
            num_heads   = num_heads,
            batch_first = True,   # (batch, seq, dim)
            dropout     = 0.0,
        )

        # Layer norm + residual projection
        self.layer_norm = nn.LayerNorm(comm_dim)
        self.out_proj   = nn.Sequential(
            nn.Linear(comm_dim, comm_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,   # (batch * N, hidden_dim)
        batch_size:    int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (B*N, hidden_dim)  – encoded observations for all agents
        batch_size    : B

        Returns
        -------
        comm_out : (B*N, comm_dim) – communication-enhanced embeddings
        """
        N = self.n_agents

        # (B*N, hidden_dim) → (B, N, comm_dim)
        msgs = self.msg_encoder(hidden_states)           # (B*N, comm_dim)
        total = msgs.shape[0]
        N = self.n_agents

        assert total % N == 0, f"Invalid shape: total={total}, N={N}"

        B = total // N

        msgs = msgs.view(B, N, self.comm_dim)

        # Self-attention: each agent (query) attends to all agents (key/value)
        # attn_out shape: (B, N, comm_dim)
        attn_out, _attn_weights = self.attn(msgs, msgs, msgs)

        # Residual connection + layer norm
        comm = self.layer_norm(msgs + attn_out)          # (B, N, comm_dim)
        comm = self.out_proj(comm)                       # (B, N, comm_dim)

        # Flatten back to (B*N, comm_dim)
        return comm.view(B * N, self.comm_dim)