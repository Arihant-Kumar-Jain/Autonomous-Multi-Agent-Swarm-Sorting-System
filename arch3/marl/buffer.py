"""
marl/buffer.py

Rollout buffer for on-policy PPO.
Stores transitions for ALL agents interleaved: shape (T*N, ...).

NO replay buffer — PPO is purely on-policy.  Data is cleared after
every update.
"""

import numpy as np
import torch
from config import cfg


class RolloutBuffer:
    """
    Collects T steps × N agents of experience, then computes
    GAE advantages for PPO updates.

    Storage layout: index = step * N_agents + agent_id
    """

    def __init__(
        self,
        capacity:         int = cfg.UPDATE_INTERVAL,
        obs_dim:          int = cfg.OBS_SIZE,
        global_state_dim: int = cfg.GLOBAL_STATE_SIZE,
        n_agents:         int = cfg.NUM_AGENTS,
        gamma:            float = cfg.GAMMA,
        gae_lambda:       float = cfg.GAE_LAMBDA,
        device:           str  = cfg.DEVICE,
    ):
        self.capacity         = capacity
        self.obs_dim          = obs_dim
        self.global_state_dim = global_state_dim
        self.n_agents         = n_agents
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.device           = device

        # Total slots = capacity steps × n_agents
        T = capacity
        N = n_agents

        self.obs          = np.zeros((T, N, obs_dim),          dtype=np.float32)
        self.global_state = np.zeros((T, global_state_dim),    dtype=np.float32)
        self.actions      = np.zeros((T, N),                   dtype=np.int64)
        self.log_probs    = np.zeros((T, N),                   dtype=np.float32)
        self.rewards      = np.zeros((T, N),                   dtype=np.float32)
        self.values       = np.zeros((T,),                     dtype=np.float32)
        self.dones        = np.zeros((T,),                     dtype=np.float32)

        self.ptr  = 0   # current write index
        self.full = False

    # ── Writing ──────────────────────────────────────────────────────────────

    def store(
        self,
        obs:          list[np.ndarray],  # list of N arrays (obs_dim,)
        global_state: np.ndarray,        # (global_state_dim,)
        actions:      list[int],         # (N,)
        log_probs:    np.ndarray,        # (N,)
        rewards:      list[float],       # (N,)
        value:        float,             # scalar V(s)
        done:         bool,
    ):
        t = self.ptr
        for i in range(self.n_agents):
            self.obs[t, i]     = obs[i]
            self.actions[t, i] = actions[i]
            self.log_probs[t, i] = log_probs[i]
            self.rewards[t, i] = rewards[i]
        self.global_state[t] = global_state
        self.values[t]       = value
        self.dones[t]        = float(done)

        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    @property
    def is_ready(self) -> bool:
        return self.full

    # ── GAE computation ──────────────────────────────────────────────────────

    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE-λ).

        Returns
        -------
        advantages : (T, N)  normalized per-agent advantages
        returns    : (T,)    GAE targets for critic
        """
        T = self.ptr
        advantages = np.zeros((T, self.n_agents), dtype=np.float32)
        returns    = np.zeros(T,                   dtype=np.float32)

        # Mean reward across agents as the team signal for critic
        mean_rewards = self.rewards[:T].mean(axis=1)  # (T,)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_value * (1 - self.dones[t])
            else:
                next_val = self.values[t + 1] * (1 - self.dones[t])
            delta = mean_rewards[t] + self.gamma * next_val - self.values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            returns[t]    = gae + self.values[t]

        # Broadcast returns to per-agent: use same V(s) but per-agent reward shaping
        # Advantages per agent: r_i + γV(s') - V(s)
        for i in range(self.n_agents):
            agent_gae = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_value * (1 - self.dones[t])
                else:
                    next_val = self.values[t + 1] * (1 - self.dones[t])
                delta = self.rewards[t, i] + self.gamma * next_val - self.values[t]
                agent_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * agent_gae
                advantages[t, i] = agent_gae

        # Normalize advantages
        flat_adv = advantages.flatten()
        advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        return advantages, returns

    def get_tensors(
        self, advantages: np.ndarray, returns: np.ndarray
    ) -> dict[str, torch.Tensor]:
        """
        Convert buffer to flat tensors for PPO mini-batch training.
        Layout: (T*N, ...) — all agents, all timesteps interleaved.
        """
        T = self.ptr
        N = self.n_agents

        # Flatten (T, N) → (T*N,)
        obs_flat       = torch.FloatTensor(self.obs[:T].reshape(T*N, -1))
        actions_flat   = torch.LongTensor(self.actions[:T].reshape(T*N))
        log_probs_flat = torch.FloatTensor(self.log_probs[:T].reshape(T*N))
        adv_flat       = torch.FloatTensor(advantages.reshape(T*N))
        # global state repeated N times for each timestep: (T*N, gs_dim)
        gs_repeated    = np.repeat(self.global_state[:T], N, axis=0)
        gs_flat        = torch.FloatTensor(gs_repeated)
        returns_rep    = np.repeat(returns[:, None], N, axis=1).reshape(T*N)
        ret_flat       = torch.FloatTensor(returns_rep)

        return {
            "obs":          obs_flat,
            "global_state": gs_flat,
            "actions":      actions_flat,
            "log_probs":    log_probs_flat,
            "advantages":   adv_flat,
            "returns":      ret_flat,
        }

    def reset(self):
        self.ptr  = 0
        self.full = False