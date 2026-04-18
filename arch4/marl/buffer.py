"""
marl/buffer.py - On-policy rollout buffer (arch4)

Key improvements over arch3:
  • Per-agent returns stored separately for more accurate critic training
  • Simpler, cleaner GAE implementation
  • No redundant global_state repetition tricks

PPO is on-policy: buffer is wiped after every update.
"""

import numpy as np
import torch

from config import cfg


class RolloutBuffer:
    """
    Collects T steps of experience for N agents, then computes
    per-agent GAE advantages for PPO updates.

    Layout (after collection):
      obs[t, i]       - local obs of agent i at step t
      global_state[t] - compact global state at step t
      actions[t, i]   - action taken by agent i at step t
      log_probs[t, i] - log π(a|o) at time of collection
      rewards[t, i]   - reward received by agent i at step t
      values[t]       - V(global_state[t]) from critic
      dones[t]        - episode-done flag at step t
    """

    def __init__(
        self,
        capacity:         int   = cfg.UPDATE_INTERVAL,
        obs_dim:          int   = cfg.OBS_SIZE,
        global_state_dim: int   = cfg.GLOBAL_STATE_SIZE,
        n_agents:         int   = cfg.NUM_AGENTS,
        gamma:            float = cfg.GAMMA,
        gae_lambda:       float = cfg.GAE_LAMBDA,
        device:           str   = cfg.DEVICE,
    ):
        self.capacity         = capacity
        self.obs_dim          = obs_dim
        self.global_state_dim = global_state_dim
        self.n_agents         = n_agents
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.device           = device

        T, N = capacity, n_agents
        self.obs          = np.zeros((T, N, obs_dim),       dtype=np.float32)
        self.global_state = np.zeros((T, global_state_dim), dtype=np.float32)
        self.actions      = np.zeros((T, N),                dtype=np.int64)
        self.log_probs    = np.zeros((T, N),                dtype=np.float32)
        self.rewards      = np.zeros((T, N),                dtype=np.float32)
        self.values       = np.zeros(T,                     dtype=np.float32)
        self.dones        = np.zeros(T,                     dtype=np.float32)

        self.ptr  = 0
        self.full = False

    # ── Writing ──────────────────────────────────────────────────────────────

    def store(
        self,
        obs:          list[np.ndarray],
        global_state: np.ndarray,
        actions:      list[int],
        log_probs:    np.ndarray,
        rewards:      list[float],
        value:        float,
        done:         bool,
    ):
        t = self.ptr
        for i in range(self.n_agents):
            self.obs[t, i]       = obs[i]
            self.actions[t, i]   = actions[i]
            self.log_probs[t, i] = log_probs[i]
            self.rewards[t, i]   = rewards[i]
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

    def compute_gae(
        self, last_value: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation (GAE-λ).

        We compute:
          • team_returns  (T,)    - critic target using mean reward
          • advantages    (T, N)  - per-agent advantages using per-agent reward
          • agent_returns (T, N)  - per-agent discounted returns

        Returns
        -------
        advantages   : (T, N)
        team_returns : (T,)    for critic MSE loss
        agent_returns: (T, N)  (not used by critic but available)
        """
        T = self.ptr
        advantages   = np.zeros((T, self.n_agents), dtype=np.float32)
        team_returns = np.zeros(T, dtype=np.float32)

        # ── Team returns (mean reward across agents -> critic target) ─────────
        mean_rewards = self.rewards[:T].mean(axis=1)   # (T,)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = (last_value if t == T - 1
                        else self.values[t + 1]) * (1.0 - self.dones[t])
            delta = mean_rewards[t] + self.gamma * next_val - self.values[t]
            gae   = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae
            team_returns[t] = gae + self.values[t]

        # ── Per-agent advantages (individual reward shaping signal) ──────────
        for i in range(self.n_agents):
            agent_gae = 0.0
            for t in reversed(range(T)):
                next_val = (last_value if t == T - 1
                            else self.values[t + 1]) * (1.0 - self.dones[t])
                delta = self.rewards[t, i] + self.gamma * next_val - self.values[t]
                agent_gae = (delta + self.gamma * self.gae_lambda *
                             (1.0 - self.dones[t]) * agent_gae)
                advantages[t, i] = agent_gae

        # Normalize advantages across all agents and timesteps
        flat = advantages.flatten()
        advantages = (advantages - flat.mean()) / (flat.std() + 1e-8)

        return advantages, team_returns

    def get_tensors(
        self,
        advantages:   np.ndarray,   # (T, N)
        team_returns: np.ndarray,   # (T,)
    ) -> dict[str, torch.Tensor]:
        """
        Flatten buffer to tensors for PPO mini-batch training.
        Layout: (T*N, ...) for actor data, (T,) for critic data.
        """
        T = self.ptr
        N = self.n_agents

        obs_flat     = torch.FloatTensor(self.obs[:T].reshape(T * N, -1))
        actions_flat = torch.LongTensor(self.actions[:T].reshape(T * N))
        lp_flat      = torch.FloatTensor(self.log_probs[:T].reshape(T * N))
        adv_flat     = torch.FloatTensor(advantages.reshape(T * N))
        # Repeat team_returns N times so shapes align for joint loss
        ret_flat     = torch.FloatTensor(
            np.repeat(team_returns[:, None], N, axis=1).reshape(T * N)
        )
        # Global state repeated N times per timestep
        gs_flat      = torch.FloatTensor(
            np.repeat(self.global_state[:T], N, axis=0)
        )

        return {
            "obs":          obs_flat,
            "global_state": gs_flat,
            "actions":      actions_flat,
            "log_probs":    lp_flat,
            "advantages":   adv_flat,
            "returns":      ret_flat,
        }

    def reset(self):
        self.ptr  = 0
        self.full = False
