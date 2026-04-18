"""
marl/mappo.py

MAPPO: Multi-Agent Proximal Policy Optimization

Implements CTDE:
  • Centralized Training  → critic sees global state
  • Decentralized Execution → actor sees only local obs

PPO update steps:
  1. Collect rollout (T steps, N agents)
  2. Compute GAE advantages
  3. For K epochs, shuffle data into mini-batches:
       a. Evaluate new log-probs and entropy
       b. Compute clipped policy loss
       c. Compute value loss
       d. Compute entropy bonus
       e. Backprop combined loss
  4. Clear buffer

Constraints:
  ✓ Clipped PPO objective
  ✓ GAE
  ✓ Entropy regularization
  ✓ No replay buffer
  ✓ No ε-greedy
  ✓ Shared actor
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from marl.actor  import SharedActor
from marl.critic import CentralizedCritic
from marl.buffer import RolloutBuffer
from utils.helpers import RunningMeanStd


class MAPPO:
    """
    MAPPO trainer: owns actor, critic, buffer, and optimizers.
    """

    def __init__(self, use_comm: bool = True, writer: SummaryWriter | None = None):
        self.device   = torch.device(cfg.DEVICE)
        self.n_agents = cfg.NUM_AGENTS
        self.writer   = writer
        self.update_count = 0

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor  = SharedActor(use_comm=use_comm).to(self.device)
        self.critic = CentralizedCritic().to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=cfg.ACTOR_LR,  eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR, eps=1e-5)

        # ── Rollout buffer ─────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(device=str(self.device))

        # ── Observation & reward normalizers ──────────────────────────────────
        self.obs_rms    = RunningMeanStd(shape=(cfg.OBS_SIZE,))
        self.reward_rms = RunningMeanStd(shape=())

    # ── Action selection (used during rollout collection) ─────────────────────

    @torch.no_grad()
    def select_actions(
        self,
        obs:          list[np.ndarray],
        deterministic: bool = False,
    ) -> tuple[list[int], np.ndarray, float]:
        """
        Parameters
        ----------
        obs : list of N arrays (obs_dim,)

        Returns
        -------
        actions   : list[int]   length N
        log_probs : np.ndarray  (N,)
        value     : float       V(global_state)  – needs global_state separately
        """
        # Normalize obs
        norm_obs = [self.obs_rms.normalize(o) for o in obs]
        obs_t = torch.FloatTensor(np.stack(norm_obs)).to(self.device)  # (N, obs_dim)

        actions_t, log_probs_t, _ = self.actor.get_action(obs_t, deterministic)

        actions   = actions_t.cpu().numpy().tolist()
        log_probs = log_probs_t.cpu().numpy()
        return actions, log_probs

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).squeeze().item()

    # ── Store transition ───────────────────────────────────────────────────────

    def store(self, obs, global_state, actions, log_probs, rewards, value, done):
        # Update running stats
        for o in obs:
            self.obs_rms.update(o)
        for r in rewards:
            self.reward_rms.update(np.array(r))

        # Normalize rewards before storing
        norm_rewards = [self.reward_rms.normalize_reward(r) for r in rewards]
        self.buffer.store(obs, global_state, actions, log_probs, norm_rewards, value, done)

    @property
    def buffer_ready(self) -> bool:
        return self.buffer.is_ready

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(self, last_obs: list[np.ndarray], last_global_state: np.ndarray):
        """
        Full PPO update using the current rollout buffer.

        Returns dict of loss metrics for logging.
        """
        # Bootstrap value for last state
        last_value = self.get_value(last_global_state)

        # Compute GAE advantages and returns
        advantages, returns = self.buffer.compute_gae(last_value)

        # Convert to tensors (flattened T*N)
        data = self.buffer.get_tensors(advantages, returns)
        obs_b   = data["obs"].to(self.device)
        gs_b    = data["global_state"].to(self.device)
        act_b   = data["actions"].to(self.device)
        lp_b    = data["log_probs"].to(self.device)
        adv_b   = data["advantages"].to(self.device)
        ret_b   = data["returns"].to(self.device)

        # Normalize advantages (already done in buffer, belt+suspenders)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ── Mini-batch PPO epochs ──────────────────────────────────────────────
        T_N = obs_b.shape[0]
        N = self.n_agents
        T = T_N // N

        obs_b = obs_b.view(T, N, -1)
        act_b = act_b.view(T, N)
        lp_b  = lp_b.view(T, N)
        adv_b = adv_b.view(T, N)
        ret_b = ret_b.view(T, N)
        gs_b  = gs_b.view(T, N, -1)
        losses = {"policy": [], "value": [], "entropy": [], "total": []}

        for _ in range(cfg.PPO_EPOCHS):
            # Shuffle indices
            perm = torch.randperm(T, device=self.device)
            batch_size = cfg.MINI_BATCH_SIZE
            batch_size = (batch_size // self.n_agents) * self.n_agents
            batch_size = max(batch_size, self.n_agents)
            for start in range(0, T, batch_size):
                idx = perm[start : start + batch_size]
                if len(idx) < 2:
                    continue

                mb_obs = obs_b[idx]
                mb_gs  = gs_b[idx]
                mb_act = act_b[idx]
                mb_lp  = lp_b[idx]
                mb_adv = adv_b[idx]
                mb_ret = ret_b[idx]
                B = mb_obs.shape[0]
                N = self.n_agents

                # flatten for actor + critic
                mb_obs = mb_obs.view(B * N, -1)
                mb_act = mb_act.view(B * N)
                mb_lp  = mb_lp.view(B * N)
                mb_adv = mb_adv.view(B * N)
                mb_ret = mb_ret.view(B * N)
                mb_gs  = mb_gs.view(B * N, -1)

                bs = B   # batch size for communication
                # ── Critic loss (MSE) ───────────────────────────────────────
                # Use unique global states (one per timestep, repeated N times)
                # For simplicity: use all rows (some redundancy, acceptable)
                values_pred = self.critic(mb_gs).squeeze(-1)     # (mb,)
                value_loss  = nn.functional.mse_loss(values_pred, mb_ret)
 
                new_lp, entropy = self.actor.evaluate_actions(mb_obs, mb_act, batch_size=bs)

                ratio       = torch.exp(new_lp - mb_lp)
                clip_ratio  = torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS)
                policy_loss = -torch.min(ratio * mb_adv, clip_ratio * mb_adv).mean()

                entropy_loss = -entropy.mean()   # negative because we MAXIMIZE entropy

                # ── Combined loss ───────────────────────────────────────────
                total_loss = (policy_loss
                              + cfg.VALUE_COEF * value_loss
                              + cfg.ENTROPY_COEF * entropy_loss)

                # ── Backprop ────────────────────────────────────────────────
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  cfg.MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
                self.actor_opt.step()
                self.critic_opt.step()

                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(-entropy_loss.item())
                losses["total"].append(total_loss.item())

        self.buffer.reset()
        self.update_count += 1

        metrics = {k: float(np.mean(v)) for k, v in losses.items()}

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"loss/{k}", v, self.update_count)

        return metrics

    # ── Model persistence ─────────────────────────────────────────────────────

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "obs_rms":    self.obs_rms.state_dict(),
            "reward_rms": self.reward_rms.state_dict(),
        }, f"{path_prefix}.pt")
        print(f"[MAPPO] Model saved → {path_prefix}.pt")

    def load(self, path_prefix: str) -> bool:
        path = f"{path_prefix}.pt"
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device,weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if "reward_rms" in ckpt:
            self.reward_rms.load_state_dict(ckpt["reward_rms"])
        print(f"[MAPPO] Model loaded ← {path}")
        return True