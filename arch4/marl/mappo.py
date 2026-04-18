"""
marl/mappo.py - MAPPO trainer (arch4)

Improvements over arch3:
  • LR scheduling: cosine annealing -> naturally decays over training
  • Entropy coefficient annealing: start high (exploration), decay to min
  • No reward normalization in Phase 0: dense shaping must remain legible
  • More PPO epochs (8 vs 4): better sample efficiency per update
  • Obs normalization kept (helps stability), but reward normalization removed
  • Gradient clipping applied per-parameter-group (actor + critic separate)
  • Cleaner update: actor and critic backprop separately

PPO update steps:
  1. Collect rollout (UPDATE_INTERVAL steps, N agents)
  2. Compute GAE advantages (per-agent) + team returns (for critic)
  3. For PPO_EPOCHS epochs, shuffle data into mini-batches:
       a. Evaluate new log-probs and entropy
       b. Clipped policy loss
       c. Value loss (critic MSE)
       d. Entropy bonus (annealed)
       e. Separate backprop for actor and critic
  4. Clear buffer, anneal LR + entropy coeff
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from marl.actor  import SharedActor
from marl.critic import CentralizedCritic
from marl.buffer import RolloutBuffer
from utils.helpers import RunningMeanStd


class MAPPO:
    """MAPPO trainer: owns actor, critic, buffer, and optimizers."""

    def __init__(self, writer=None):
        self.device      = torch.device(cfg.DEVICE)
        self.n_agents    = cfg.NUM_AGENTS
        self.writer      = writer
        self.update_count = 0

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor  = SharedActor().to(self.device)
        self.critic = CentralizedCritic().to(self.device)

        # ── Optimizers ────────────────────────────────────────────────────────
        self.actor_opt  = optim.Adam(self.actor.parameters(),
                                     lr=cfg.ACTOR_LR, eps=1e-5)
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=cfg.CRITIC_LR, eps=1e-5)

        # ── LR Schedulers (cosine annealing over total updates) ───────────────
        # Estimate total updates: total_episodes * avg_steps_per_ep / update_interval
        # We use a generous upper bound; cosine never goes below eta_min
        total_updates = (cfg.TOTAL_EPISODES * cfg.MAX_STEPS) // cfg.UPDATE_INTERVAL
        self.actor_sched  = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt,  T_max=max(total_updates, 1), eta_min=1e-5
        )
        self.critic_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_opt, T_max=max(total_updates, 1), eta_min=3e-5
        )

        # ── Entropy coefficient (annealed from ENTROPY_COEF -> ENTROPY_COEF_MIN) ─
        self.entropy_coef     = cfg.ENTROPY_COEF
        self.entropy_coef_min = cfg.ENTROPY_COEF_MIN

        # ── Rollout buffer ─────────────────────────────────────────────────────
        self.buffer = RolloutBuffer(device=str(self.device))

        # ── Observation normalizer (running mean/std) ─────────────────────────
        # Reward normalization is intentionally removed: it was dividing the
        # dense distance-shaping signal close to zero early in training.
        self.obs_rms = RunningMeanStd(shape=(cfg.OBS_SIZE,))

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_actions(
        self,
        obs: list[np.ndarray],
        deterministic: bool = False,
    ) -> tuple[list[int], np.ndarray]:
        """
        Parameters
        ----------
        obs : list of N arrays (obs_dim,)

        Returns
        -------
        actions   : list[int]  length N
        log_probs : np.ndarray (N,)
        """
        norm_obs = [self.obs_rms.normalize(o) for o in obs]
        obs_t = torch.FloatTensor(np.stack(norm_obs)).to(self.device)

        actions_t, log_probs_t, _ = self.actor.get_action(obs_t, deterministic)
        return actions_t.cpu().numpy().tolist(), log_probs_t.cpu().numpy()

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).squeeze().item()

    # ── Store transition ──────────────────────────────────────────────────────

    def store(self, obs, global_state, actions, log_probs, rewards, value, done):
        # Update obs running stats
        for o in obs:
            self.obs_rms.update(o)
        self.buffer.store(obs, global_state, actions, log_probs, rewards, value, done)

    @property
    def buffer_ready(self) -> bool:
        return self.buffer.is_ready

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(self, last_obs: list[np.ndarray], last_global_state: np.ndarray):
        """
        Full PPO update. Returns dict of mean loss metrics.
        """
        last_value = self.get_value(last_global_state)
        advantages, team_returns = self.buffer.compute_gae(last_value)
        data = self.buffer.get_tensors(advantages, team_returns)

        obs_b  = data["obs"].to(self.device)
        gs_b   = data["global_state"].to(self.device)
        act_b  = data["actions"].to(self.device)
        lp_b   = data["log_probs"].to(self.device)
        adv_b  = data["advantages"].to(self.device)
        ret_b  = data["returns"].to(self.device)

        # Final advantage normalization (belt + suspenders)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # Reshape for mini-batch iteration
        T_N = obs_b.shape[0]
        N   = self.n_agents
        T   = T_N // N

        obs_b = obs_b.view(T, N, -1)
        act_b = act_b.view(T, N)
        lp_b  = lp_b.view(T, N)
        adv_b = adv_b.view(T, N)
        ret_b = ret_b.view(T, N)
        gs_b  = gs_b.view(T, N, -1)

        losses = {"policy": [], "value": [], "entropy": [], "total": []}
        mb_size = max((cfg.MINI_BATCH_SIZE // N) * N, N)

        for _ in range(cfg.PPO_EPOCHS):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, mb_size // N):
                idx = perm[start: start + mb_size // N]
                if len(idx) < 2:
                    continue

                B = len(idx)
                mb_obs = obs_b[idx].reshape(B * N, -1)
                mb_act = act_b[idx].reshape(B * N)
                mb_lp  = lp_b[idx].reshape(B * N)
                mb_adv = adv_b[idx].reshape(B * N)
                mb_ret = ret_b[idx].reshape(B * N)
                # One global state per timestep (take agent-0 slice)
                mb_gs  = gs_b[idx, 0, :]        # (B, gs_dim)

                # ── Actor loss ──────────────────────────────────────────────
                new_lp, entropy = self.actor.evaluate_actions(mb_obs, mb_act)
                ratio      = torch.exp(new_lp - mb_lp)
                clip_ratio = torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS)
                policy_loss = -torch.min(ratio * mb_adv,
                                         clip_ratio * mb_adv).mean()
                entropy_loss = -entropy.mean()
                actor_loss   = policy_loss + self.entropy_coef * entropy_loss

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.MAX_GRAD_NORM)
                self.actor_opt.step()

                # ── Critic loss ─────────────────────────────────────────────
                values_pred = self.critic(mb_gs).squeeze(-1)     # (B,)
                # Use mean of per-agent returns as critic target
                critic_target = mb_ret.view(B, N).mean(dim=1)    # (B,)
                value_loss   = nn.functional.mse_loss(values_pred, critic_target)

                self.critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
                self.critic_opt.step()

                total_loss = actor_loss.item() + cfg.VALUE_COEF * value_loss.item()
                losses["policy"].append(policy_loss.item())
                losses["value"].append(value_loss.item())
                losses["entropy"].append(-entropy_loss.item())
                losses["total"].append(total_loss)

        self.buffer.reset()
        self.update_count += 1

        # ── Anneal LR and entropy coeff ───────────────────────────────────────
        self.actor_sched.step()
        self.critic_sched.step()
        # Linear entropy annealing
        decay = (cfg.ENTROPY_COEF - cfg.ENTROPY_COEF_MIN) / max(
            (cfg.TOTAL_EPISODES * cfg.MAX_STEPS) // cfg.UPDATE_INTERVAL, 1
        )
        self.entropy_coef = max(self.entropy_coef - decay, cfg.ENTROPY_COEF_MIN)

        metrics = {k: float(np.mean(v)) for k, v in losses.items()}

        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"loss/{k}", v, self.update_count)
            self.writer.add_scalar("train/entropy_coef",
                                   self.entropy_coef, self.update_count)
            actor_lr = self.actor_opt.param_groups[0]["lr"]
            self.writer.add_scalar("train/actor_lr", actor_lr, self.update_count)

        return metrics

    # ── Model persistence ─────────────────────────────────────────────────────

    def save(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
            "obs_rms":      self.obs_rms.state_dict(),
            "entropy_coef": self.entropy_coef,
            "update_count": self.update_count,
        }, f"{path_prefix}.pt")
        print(f"[MAPPO] Model saved -> {path_prefix}.pt")

    def load(self, path_prefix: str) -> bool:
        path = f"{path_prefix}.pt"
        if not os.path.exists(path):
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if "entropy_coef" in ckpt:
            self.entropy_coef = ckpt["entropy_coef"]
        if "update_count" in ckpt:
            self.update_count = ckpt["update_count"]
        print(f"[MAPPO] Model loaded <- {path}")
        return True
