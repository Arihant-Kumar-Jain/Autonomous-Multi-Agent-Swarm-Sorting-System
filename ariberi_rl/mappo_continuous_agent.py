"""
MAPPO Continuous Agent — Multi-Agent PPO with continuous actions for Gazebo transfer.

Action space: (linear_vel, angular_vel) ∈ [-1, 1]²
- Gaussian policy with tanh squashing
- Same centralized critic as discrete MAPPO
- Designed for direct cmd_vel mapping in ROS2/Gazebo

Grid mapping:
  linear_vel  >  0.3 → move 1 cell forward (in heading direction)
  linear_vel  < -0.3 → move 1 cell backward
  angular_vel >  0.3 → turn right 90°
  angular_vel < -0.3 → turn left 90°
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import config as cfg


# ─── Heading utilities ───────────────────────────────────────────────

# 4 discrete headings on the grid: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
HEADING_ANGLES = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
HEADING_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # (dr, dc) per heading


def heading_to_sincos(heading_idx):
    """Convert heading index to (sin, cos) for observation."""
    angle = HEADING_ANGLES[heading_idx]
    return math.sin(angle), math.cos(angle)


def rotate_to_ego(dx, dy, heading_idx):
    """Rotate world-frame (dx, dy) into ego-centric frame.

    Ego-centric: x = forward (in heading direction), y = left
    This is critical for Gazebo transfer — the robot always thinks
    in terms of 'forward/left/right', not 'north/south/east/west'.
    """
    angle = HEADING_ANGLES[heading_idx]
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # Rotation matrix (world → ego):
    #   ego_forward = dx * cos + dy * sin
    #   ego_left    = -dx * sin + dy * cos
    ego_fwd = dx * cos_a + dy * sin_a
    ego_left = -dx * sin_a + dy * cos_a
    return ego_fwd, ego_left


# ─── Actor (Gaussian policy) ────────────────────────────────────────

class ContinuousActor(nn.Module):
    """Decentralized actor — outputs (mean, log_std) for continuous actions."""

    ACTION_DIM = 2  # (linear_vel, angular_vel)

    def __init__(self, state_size, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, self.ACTION_DIM)
        # Learnable log_std (state-independent)
        self.log_std = nn.Parameter(torch.zeros(self.ACTION_DIM))

        self._init_weights()

    def _init_weights(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small init for mean head — initial policy is near-zero (cautious)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)
        std = torch.clamp(self.log_std, -3, 0.5).exp()
        std = torch.clamp(std, min=0.2)  # hard floor — never collapse below σ=0.2
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            raw_action = mean
        else:
            dist = Normal(mean, std)
            raw_action = dist.rsample()  # reparameterized sample

        # Squash to [-1, 1] via tanh
        action = torch.tanh(raw_action)

        # Log prob with tanh correction
        if not deterministic:
            log_prob = dist.log_prob(raw_action)
            # Correction for tanh squashing: log_prob -= log(1 - tanh²(x) + ε)
            log_prob -= torch.log(1 - action.pow(2) + 1e-4)
            log_prob = log_prob.sum(dim=-1)
        else:
            log_prob = torch.zeros(obs.shape[0], device=obs.device)

        entropy = 0.5 * (1 + torch.log(2 * math.pi * std.pow(2))).mean()
        return action, log_prob, entropy

    def evaluate_actions(self, obs, actions):
        """Evaluate log probability and entropy for given (obs, actions) pairs."""
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        # Invert tanh to get raw_action — clamp at 0.95 to avoid gradient explosion
        # atanh(0.95) ≈ 1.83, gradient ≈ 10x (vs atanh(0.999) → gradient ≈ 500x)
        raw_action = torch.atanh(actions.clamp(-0.95, 0.95))
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-4)
        log_prob = log_prob.sum(dim=-1)

        entropy = 0.5 * (1 + torch.log(2 * math.pi * std.pow(2))).mean()
        return log_prob, entropy


# ─── Critic (same as discrete MAPPO) ────────────────────────────────

class ContinuousCritic(nn.Module):
    """Centralized critic — takes concatenated global state → V(s)."""

    def __init__(self, global_state_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


# ─── Rollout Buffer ─────────────────────────────────────────────────

class ContinuousRolloutBuffer:
    """Rollout buffer for continuous MAPPO — stores float actions."""

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.obs = [[] for _ in range(num_agents)]
        self.actions = [[] for _ in range(num_agents)]      # now 2D floats
        self.log_probs = [[] for _ in range(num_agents)]
        self.rewards = [[] for _ in range(num_agents)]
        self.dones = [[] for _ in range(num_agents)]
        self.global_states = []
        self.global_values = []

    def add(self, agent_id, obs, action, log_prob, reward, done,
            global_state=None, global_value=None):
        self.obs[agent_id].append(obs)
        self.actions[agent_id].append(action)
        self.log_probs[agent_id].append(log_prob)
        self.rewards[agent_id].append(reward)
        self.dones[agent_id].append(done)
        if agent_id == 0 and global_state is not None:
            self.global_states.append(global_state)
            self.global_values.append(global_value)

    def clear(self):
        for i in range(self.num_agents):
            self.obs[i].clear()
            self.actions[i].clear()
            self.log_probs[i].clear()
            self.rewards[i].clear()
            self.dones[i].clear()
        self.global_states.clear()
        self.global_values.clear()

    def compute_returns_and_advantages(self, last_global_value, gamma=0.99,
                                        gae_lambda=0.95):
        T = len(self.global_values)
        values = np.array(self.global_values)
        mean_rewards = np.zeros(T, dtype=np.float32)
        mean_dones = np.zeros(T, dtype=np.float32)
        for t in range(T):
            rews = [self.rewards[a][t] for a in range(self.num_agents)
                    if t < len(self.rewards[a])]
            mean_rewards[t] = np.mean(rews) if rews else 0.0
            dones = [self.dones[a][t] for a in range(self.num_agents)
                     if t < len(self.dones[a])]
            mean_dones[t] = np.max(dones) if dones else 0.0

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_global_value
                next_done = 0.0
            else:
                next_value = values[t + 1]
                next_done = mean_dones[t + 1]
            delta = mean_rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae

        returns = advantages + values
        return returns, advantages

    def __len__(self):
        return len(self.global_states)


# ─── Agent ───────────────────────────────────────────────────────────

class MAPPOContinuousAgent:
    """Multi-Agent PPO with continuous actions and centralized critic."""

    def __init__(self, state_size, num_agents=3, device=None,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, ppo_epochs=10, batch_size=256,
                 rollout_length=1024):
        self.state_size = state_size
        self.num_agents = num_agents
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_end = 0.005  # higher floor for properly-scaled entropy
        self.entropy_anneal_steps = 2_000_000  # anneal over ~2M env steps
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        self.actor = ContinuousActor(state_size, hidden=256).to(self.device)

        global_state_size = state_size * num_agents
        self.critic = ContinuousCritic(global_state_size, hidden=256).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.95)

        self.buffer = ContinuousRolloutBuffer(num_agents)
        self.total_steps = 0
        self.updates = 0

    def select_action(self, obs, training=True):
        """Select continuous action for a single robot.

        Returns: (action_np [2], log_prob float)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, entropy = self.actor.get_action(obs_t, deterministic=not training)
            return action.cpu().numpy()[0], log_prob.item()

    def get_global_value(self, all_obs):
        with torch.no_grad():
            global_state = np.concatenate(all_obs)
            gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            value = self.critic(gs_t)
            return value.item()

    def store_transition(self, agent_id, obs, action, log_prob, reward, done,
                         global_state=None, global_value=None):
        self.buffer.add(agent_id, obs, action, log_prob, reward, done,
                        global_state, global_value)
        if agent_id == 0:
            self.total_steps += 1

    def should_update(self):
        return len(self.buffer) >= self.rollout_length

    def update(self, last_all_obs):
        """Run MAPPO update with continuous actions."""
        last_global_value = self.get_global_value(last_all_obs)

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_global_value, self.gamma, self.gae_lambda)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(returns)

        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for t in range(T):
            for a in range(self.num_agents):
                if t < len(self.buffer.obs[a]):
                    all_obs.append(self.buffer.obs[a][t])
                    all_actions.append(self.buffer.actions[a][t])
                    all_old_log_probs.append(self.buffer.log_probs[a][t])
                    all_advantages.append(advantages[t])

        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions, dtype=np.float32)  # [N, 2]
        all_old_log_probs = np.array(all_old_log_probs)
        all_advantages_arr = np.array(all_advantages)

        global_states = np.array(self.buffer.global_states)
        returns_arr = returns

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.ppo_epochs):
            N = len(all_obs)
            indices = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                obs_t = torch.FloatTensor(all_obs[idx]).to(self.device)
                act_t = torch.FloatTensor(all_actions[idx]).to(self.device)
                old_lp_t = torch.FloatTensor(all_old_log_probs[idx]).to(self.device)
                adv_t = torch.FloatTensor(all_advantages_arr[idx]).to(self.device)

                new_log_probs, entropy = self.actor.evaluate_actions(obs_t, act_t)

                ratio = torch.exp(new_log_probs - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                pg_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy

                # Entropy annealing: high early → low late
                frac = min(1.0, self.total_steps / self.entropy_anneal_steps)
                entropy_coef = self.entropy_coef_start + frac * (self.entropy_coef_end - self.entropy_coef_start)

                # Log_std regularization to prevent σ runaway
                log_std_penalty = 0.001 * self.actor.log_std.pow(2).sum()

                actor_loss = pg_loss + entropy_coef * entropy_loss + log_std_penalty

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                total_pg_loss += pg_loss.item()
                total_entropy += entropy.item()
                num_batches += 1

            # Critic update
            M = len(global_states)
            c_indices = np.random.permutation(M)
            for start in range(0, M, self.batch_size):
                end = start + self.batch_size
                idx = c_indices[start:end]

                gs_t = torch.FloatTensor(global_states[idx]).to(self.device)
                ret_t = torch.FloatTensor(returns_arr[idx]).to(self.device)

                values = self.critic(gs_t)
                vf_loss = 0.5 * (ret_t - values).pow(2).mean()

                self.critic_optimizer.zero_grad()
                vf_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_vf_loss += vf_loss.item()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.buffer.clear()
        self.updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_batches, 1),
            "vf_loss": total_vf_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "lr_actor": self.actor_optimizer.param_groups[0]["lr"],
            "lr_critic": self.critic_optimizer.param_groups[0]["lr"],
            "log_std": self.actor.log_std.data.cpu().numpy().tolist(),
        }

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.updates = checkpoint.get("updates", 0)
