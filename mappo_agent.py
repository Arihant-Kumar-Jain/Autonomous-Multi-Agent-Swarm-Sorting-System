"""
MAPPO Agent — Multi-Agent PPO with Centralized Training, Decentralized Execution.

Architecture:
  - Shared Actor: per-robot policy (takes single-agent obs → action)
  - Centralized Critic: takes concatenated obs of ALL robots → V(s_global)
  - GAE computed using global value estimates
  - Clipped surrogate objective (same as PPO)

This is the CTDE paradigm:
  - Training: critic sees everything (centralized)
  - Execution: each robot uses only its own actor (decentralized)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import config as cfg


class MAPPOActor(nn.Module):
    """Decentralized actor — per-robot policy network."""

    def __init__(self, state_size, action_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small init for policy output
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x):
        return self.net(x)

    def get_action(self, obs, deterministic=False):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, obs, actions):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class MAPPOCritic(nn.Module):
    """Centralized critic — takes concatenated global state."""

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


class MAPPORolloutBuffer:
    """Rollout buffer for MAPPO — stores per-agent transitions with global state."""

    def __init__(self, num_agents):
        self.num_agents = num_agents
        # Per-agent data
        self.obs = [[] for _ in range(num_agents)]
        self.actions = [[] for _ in range(num_agents)]
        self.log_probs = [[] for _ in range(num_agents)]
        self.rewards = [[] for _ in range(num_agents)]
        self.dones = [[] for _ in range(num_agents)]
        # Global data (shared across agents per timestep)
        self.global_states = []
        self.global_values = []

    def add(self, agent_id, obs, action, log_prob, reward, done,
            global_state=None, global_value=None):
        self.obs[agent_id].append(obs)
        self.actions[agent_id].append(action)
        self.log_probs[agent_id].append(log_prob)
        self.rewards[agent_id].append(reward)
        self.dones[agent_id].append(done)
        # Only store global state once per timestep (from agent 0)
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
        """Compute GAE per agent using centralized value estimates."""
        T = len(self.global_values)
        values = np.array(self.global_values)
        # Use mean reward across agents for centralized value target
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


class MAPPOAgent:
    """Multi-Agent PPO with centralized critic."""

    def __init__(self, state_size, action_size=5, num_agents=3, device=None,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, ppo_epochs=10, batch_size=256,
                 rollout_length=1024):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        # Shared actor (all robots use the same policy)
        self.actor = MAPPOActor(state_size, action_size, hidden=256).to(self.device)

        # Centralized critic (sees all agents' observations concatenated)
        global_state_size = state_size * num_agents
        self.critic = MAPPOCritic(global_state_size, hidden=256).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.95)

        self.buffer = MAPPORolloutBuffer(num_agents)
        self.total_steps = 0
        self.updates = 0

    def select_action(self, obs, training=True):
        """Select action for a single robot."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, entropy = self.actor.get_action(obs_t, deterministic=not training)
            return action.item(), log_prob.item()

    def get_global_value(self, all_obs):
        """Get centralized value from concatenated observations."""
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
        """Run MAPPO update."""
        # Get last global value
        last_global_value = self.get_global_value(last_all_obs)

        # Compute returns and advantages (centralized)
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_global_value, self.gamma, self.gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        T = len(returns)

        # Flatten per-agent data for actor update
        all_obs = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []  # broadcast same advantage to all agents

        for t in range(T):
            for a in range(self.num_agents):
                if t < len(self.buffer.obs[a]):
                    all_obs.append(self.buffer.obs[a][t])
                    all_actions.append(self.buffer.actions[a][t])
                    all_old_log_probs.append(self.buffer.log_probs[a][t])
                    all_advantages.append(advantages[t])

        all_obs = np.array(all_obs)
        all_actions = np.array(all_actions)
        all_old_log_probs = np.array(all_old_log_probs)
        all_advantages_arr = np.array(all_advantages)

        # Global states and returns for critic
        global_states = np.array(self.buffer.global_states)
        returns_arr = returns

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.ppo_epochs):
            # Actor update (mini-batches over all agent transitions)
            N = len(all_obs)
            indices = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                obs_t = torch.FloatTensor(all_obs[idx]).to(self.device)
                act_t = torch.LongTensor(all_actions[idx]).to(self.device)
                old_lp_t = torch.FloatTensor(all_old_log_probs[idx]).to(self.device)
                adv_t = torch.FloatTensor(all_advantages_arr[idx]).to(self.device)

                new_log_probs, entropy = self.actor.evaluate_actions(obs_t, act_t)

                ratio = torch.exp(new_log_probs - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t
                pg_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                actor_loss = pg_loss + self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                total_pg_loss += pg_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

            # Critic update (mini-batches over timesteps)
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
