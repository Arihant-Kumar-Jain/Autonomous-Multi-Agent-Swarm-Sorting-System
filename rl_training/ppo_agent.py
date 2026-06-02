"""
PPO Agent — Proximal Policy Optimization for multi-agent warehouse navigation.

Architecture:
  - Actor-Critic with shared feature extractor
  - GAE (Generalized Advantage Estimation)
  - Clipped surrogate objective
  - Entropy bonus for exploration

Two variants:
  - Basic PPO: 12-dim state
  - Improved PPO: 14-dim state (congestion-aware)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, state_size, action_size, hidden=256):
        super().__init__()

        # Shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_size),
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Smaller init for policy head (more uniform initial distribution)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, state, deterministic=False):
        """Sample action from policy."""
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, value.squeeze(-1), entropy

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update."""
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Storage for PPO rollouts."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute GAE advantages and discounted returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae

        returns = advantages + values
        return returns, advantages

    def get_batches(self, batch_size, returns, advantages):
        """Yield mini-batches for PPO update."""
        states = np.array(self.states)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)

        T = len(states)
        indices = np.random.permutation(T)

        for start in range(0, T, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            yield (
                states[idx],
                actions[idx],
                log_probs[idx],
                returns[idx],
                advantages[idx],
            )

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO agent for warehouse navigation."""

    def __init__(self, state_size, action_size=5, device=None,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
                 max_grad_norm=0.5, ppo_epochs=10, batch_size=128,
                 rollout_length=512):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        # Network
        self.network = ActorCritic(state_size, action_size, hidden=256).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.95)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.total_steps = 0
        self.updates = 0

    def select_action(self, state, training=True):
        """Select action from policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value, entropy = self.network.get_action(
                state_t, deterministic=not training)
            return (action.item(), log_prob.item(), value.item())

    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store transition in rollout buffer."""
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    def should_update(self):
        """Check if we have enough samples for an update."""
        return len(self.buffer) >= self.rollout_length

    def update(self, last_obs):
        """Run PPO update on collected rollout."""
        # Get last value for GAE computation
        with torch.no_grad():
            state_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(state_t)
            last_value = last_value.item()

        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_loss = 0.0
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.batch_size, returns, advantages):
                states_b, actions_b, old_log_probs_b, returns_b, advantages_b = batch

                states_t = torch.FloatTensor(states_b).to(self.device)
                actions_t = torch.LongTensor(actions_b).to(self.device)
                old_log_probs_t = torch.FloatTensor(old_log_probs_b).to(self.device)
                returns_t = torch.FloatTensor(returns_b).to(self.device)
                advantages_t = torch.FloatTensor(advantages_b).to(self.device)

                # Evaluate current policy
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    states_t, actions_t)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = 0.5 * (returns_t - values).pow(2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.value_coef * vf_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        self.scheduler.step()
        self.buffer.clear()
        self.updates += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "pg_loss": total_pg_loss / max(num_batches, 1),
            "vf_loss": total_vf_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def save(self, path):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.updates = checkpoint.get("updates", 0)
