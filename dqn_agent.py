"""
DQN Agent for multi-agent warehouse navigation.

Two variants:
  - Basic DQN: standard state (12 dims)
  - Improved DQN: congestion-aware state (14 dims)
"""

import random
import math
import numpy as np
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import config as cfg

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayMemory:
    """Experience replay buffer."""

    def __init__(self, capacity=cfg.MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNetwork(nn.Module):
    """Deep Q-Network with dueling architecture."""

    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_size),
        )

    def forward(self, x):
        feat = self.feature(x)
        val = self.value(feat)
        adv = self.advantage(feat)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNAgent:
    """DQN agent for a single robot."""

    def __init__(self, state_size, action_size=cfg.NUM_ACTIONS, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LR)
        self.memory = ReplayMemory()

        self.steps_done = 0

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        eps_threshold = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
                        math.exp(-1.0 * self.steps_done / cfg.EPS_DECAY)

        if training:
            self.steps_done += 1

        if training and random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """One gradient step."""
        if len(self.memory) < cfg.BATCH_SIZE:
            return 0.0

        transitions = self.memory.sample(cfg.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + cfg.GAMMA * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Copy policy net weights to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]

    def get_epsilon(self):
        """Get current epsilon."""
        return cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
               math.exp(-1.0 * self.steps_done / cfg.EPS_DECAY)
