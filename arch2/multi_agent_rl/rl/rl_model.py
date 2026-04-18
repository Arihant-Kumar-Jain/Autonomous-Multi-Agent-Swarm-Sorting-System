# rl/rl_model.py — Neural Policy Network with Multi-Agent Communication
# Each agent has a neural network policy that outputs action probabilities

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import sys
import random
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C

# Action constants
ACTION_UP    = 0
ACTION_DOWN  = 1
ACTION_LEFT  = 2
ACTION_RIGHT = 3
ACTION_STAY  = 4
NUM_ACTIONS  = 5

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY]
ACTION_DELTAS = {
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  ( 1,  0),
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0,  1),
    ACTION_STAY:  ( 0,  0),
}


class CommunicationModule(nn.Module):
    """Multi-agent communication module using attention mechanism"""
    
    def __init__(self, state_dim, comm_dim=64, num_heads=4):
        super().__init__()
        self.comm_dim = comm_dim
        self.num_heads = num_heads
        
        # Message encoder
        self.message_encoder = nn.Linear(state_dim, comm_dim)
        
        # Multi-head attention for processing other agents' messages
        self.attention = nn.MultiheadAttention(comm_dim, num_heads, batch_first=True)
        
        # Message integrator
        self.message_integrator = nn.Linear(state_dim + comm_dim, state_dim)
        
    def forward(self, state, other_states=None):
        """
        Args:
            state: agent's own state [batch, state_dim]
            other_states: states of other agents [batch, num_others, state_dim]
        
        Returns:
            integrated_state: state with communication info [batch, state_dim]
            messages: messages sent by this agent [batch, 1, comm_dim]
        """
        batch_size = state.shape[0]
        
        # Encode own state to message
        own_message = self.message_encoder(state).unsqueeze(1)  # [batch, 1, comm_dim]
        
        if other_states is not None and other_states.shape[1] > 0:
            # Encode other agents' states to messages
            other_messages = self.message_encoder(other_states)  # [batch, num_others, comm_dim]
            
            # Attention over other agents' messages
            attended, attention_weights = self.attention(
                own_message, other_messages, other_messages
            )
            
            # Integrate attended information with own state
            integrated_state = self.message_integrator(
                torch.cat([state, attended.squeeze(1)], dim=-1)
            )
        else:
            integrated_state = state
            
        return integrated_state, own_message


class PolicyNetwork(nn.Module):
    """Neural network policy that outputs action probabilities"""
    
    def __init__(self, input_dim, hidden_dims=[128, 128], output_dim=NUM_ACTIONS):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """Critic network for advantage estimation"""
    
    def __init__(self, input_dim, hidden_dims=[128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for storing trajectories"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, trajectory):
        self.buffer.append(trajectory)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class MultiAgentPPO:
    """
    Proximal Policy Optimization with Multi-Agent Communication
    Each agent has its own policy and value network, but they share a communication module
    """
    
    def __init__(self, agent_id: int, state_dim: int, comm_dim: int = 64):
        self.agent_id = agent_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Communication module (shared across agents for consistency)
        self.comm_module = CommunicationModule(state_dim, comm_dim).to(self.device)
        
        # Policy and value networks
        self.policy = PolicyNetwork(state_dim + comm_dim).to(self.device)
        self.value = ValueNetwork(state_dim + comm_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=C.LEARNING_RATE)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=C.LEARNING_RATE)
        
        # Training hyperparameters
        self.clip_epsilon = 0.2
        self.epochs = 4
        self.batch_size = 64
        self.gamma = C.GAMMA
        self.gae_lambda = 0.95
        
        # Memory for current episode
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.other_states_history = []
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Training stats
        self.total_updates = 0
        self.episode_rewards = []
        
        # Exploration noise (for early training)
        self.exploration_noise = C.EPSILON_START
        
        # Load saved model if exists
        self.load()
    
    def _encode_state(self, pos, balls, others_pos, obstacles):
        """
        Encode the environment state into a feature vector.
        
        Args:
            pos: (row, col) position
            balls: list of ball positions
            others_pos: list of other agent positions
            obstacles: set of obstacle positions
        
        Returns:
            state_vector: numpy array of features
        """
        r, c = pos
        
        # 1. One-hot position encoding
        pos_encoding = np.zeros(C.GRID_ROWS * C.GRID_COLS)
        pos_encoding[r * C.GRID_COLS + c] = 1
        
        # 2. Distance to nearest ball
        if balls:
            distances = [abs(r - br) + abs(c - bc) for (br, bc) in balls]
            min_dist = min(distances)
        else:
            min_dist = C.GRID_ROWS + C.GRID_COLS
        
        # 3. Distance to nearest other agent
        if others_pos:
            distances = [abs(r - or_) + abs(c - oc) for (or_, oc) in others_pos]
            min_agent_dist = min(distances)
        else:
            min_agent_dist = C.GRID_ROWS + C.GRID_COLS
        
        # 4. Direction to nearest ball
        if balls:
            nearest_ball = min(balls, key=lambda b: abs(r - b[0]) + abs(c - b[1]))
            dr = nearest_ball[0] - r
            dc = nearest_ball[1] - c
            ball_dir = np.array([dr / C.GRID_ROWS, dc / C.GRID_COLS])
        else:
            ball_dir = np.zeros(2)
        
        # 5. Local obstacle map (5x5 grid around agent)
        obstacle_map = np.zeros(25)
        for i in range(-2, 3):
            for j in range(-2, 3):
                nr, nc = r + i, c + j
                if 0 <= nr < C.GRID_ROWS and 0 <= nc < C.GRID_COLS:
                    if (nr, nc) in obstacles:
                        obstacle_map[(i+2)*5 + (j+2)] = 1
        
        # 6. Agent ID (one-hot)
        agent_id_encoding = np.zeros(C.NUM_AGENTS)
        agent_id_encoding[self.agent_id] = 1
        
        # Concatenate all features
        state_vector = np.concatenate([
            pos_encoding,
            [min_dist / (C.GRID_ROWS + C.GRID_COLS)],  # Normalized distance
            [min_agent_dist / (C.GRID_ROWS + C.GRID_COLS)],
            ball_dir,
            obstacle_map,
            agent_id_encoding,
            [len(balls) / C.MAX_BALLS]  # Number of balls remaining
        ])
        
        return state_vector.astype(np.float32)
    
    def get_action(self, pos, balls, others_pos, obstacles, evaluate=False):
        """
        Select action using the policy network.
        
        Args:
            pos: agent position
            balls: list of ball positions
            others_pos: list of other agent positions
            obstacles: set of obstacle positions
            evaluate: if True, use deterministic action (for testing)
        
        Returns:
            action: int action index
            action_probs: probability distribution over actions
        """
        # Encode state
        state = self._encode_state(pos, balls, others_pos, obstacles)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Encode other agents' states
        other_states = []
        for other_pos in others_pos:
            other_state = self._encode_state(other_pos, balls, [], obstacles)
            other_states.append(other_state)
        
        other_states_tensor = None
        if other_states:
            other_states_tensor = torch.FloatTensor(np.array(other_states)).unsqueeze(0).to(self.device)
        
        # Get communication-enhanced state
        with torch.no_grad():
            integrated_state, _ = self.comm_module(state_tensor, other_states_tensor)
            action_probs = self.policy(integrated_state)
            
            if evaluate:
                # Deterministic action for evaluation
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample action from probability distribution with exploration
                if random.random() < self.exploration_noise:
                    action = random.choice(ACTIONS)
                else:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample().item()
        
        return action, action_probs.cpu().numpy()[0]
    
    def store_transition(self, pos, balls, others_pos, obstacles, action, reward, done):
        """Store transition for later learning"""
        state = self._encode_state(pos, balls, others_pos, obstacles)
        
        # Encode other agents' states
        other_states = []
        for other_pos in others_pos:
            other_state = self._encode_state(other_pos, balls, [], obstacles)
            other_states.append(other_state)
        
        # Get value estimate
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        other_states_tensor = None
        if other_states:
            other_states_tensor = torch.FloatTensor(np.array(other_states)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            integrated_state, _ = self.comm_module(state_tensor, other_states_tensor)
            value = self.value(integrated_state).item()
        
        # Store
        self.states.append(state)
        self.other_states_history.append(other_states)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        # Store log probability of the action taken
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if other_states:
            other_states_tensor = torch.FloatTensor(np.array(other_states)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            integrated_state, _ = self.comm_module(state_tensor, other_states_tensor)
            action_probs = self.policy(integrated_state)
            log_prob = torch.log(action_probs[0, action] + 1e-10)
            self.log_probs.append(log_prob.item())
    
    def compute_advantages(self):
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        values = self.values + [0]  # Add final value of 0
        dones = self.dones + [True]
        
        for t in reversed(range(len(self.rewards))):
            if dones[t]:
                gae = 0
            else:
                delta = self.rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def learn(self, done=False):
        """
        Learn from collected trajectories using PPO
        
        Args:
            done: if True, episode is complete and we should update
        """
        if not done and len(self.states) < self.batch_size:
            return
        
        # Compute advantages
        advantages = self.compute_advantages()
        returns = np.array(self.values) + advantages
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Process other states for batch
        other_states_batch = []
        for other_states in self.other_states_history:
            if other_states:
                other_states_batch.append(np.array(other_states))
            else:
                other_states_batch.append(np.zeros((0, states.shape[1])))
        
        # PPO update for multiple epochs
        for epoch in range(self.epochs):
            # Get current policy and value predictions
            integrated_states = []
            for i in range(len(states)):
                state_tensor = states[i:i+1]
                other_tensor = None
                if len(other_states_batch[i]) > 0:
                    other_tensor = torch.FloatTensor(other_states_batch[i]).unsqueeze(0).to(self.device)
                
                integrated_state, _ = self.comm_module(state_tensor, other_tensor)
                integrated_states.append(integrated_state)
            
            integrated_states = torch.cat(integrated_states, dim=0)
            
            # Policy loss
            action_probs = self.policy(integrated_states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values_pred = self.value(integrated_states).squeeze()
            value_loss = F.mse_loss(values_pred, returns_tensor)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update policy
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.other_states_history = []
        
        self.total_updates += 1
    
    def decay_exploration(self):
        """Decay exploration noise over time"""
        self.exploration_noise = max(C.EPSILON_MIN, self.exploration_noise * C.EPSILON_DECAY)
    
    def save(self, directory: str = C.MODEL_DIR):
        """Save neural network models"""
        os.makedirs(directory, exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'comm_state_dict': self.comm_module.state_dict(),
            'exploration_noise': self.exploration_noise,
            'total_updates': self.total_updates,
        }, os.path.join(directory, f'agent_{self.agent_id}_ppo.pt'))
        
        print(f"[Agent {self.agent_id}] Model saved")
    
    def load(self, directory: str = C.MODEL_DIR):
        """Load neural network models"""
        path = os.path.join(directory, f'agent_{self.agent_id}_ppo.pt')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.comm_module.load_state_dict(checkpoint['comm_state_dict'])
            self.exploration_noise = checkpoint.get('exploration_noise', C.EPSILON_START)
            self.total_updates = checkpoint.get('total_updates', 0)
            print(f"[Agent {self.agent_id}] Model loaded (updates: {self.total_updates})")
        else:
            print(f"[Agent {self.agent_id}] No saved model found — starting fresh.")


# Backward compatibility wrapper
class QLearner:
    """Wrapper for backward compatibility with existing code"""
    
    def __init__(self, agent_id: int, epsilon: float = C.EPSILON_START):
        self.agent_id = agent_id
        self.epsilon = epsilon
        self.q_table_size = 0
        self.total_updates = 0
        self.episode_rewards = []
    
    def best_action(self, state):
        return random.choice(ACTIONS)
    
    def select_action(self, state):
        return random.choice(ACTIONS)
    
    def update(self, state, action, reward, next_state, done):
        pass
    
    def decay_epsilon(self):
        self.epsilon = max(C.EPSILON_MIN, self.epsilon * C.EPSILON_DECAY)
    
    def save(self, directory=C.MODEL_DIR):
        pass
    
    def load(self, directory=C.MODEL_DIR):
        pass