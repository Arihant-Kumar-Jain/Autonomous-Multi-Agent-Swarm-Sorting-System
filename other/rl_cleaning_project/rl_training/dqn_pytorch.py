import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, num_agents=3, fov=11, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, batch_size=64):
        self.num_agents = num_agents
        self.fov = fov
        self.state_dim = fov * fov + 2
        self.action_dim = 5
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        
    def _extract_state(self, state_dict, agent_idx):
        grid = state_dict['grid']
        pos = state_dict['robots'][agent_idx]
        r, c = pos
        
        half_fov = self.fov // 2
        local_grid = np.full((self.fov, self.fov), -1.0) # default wall
        
        for i in range(self.fov):
            for j in range(self.fov):
                gr = r - half_fov + i
                gc = c - half_fov + j
                if 0 <= gr < grid.shape[0] and 0 <= gc < grid.shape[1]:
                    local_grid[i, j] = grid[gr, gc]
                    
        # mark other robots
        for j, other_pos in enumerate(state_dict['robots']):
            if j != agent_idx:
                ogr, ogc = other_pos
                if abs(ogr - r) <= half_fov and abs(ogc - c) <= half_fov:
                    local_grid[ogr - r + half_fov, ogc - c + half_fov] = 2.0
                    
        # direction to dirt
        dirt_list = state_dict['dirt']
        if dirt_list:
            closest_dirt = min(dirt_list, key=lambda d: abs(pos[0]-d[0]) + abs(pos[1]-d[1]))
            dy = closest_dirt[0] - r
            dx = closest_dirt[1] - c
            norm = np.sqrt(dx**2 + dy**2) + 1e-5
            dx /= norm
            dy /= norm
        else:
            dx, dy = 0.0, 0.0
            
        flat_state = np.concatenate([local_grid.flatten(), [dy, dx]])
        return flat_state

    def act(self, state_dict, explore=True):
        actions = []
        for i in range(self.num_agents):
            if explore and random.random() < self.epsilon:
                actions.append(random.randint(0, 4))
            else:
                s = self._extract_state(state_dict, i)
                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_vals = self.q_network(s_tensor)
                actions.append(torch.argmax(q_vals).item())
        return actions

    def learn(self, state_dict, actions, rewards, next_state_dict, done):
        for i in range(self.num_agents):
            s = self._extract_state(state_dict, i)
            ns = self._extract_state(next_state_dict, i)
            a = actions[i]
            r = rewards[i]
            self.memory.append((s, a, r, ns, done))
            
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, acts, rews, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        acts = torch.LongTensor(acts).to(self.device)
        rews = torch.FloatTensor(rews).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        q_vals = self.q_network(states).gather(1, acts.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_vals = self.target_network(next_states).max(1)[0]
        target_q = rews + self.gamma * next_q_vals * (1 - dones)
        
        loss = nn.MSELoss()(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
