import numpy as np

class TabularQAgent:
    def __init__(self, num_agents=3, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.num_agents = num_agents
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: State -> Action(5)
        self.q_table = {}
        
    def _get_direction_bin(self, r1, c1, r2, c2):
        if r1 == r2 and c1 == c2: return 0
        angle = np.arctan2(r2 - r1, c2 - c1) # returns -pi to pi
        # Bin into 8 directions (1 to 8)
        bin_idx = int(np.round(angle / (np.pi/4))) % 8
        return bin_idx + 1
        
    def _get_dist_bin(self, r1, c1, r2, c2):
        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist == 0: return 0
        if dist <= 5: return 1
        if dist <= 15: return 2
        return 3

    def _extract_state(self, state_dict, agent_idx):
        pos = state_dict['robots'][agent_idx]
        dirt_list = state_dict['dirt']
        
        if not dirt_list:
            dirt_dir = 0
            dirt_dist = 0
        else:
            closest_dirt = min(dirt_list, key=lambda d: abs(pos[0]-d[0]) + abs(pos[1]-d[1]))
            dirt_dir = self._get_direction_bin(pos[0], pos[1], closest_dirt[0], closest_dirt[1])
            dirt_dist = self._get_dist_bin(pos[0], pos[1], closest_dirt[0], closest_dirt[1])
            
        other_robots = [state_dict['robots'][i] for i in range(self.num_agents) if i != agent_idx]
        if not other_robots:
            rob_dir = 0
            rob_dist = 0
        else:
            closest_rob = min(other_robots, key=lambda r: abs(pos[0]-r[0]) + abs(pos[1]-r[1]))
            rob_dir = self._get_direction_bin(pos[0], pos[1], closest_rob[0], closest_rob[1])
            rob_dist = 1 if (abs(pos[0]-closest_rob[0]) + abs(pos[1]-closest_rob[1])) <= 3 else 2
            
        return (dirt_dir, dirt_dist, rob_dir, rob_dist)

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(5)
        return self.q_table[state]

    def act(self, state_dict, explore=True):
        actions = []
        for i in range(self.num_agents):
            s = self._extract_state(state_dict, i)
            if explore and np.random.rand() < self.epsilon:
                actions.append(np.random.randint(5))
            else:
                q_vals = self._get_q(s)
                actions.append(np.argmax(q_vals))
        return actions

    def learn(self, state_dict, actions, rewards, next_state_dict, done):
        for i in range(self.num_agents):
            s = self._extract_state(state_dict, i)
            ns = self._extract_state(next_state_dict, i)
            a = actions[i]
            r = rewards[i]
            
            q_s = self._get_q(s)
            q_ns = self._get_q(ns)
            
            best_next_q = np.max(q_ns) if not done else 0.0
            
            q_s[a] = q_s[a] + self.alpha * (r + self.gamma * best_next_q - q_s[a])
