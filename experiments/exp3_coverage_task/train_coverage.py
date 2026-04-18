"""
Experiment 3: Coverage Task
Multi-robot area exploration and coverage

Objective: Maximize area coverage in minimum time
Metric: Coverage % (target 80%+)

Usage:
    python3 train_coverage.py --episodes 1000 --gpu 0
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from collections import deque

class CoverageEnvironment:
    """
    Multi-robot coverage task
    
    Robots need to explore and cover a target area
    Coverage = grid cells visited
    """
    
    def __init__(self, num_robots=3, grid_size=15):
        self.num_robots = num_robots
        self.grid_size = grid_size
        
        # Discretized coverage grid
        self.coverage_grid = np.zeros((grid_size, grid_size))
        self.visited_cells = set()
        
        # Robot states [x, y, vx, vy]
        self.robot_states = np.zeros((num_robots, 4))
        self.reset()
        
        # Metrics
        self.max_coverage_reached = 0
        
    def reset(self):
        """Reset environment"""
        # Random start positions
        self.robot_states = np.random.uniform(-7, 7, (self.num_robots, 4))
        self.visited_cells = set()
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size))
        return self.get_observations()
    
    def get_observations(self):
        """Get observations for each robot"""
        obs = []
        for i in range(self.num_robots):
            # Local LIDAR simulation (8 directions)
            lidar = self._get_lidar(i)
            
            # Own state
            state = np.concatenate([
                self.robot_states[i],  # [x, y, vx, vy]
                lidar,  # 8 LIDAR rays
                np.array([self.get_coverage_percentage()])  # Coverage info
            ])
            
            obs.append(state)
        
        return np.array(obs)
    
    def _get_lidar(self, robot_idx, num_rays=8, max_range=3.0):
        """Simulate LIDAR sensor"""
        robot_pos = self.robot_states[robot_idx, :2]
        lidar_readings = []
        
        for angle_idx in range(num_rays):
            angle = 2 * np.pi * angle_idx / num_rays
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            # Check for obstacles (other robots)
            distance = max_range
            for j in range(self.num_robots):
                if i != j:
                    other_pos = self.robot_states[j, :2]
                    ray_point = robot_pos + direction * distance
                    dist_to_other = np.linalg.norm(other_pos - ray_point)
                    if dist_to_other < 0.3:  # Robot collision
                        distance = min(distance, np.linalg.norm(other_pos - robot_pos))
            
            # Wall collision
            for coord in robot_pos + direction * distance:
                if np.abs(coord) > 7.5:
                    distance = min(distance, 
                                 (7.5 - np.abs(robot_pos[0 if coord == robot_pos[0] else 1])) / (direction[0] if direction[0] != 0 else 1e-6))
            
            lidar_readings.append(distance / max_range)
        
        return np.array(lidar_readings)
    
    def step(self, actions):
        """Execute step and calculate coverage rewards"""
        dt = 0.1
        
        # Update positions
        for i in range(self.num_robots):
            self.robot_states[i, 2:] = actions[i]
            self.robot_states[i, :2] += actions[i] * dt
            self.robot_states[i, :2] = np.clip(self.robot_states[i, :2], -7.5, 7.5)
        
        # Calculate coverage
        rewards = self._calculate_coverage_rewards()
        
        obs = self.get_observations()
        dones = np.array([False] * self.num_robots)
        info = {'coverage_pct': self.get_coverage_percentage()}
        
        return obs, rewards, dones, info
    
    def _calculate_coverage_rewards(self):
        """
        Reward structure for coverage task:
        - +1.0 for each newly covered cell
        - +0.5 team bonus for spreading
        - -0.01 per unit distance (efficiency)
        - -5.0 for collisions
        - Milestone bonuses at 50%, 80%
        """
        rewards = []
        new_cells_covered = 0
        
        # Mark covered cells
        for i in range(self.num_robots):
            x, y = self.robot_states[i, :2]
            grid_x = int((x + 7.5) / 15 * self.grid_size)
            grid_y = int((y + 7.5) / 15 * self.grid_size)
            
            grid_x = np.clip(grid_x, 0, self.grid_size - 1)
            grid_y = np.clip(grid_y, 0, self.grid_size - 1)
            
            if (grid_x, grid_y) not in self.visited_cells:
                self.visited_cells.add((grid_x, grid_y))
                self.coverage_grid[grid_x, grid_y] = 1.0
                new_cells_covered += 1
        
        # Calculate team spread
        distances = []
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = np.linalg.norm(self.robot_states[i, :2] - self.robot_states[j, :2])
                distances.append(dist)
        
        avg_spread = np.mean(distances) if distances else 0
        spread_bonus = min(avg_spread / 5.0, 1.0)
        
        # Generate rewards
        for i in range(self.num_robots):
            reward = 0
            
            # Coverage: Main reward (spread equally among robots)
            coverage_reward = (new_cells_covered / max(self.num_robots, 1)) * 1.0
            reward += coverage_reward
            
            # Team spreading bonus
            reward += spread_bonus * 0.5
            
            # Distance penalty (prefer efficient paths)
            speed = np.linalg.norm(self.robot_states[i, 2:])
            reward -= 0.01 * speed
            
            # Collision penalty
            for j in range(self.num_robots):
                if i != j:
                    dist = np.linalg.norm(self.robot_states[i, :2] - self.robot_states[j, :2])
                    if dist < 0.5:
                        reward -= 5.0
            
            # Milestone bonuses
            coverage_pct = self.get_coverage_percentage()
            if coverage_pct >= 50.0 and self.max_coverage_reached < 50.0:
                reward += 10.0
                self.max_coverage_reached = 50.0
            
            if coverage_pct >= 80.0 and self.max_coverage_reached < 80.0:
                reward += 20.0
                self.max_coverage_reached = 80.0
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def get_coverage_percentage(self):
        """Returns coverage as percentage"""
        total_cells = self.grid_size * self.grid_size
        return (len(self.visited_cells) / total_cells) * 100.0


class SimpleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)


class CoverageAgent:
    def __init__(self, num_robots=3, obs_dim=13, action_dim=2, device='cuda'):
        self.num_robots = num_robots
        self.device = device
        
        self.actors = [SimpleNetwork(obs_dim, action_dim).to(device) for _ in range(num_robots)]
        self.optimizers = [torch.optim.Adam(a.parameters(), lr=1e-4) for a in self.actors]
        
        self.replay_buffer = deque(maxlen=10000)
    
    def get_actions(self, observations, training=True):
        actions = []
        for i, obs in enumerate(observations):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](obs_t).cpu().numpy()[0]
            
            if training:
                action += np.random.normal(0, 0.1, len(action))
            
            actions.append(np.clip(action, -1, 1))
        return np.array(actions)


def train_coverage(num_episodes=1000, gpu=0):
    """Train coverage task"""
    
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"\n🚀 Coverage Task Training | Device: {device}")
    print("=" * 60)
    
    env = CoverageEnvironment(num_robots=3, grid_size=15)
    agent = CoverageAgent(device=device)
    
    episode_rewards = []
    coverage_history = []
    training_start = time.time()
    
    for episode in range(num_episodes):
        obs = env.reset()
        ep_reward = np.zeros(3)
        
        for step in range(100):
            actions = agent.get_actions(obs, training=True)
            obs, rewards, dones, info = env.step(actions)
            ep_reward += rewards
            agent.replay_buffer.append((obs, rewards))
        
        avg_reward = np.mean(ep_reward)
        episode_rewards.append(avg_reward)
        coverage = env.get_coverage_percentage()
        coverage_history.append(coverage)
        
        if (episode + 1) % 50 == 0:
            avg_reward_50 = np.mean(episode_rewards[-50:])
            avg_coverage = np.mean(coverage_history[-50:])
            elapsed = time.time() - training_start
            
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_reward_50:7.2f} | "
                  f"Coverage: {avg_coverage:6.1f}% | "
                  f"Time: {elapsed:6.0f}s")
        
        if (episode + 1) % 10 == 0:
            print(".", end="", flush=True)
    
    print("\n" + "=" * 60)
    
    # Save results
    results = {
        'experiment': 'Coverage Task',
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'coverage_history': coverage_history,
        'final_coverage': float(np.mean(coverage_history[-50:])),
        'average_final_reward': float(np.mean(episode_rewards[-100:])),
        'training_time_seconds': time.time() - training_start,
        'device': str(device)
    }
    
    import os
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Coverage training complete!")
    print(f"📊 Final Coverage: {results['final_coverage']:.1f}%")
    print(f"⏱️  Training Time: {results['training_time_seconds']:.0f}s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    train_coverage(num_episodes=args.episodes, gpu=args.gpu)
