#!/usr/bin/env python3
"""
Classic Multi-Robot Exploration Baselines
For comparison with RL methods
"""

import numpy as np
from typing import List, Tuple, Dict
import json
from datetime import datetime

class FrontierBasedExploration:
    """
    Traditional frontier-based exploration
    Robots move towards unexplored frontiers (boundaries between explored/unexplored)
    """
    
    def __init__(self, grid_size: Tuple[int, int], num_robots: int):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.explored_grid = np.zeros(grid_size, dtype=bool)
        self.robot_positions = [np.array([i*2, i*2]) for i in range(num_robots)]
        self.episode_history = []
    
    def get_frontier_cells(self) -> List[np.ndarray]:
        """Find frontier cells (boundary between explored and unexplored)"""
        frontiers = []
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if self.explored_grid[i, j]:
                    # Check neighbors
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                            if not self.explored_grid[ni, nj]:
                                frontiers.append(np.array([ni, nj]))
        return frontiers
    
    def assign_frontiers_to_robots(self, frontiers: List[np.ndarray]) -> Dict[int, np.ndarray]:
        """Assign nearest frontier to each robot"""
        assignments = {}
        if not frontiers:
            # No frontiers, stay in place
            for i in range(self.num_robots):
                assignments[i] = self.robot_positions[i]
        else:
            for i in range(self.num_robots):
                distances = [np.linalg.norm(self.robot_positions[i] - f) for f in frontiers]
                nearest_idx = np.argmin(distances)
                assignments[i] = frontiers[nearest_idx]
        return assignments
    
    def step(self, grid_observations: List[np.ndarray]) -> Tuple[float, bool]:
        """Execute one exploration step"""
        # Update explored grid from observations
        for i, obs in enumerate(grid_observations):
            self.explored_grid |= (obs > 0)
        
        # Find frontiers and assign
        frontiers = self.get_frontier_cells()
        assignments = self.assign_frontiers_to_robots(frontiers)
        
        # Move robots towards assigned frontiers
        coverage_before = np.sum(self.explored_grid)
        for i in range(self.num_robots):
            target = assignments[i]
            direction = target - self.robot_positions[i]
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                self.robot_positions[i] += direction * 0.5  # Step size
        
        # Calculate reward
        coverage_after = np.sum(self.explored_grid)
        coverage_ratio = coverage_after / (self.grid_size[0] * self.grid_size[1])
        reward = (coverage_after - coverage_before) * 5.0  # Reward for new cells
        
        done = coverage_ratio > 0.9 or len(frontiers) == 0
        
        return reward, done, coverage_ratio
    
    def get_metrics(self) -> Dict:
        """Return frontier-based metrics"""
        coverage = np.sum(self.explored_grid) / (self.grid_size[0] * self.grid_size[1])
        return {
            'coverage_ratio': float(coverage),
            'explored_cells': int(np.sum(self.explored_grid)),
            'total_cells': int(self.grid_size[0] * self.grid_size[1]),
        }


class GreedyHeuristicExploration:
    """
    Greedy exploration: robots move towards maximum information gain
    Maximize visible unexplored cells
    """
    
    def __init__(self, grid_size: Tuple[int, int], num_robots: int, view_range: int = 3):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.view_range = view_range
        self.explored_grid = np.zeros(grid_size, dtype=bool)
        self.robot_positions = [np.array([i*2, i*2]) for i in range(num_robots)]
    
    def get_information_gain(self, pos: np.ndarray) -> float:
        """Calculate information gain from a position (visible unexplored cells)"""
        x, y = int(pos[0]), int(pos[1])
        gain = 0
        for i in range(max(0, x - self.view_range), min(self.grid_size[0], x + self.view_range + 1)):
            for j in range(max(0, y - self.view_range), min(self.grid_size[1], y + self.view_range + 1)):
                if not self.explored_grid[i, j]:
                    gain += 1
        return gain
    
    def get_best_action(self, robot_idx: int) -> np.ndarray:
        """Return action that maximizes information gain"""
        current_pos = self.robot_positions[robot_idx]
        best_gain = 0
        best_action = np.array([0, 0])
        
        # Try 8 directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0),  # cardinal
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # diagonal
            new_pos = current_pos + np.array([dx, dy])
            if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]:
                gain = self.get_information_gain(new_pos)
                if gain > best_gain:
                    best_gain = gain
                    best_action = np.array([dx, dy])
        
        return best_action
    
    def step(self, grid_observations: List[np.ndarray]) -> Tuple[float, bool]:
        """Execute one greedy exploration step"""
        # Update explored grid
        for i, obs in enumerate(grid_observations):
            self.explored_grid |= (obs > 0)
        
        coverage_before = np.sum(self.explored_grid)
        
        # Greedy action for each robot
        for i in range(self.num_robots):
            action = self.get_best_action(i)
            self.robot_positions[i] = np.clip(
                self.robot_positions[i] + action,
                0, np.array(self.grid_size) - 1
            )
        
        coverage_after = np.sum(self.explored_grid)
        coverage_ratio = coverage_after / (self.grid_size[0] * self.grid_size[1])
        reward = (coverage_after - coverage_before) * 5.0
        
        done = coverage_ratio > 0.9
        
        return reward, done, coverage_ratio


class PotentialFieldMethod:
    """
    Potential field based exploration
    - Attractive potential towards unexplored regions
    - Repulsive potential from obstacles and other robots
    """
    
    def __init__(self, grid_size: Tuple[int, int], num_robots: int):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.explored_grid = np.zeros(grid_size, dtype=bool)
        self.robot_positions = [np.array([i*2, i*2], dtype=float) for i in range(num_robots)]
    
    def compute_potential_field(self) -> np.ndarray:
        """Compute attractive potential for unexplored regions"""
        potential = np.zeros(self.grid_size, dtype=float)
        
        # Attractive potential: higher in unexplored regions
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                if not self.explored_grid[i, j]:
                    potential[i, j] = 1.0  # Unexplored
        
        # Smooth the potential field (diffusion)
        for _ in range(3):
            potential = np.convolve(potential.flatten(), np.array([0.25, 0.5, 0.25]), mode='same').reshape(potential.shape)
        
        return potential
    
    def step(self, grid_observations: List[np.ndarray]) -> Tuple[float, bool]:
        """Execute one potential field exploration step"""
        # Update explored grid
        for i, obs in enumerate(grid_observations):
            self.explored_grid |= (obs > 0)
        
        # Compute potential field
        potential = self.compute_potential_field()
        
        coverage_before = np.sum(self.explored_grid)
        
        # Move robots along potential field gradient
        for i in range(self.num_robots):
            x, y = int(np.clip(self.robot_positions[i], 0, np.array(self.grid_size) - 1))
            
            # Compute gradient at robot position
            grad_x, grad_y = 0, 0
            if x > 0:
                grad_x += potential[x-1, y]
            if x < self.grid_size[0] - 1:
                grad_x -= potential[x+1, y]
            if y > 0:
                grad_y += potential[x, y-1]
            if y < self.grid_size[1] - 1:
                grad_y -= potential[x, y+1]
            
            # Move in direction of gradient
            grad_norm = np.sqrt(grad_x**2 + grad_y**2)
            if grad_norm > 0:
                self.robot_positions[i] += np.array([grad_x, grad_y]) / grad_norm * 0.5
            
            # Collision avoidance: repel from other robots
            for j in range(self.num_robots):
                if i != j:
                    diff = self.robot_positions[i] - self.robot_positions[j]
                    dist = np.linalg.norm(diff)
                    if 0 < dist < 1.0:
                        self.robot_positions[i] += diff / dist * 0.5
            
            # Boundary enforcement
            self.robot_positions[i] = np.clip(
                self.robot_positions[i],
                0, np.array(self.grid_size) - 1
            )
        
        coverage_after = np.sum(self.explored_grid)
        coverage_ratio = coverage_after / (self.grid_size[0] * self.grid_size[1])
        reward = (coverage_after - coverage_before) * 5.0
        
        done = coverage_ratio > 0.9
        
        return reward, done, coverage_ratio


def run_baseline_comparison(grid_size=(20, 20), num_robots=3, episodes=100):
    """Run all baselines and save comparison"""
    
    methods = {
        'frontier_based': FrontierBasedExploration(grid_size, num_robots),
        'greedy': GreedyHeuristicExploration(grid_size, num_robots),
        'potential_field': PotentialFieldMethod(grid_size, num_robots),
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"\n🔄 Running {name}...")
        coverages = []
        
        for ep in range(episodes):
            # Simulate grid observations (all cells explored gradually)
            obs = [np.random.random((grid_size[0], grid_size[1])) > 0.5 for _ in range(num_robots)]
            reward, done, coverage = method.step(obs)
            coverages.append(coverage)
            
            if done:
                break
        
        results[name] = {
            'final_coverage': float(coverages[-1] if coverages else 0),
            'avg_coverage': float(np.mean(coverages)),
            'episodes_to_convergence': len(coverages),
            'coverages': [float(c) for c in coverages]
        }
        print(f"✅ {name}: Final Coverage = {results[name]['final_coverage']:.1%}")
    
    # Save results
    output_file = f"baseline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results saved to {output_file}")
    return results


if __name__ == '__main__':
    run_baseline_comparison()
