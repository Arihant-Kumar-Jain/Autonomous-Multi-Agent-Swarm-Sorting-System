# env/environment.py — Environment for multi-agent ball collection

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from utils.utils import manhattan_distance, is_valid_position, get_random_position


class Environment:
    """
    Multi-agent environment for ball collection.
    
    Features:
    - Grid world with obstacles
    - Multiple agents collecting balls
    - Collision detection between agents
    - Reward calculation
    """
    
    def __init__(self):
        self.grid_rows = C.GRID_ROWS
        self.grid_cols = C.GRID_COLS
        self.num_agents = C.NUM_AGENTS
        self.max_balls = C.MAX_BALLS
        
        # Initialize environment state
        self.obstacles = set()
        self.balls = []
        self.agent_positions = []
        
        # Statistics
        self.step_count = 0
        self.collision_count = 0
        self.balls_collected = 0
        self.episode_reward = 0
        
        # Reset to generate initial state
        self.reset()
    
    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            list: Starting positions for all agents
        """
        self.step_count = 0
        self.collision_count = 0
        self.balls_collected = 0
        self.episode_reward = 0
        
        # Generate obstacles
        self.obstacles = self._generate_obstacles()
        
        # Generate balls
        self.balls = self._generate_balls()
        
        # Generate agent positions (ensuring they don't overlap with obstacles or each other)
        self.agent_positions = self._generate_agent_positions()
        
        return self.agent_positions.copy()
    
    def _generate_obstacles(self):
        """Generate random obstacle positions."""
        obstacles = set()
        while len(obstacles) < C.NUM_OBSTACLES:
            pos = (random.randint(0, self.grid_rows - 1),
                   random.randint(0, self.grid_cols - 1))
            obstacles.add(pos)
        return obstacles
    
    def _generate_balls(self):
        """Generate random ball positions (not on obstacles)."""
        balls = []
        while len(balls) < self.max_balls:
            pos = (random.randint(0, self.grid_rows - 1),
                   random.randint(0, self.grid_cols - 1))
            if pos not in self.obstacles and pos not in balls:
                balls.append(pos)
        return balls
    
    def _generate_agent_positions(self):
        """Generate random starting positions for agents."""
        positions = []
        while len(positions) < self.num_agents:
            pos = (random.randint(0, self.grid_rows - 1),
                   random.randint(0, self.grid_cols - 1))
            if pos not in self.obstacles and pos not in positions:
                positions.append(pos)
        return positions
    
    def step(self, proposed_positions):
        """
        Execute a step in the environment.
        
        Args:
            proposed_positions: List of proposed new positions for each agent
        
        Returns:
            rewards: List of rewards for each agent
            new_positions: Actual new positions after collision resolution
            done: Whether the episode is complete
            info: Additional information dictionary
        """
        # Validate and adjust positions (boundaries and obstacles)
        validated_positions = []
        for i, pos in enumerate(proposed_positions):
            r, c = pos
            # Check boundaries
            if r < 0 or r >= self.grid_rows or c < 0 or c >= self.grid_cols:
                # Keep current position if invalid
                validated_positions.append(self.agent_positions[i])
            # Check obstacles
            elif pos in self.obstacles:
                validated_positions.append(self.agent_positions[i])
            else:
                validated_positions.append(pos)
        
        # Resolve collisions (multiple agents trying to move to same cell)
        new_positions = self._resolve_collisions(validated_positions)
        
        # Check for ball collections
        rewards = [C.REWARD_STEP] * self.num_agents
        collected_balls = []
        
        for i, pos in enumerate(new_positions):
            if pos in self.balls:
                # Agent collected a ball
                rewards[i] += C.REWARD_COLLECT_BALL
                collected_balls.append(pos)
                self.balls_collected += 1
        
        # Remove collected balls
        for ball in collected_balls:
            if ball in self.balls:
                self.balls.remove(ball)
        
        # Check for collisions between agents
        collision_count_step = 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if new_positions[i] == new_positions[j]:
                    # Both agents collided
                    rewards[i] += C.REWARD_COLLISION
                    rewards[j] += C.REWARD_COLLISION
                    collision_count_step += 1
        
        self.collision_count += collision_count_step
        
        # Update agent positions
        self.agent_positions = new_positions
        
        # Check if episode is done
        done = len(self.balls) == 0
        
        # Add bonus if all balls collected
        if done:
            for i in range(self.num_agents):
                rewards[i] += C.REWARD_BONUS_ALL_BALLS
        
        # Update step count
        self.step_count += 1
        
        # Update episode reward
        self.episode_reward = sum(rewards)
        
        # Prepare info dictionary
        info = {
            "collisions_step": collision_count_step,
            "collected": [pos in collected_balls for pos in new_positions],
            "balls_remaining": len(self.balls)
        }
        
        return rewards, new_positions, done, info
    
    def _resolve_collisions(self, proposed_positions):
        """
        Resolve collisions when multiple agents try to move to the same cell.
        
        Strategy: If multiple agents try to move to the same cell, they all stay
        in their original positions.
        
        Args:
            proposed_positions: List of proposed positions
        
        Returns:
            list: Resolved positions
        """
        resolved_positions = proposed_positions.copy()
        
        # Check for conflicts
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if resolved_positions[i] == resolved_positions[j]:
                    # Conflict detected - both stay in original positions
                    resolved_positions[i] = self.agent_positions[i]
                    resolved_positions[j] = self.agent_positions[j]
        
        return resolved_positions
    
    def get_local_obs(self, agent_pos, radius: int = C.OBS_RADIUS):
        """
        Get local observation around an agent.
        
        Args:
            agent_pos: Position of the agent
            radius: Radius of local observation window
        
        Returns:
            dict: Local observation containing nearby balls, agents, and obstacles
        """
        r, c = agent_pos
        local_balls = []
        local_agents = []
        local_obstacles = []
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                nr, nc = r + i, c + j
                if 0 <= nr < self.grid_rows and 0 <= nc < self.grid_cols:
                    pos = (nr, nc)
                    if pos in self.balls:
                        local_balls.append((i, j))
                    elif pos in self.obstacles:
                        local_obstacles.append((i, j))
                    elif pos in self.agent_positions and pos != agent_pos:
                        local_agents.append((i, j))
        
        return {
            'balls': local_balls,
            'agents': local_agents,
            'obstacles': local_obstacles
        }
    
    def render(self):
        """
        Simple text-based rendering for debugging.
        """
        grid = [['.' for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        
        # Place obstacles
        for r, c in self.obstacles:
            grid[r][c] = '#'
        
        # Place balls
        for r, c in self.balls:
            grid[r][c] = 'B'
        
        # Place agents
        for i, (r, c) in enumerate(self.agent_positions):
            if grid[r][c] == 'B':
                grid[r][c] = str(i)
            else:
                grid[r][c] = str(i)
        
        # Print grid
        print("\n" + "-" * (self.grid_cols * 2 + 1))
        for row in grid:
            print("|" + " ".join(row) + "|")
        print("-" * (self.grid_cols * 2 + 1))
        print(f"Balls remaining: {len(self.balls)}")
        print(f"Steps: {self.step_count}")
        print(f"Collisions: {self.collision_count}")
    
    def get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            dict: Current environment state
        """
        return {
            'agent_positions': self.agent_positions.copy(),
            'balls': self.balls.copy(),
            'obstacles': self.obstacles.copy(),
            'step_count': self.step_count,
            'collision_count': self.collision_count,
            'balls_collected': self.balls_collected
        }


if __name__ == "__main__":
    # Test the environment
    env = Environment()
    print("Environment initialized successfully!")
    print(f"Grid size: {env.grid_rows}x{env.grid_cols}")
    print(f"Number of agents: {env.num_agents}")
    print(f"Number of obstacles: {len(env.obstacles)}")
    print(f"Number of balls: {len(env.balls)}")
    
    # Test a few steps
    for step in range(5):
        # Random actions for testing
        proposed = []
        for pos in env.agent_positions:
            r, c = pos
            action = random.choice(['up', 'down', 'left', 'right', 'stay'])
            if action == 'up':
                proposed.append((r - 1, c))
            elif action == 'down':
                proposed.append((r + 1, c))
            elif action == 'left':
                proposed.append((r, c - 1))
            elif action == 'right':
                proposed.append((r, c + 1))
            else:
                proposed.append((r, c))
        
        rewards, new_positions, done, info = env.step(proposed)
        print(f"\nStep {step + 1}:")
        print(f"  Rewards: {rewards}")
        print(f"  Balls remaining: {info['balls_remaining']}")
        print(f"  Collisions this step: {info['collisions_step']}")
        
        if done:
            print("  Episode complete!")
            break
    
    print("\nEnvironment test complete!")