# agents/agent.py — Agent using neural policy network with communication

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from rl.rl_model import MultiAgentPPO, QLearner, ACTIONS, ACTION_DELTAS
from utils.utils import nearest_ball, manhattan_distance


class Agent:
    """
    Multi-Agent RL Agent using Neural Policy Network with Communication.
    
    Each agent has:
    - Neural network policy that outputs action probabilities
    - Communication module to share information with other agents
    - Value network for advantage estimation
    """
    
    def __init__(self, agent_id: int, mode: str = "rl"):
        self.agent_id = agent_id
        self.mode = mode
        self.pos = (0, 0)
        
        # For rule-based mode
        self.bfs_path = []
        
        # Statistics
        self.balls_collected = 0
        self.episode_reward = 0
        self.prev_dist_to_ball = 0
        
        # Store last action for learning
        self._last_action = 4  # Default to STAY
        
        # State dimension (calculated from encode_state in rl_model)
        state_dim = (C.GRID_ROWS * C.GRID_COLS) + 1 + 1 + 2 + 25 + C.NUM_AGENTS + 1
        
        # Initialize neural network (or Q-learner for fallback)
        if mode == "rl":
            self.rl_model = MultiAgentPPO(agent_id, state_dim)
        else:
            self.rl_model = QLearner(agent_id)
        
        # For compatibility
        self.q_table_size = 0
        self.epsilon = C.EPSILON_START
    
    def reset(self, start_pos):
        """Reset agent state at episode start"""
        self.pos = start_pos
        self.balls_collected = 0
        self.episode_reward = 0
        self.bfs_path = []
        self._last_action = 4  # Reset last action
    
    def select_action(self, balls, other_positions, obstacles):
        """
        Select action based on current mode.
        
        Returns:
            new_position, action_index
        """
        if self.mode == "rl":
            action, probs = self.rl_model.get_action(
                self.pos, balls, other_positions, obstacles, evaluate=False
            )
        else:
            # Rule-based using BFS
            action = self._rule_based_action(balls, other_positions, obstacles)
        
        # Store the action for learning
        self._last_action = action
        
        # Calculate new position
        dr, dc = ACTION_DELTAS[action]
        r, c = self.pos
        new_r = max(0, min(C.GRID_ROWS - 1, r + dr))
        new_c = max(0, min(C.GRID_COLS - 1, c + dc))
        new_pos = (new_r, new_c)
        
        return new_pos, action
    
    def _rule_based_action(self, balls, other_positions, obstacles):
        """Rule-based action selection using BFS to nearest ball"""
        if not balls:
            return 4  # STAY
        
        # Find nearest ball using Manhattan distance
        target = None
        min_dist = float('inf')
        
        for ball in balls:
            dist = manhattan_distance(self.pos, ball)
            if dist < min_dist:
                min_dist = dist
                target = ball
        
        if target:
            # Simple greedy movement toward target (avoid obstacles)
            r, c = self.pos
            tr, tc = target
            
            # Try to move in the direction that reduces distance
            if r < tr and (r + 1, c) not in obstacles:
                return 1  # DOWN
            elif r > tr and (r - 1, c) not in obstacles:
                return 0  # UP
            elif c < tc and (r, c + 1) not in obstacles:
                return 3  # RIGHT
            elif c > tc and (r, c - 1) not in obstacles:
                return 2  # LEFT
        
        return 4  # STAY
    
    def learn(self, reward, balls, other_positions, done):
        """
        Update agent's policy based on experience.
        """
        self.episode_reward += reward
        
        # Only store and learn if we're in RL mode
        if self.mode == "rl" and hasattr(self.rl_model, 'store_transition'):
            # Store transition in agent's memory
            self.rl_model.store_transition(
                self.pos, balls, other_positions, set(),  # obstacles
                self._last_action,
                reward,
                done
            )
            
            # Learn if episode is done
            if done:
                self.rl_model.learn(done=True)
                self.rl_model.decay_exploration()
                
                # Update epsilon for display
                if hasattr(self.rl_model, 'exploration_noise'):
                    self.epsilon = self.rl_model.exploration_noise
                if hasattr(self.rl_model, 'q_table'):
                    self.q_table_size = len(self.rl_model.q_table)
    
    def _store_last_action(self, action):
        """Store last action for learning"""
        self._last_action = action
    
    def end_episode(self):
        """Called at the end of each episode"""
        # Reset episode-specific variables
        self.episode_reward = 0
        # Don't reset balls_collected as it's cumulative across episodes
    
    def save(self):
        """Save agent's model"""
        if hasattr(self.rl_model, 'save'):
            self.rl_model.save()
    
    def load(self):
        """Load agent's model"""
        if hasattr(self.rl_model, 'load'):
            self.rl_model.load()