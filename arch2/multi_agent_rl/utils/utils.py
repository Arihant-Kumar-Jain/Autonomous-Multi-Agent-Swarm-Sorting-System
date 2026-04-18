# utils/utils.py — Utility functions for the multi-agent RL system

import os
import json
from datetime import datetime
import config as C


def manhattan_distance(pos1, pos2):
    """
    Calculate Manhattan distance between two positions.
    
    Args:
        pos1: (row, col) tuple
        pos2: (row, col) tuple
    
    Returns:
        int: Manhattan distance
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def euclidean_distance(pos1, pos2):
    """
    Calculate Euclidean distance between two positions.
    
    Args:
        pos1: (row, col) tuple
        pos2: (row, col) tuple
    
    Returns:
        float: Euclidean distance
    """
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def nearest_ball(pos, balls):
    """
    Find the nearest ball to the agent.
    
    Args:
        pos: (row, col) position of agent
        balls: list of ball positions
    
    Returns:
        tuple: (distance, nearest_ball_position)
               distance is float('inf') if no balls
    """
    if not balls:
        return float('inf'), None
    
    min_dist = float('inf')
    nearest = None
    
    for ball in balls:
        dist = manhattan_distance(pos, ball)
        if dist < min_dist:
            min_dist = dist
            nearest = ball
    
    return min_dist, nearest


def nearest_agent(pos, other_positions):
    """
    Find the nearest other agent.
    
    Args:
        pos: (row, col) position of agent
        other_positions: list of other agent positions
    
    Returns:
        tuple: (distance, nearest_agent_position)
               distance is float('inf') if no other agents
    """
    if not other_positions:
        return float('inf'), None
    
    min_dist = float('inf')
    nearest = None
    
    for other_pos in other_positions:
        dist = manhattan_distance(pos, other_pos)
        if dist < min_dist:
            min_dist = dist
            nearest = other_pos
    
    return min_dist, nearest


def encode_state(pos, balls, other_positions, obstacles, agent_id, num_agents):
    """
    Encode the full state into a feature vector for neural network input.
    
    Args:
        pos: (row, col) agent position
        balls: list of ball positions
        other_positions: list of other agent positions
        obstacles: set of obstacle positions
        agent_id: integer agent ID
        num_agents: total number of agents
    
    Returns:
        numpy array: encoded state vector
    """
    import numpy as np
    
    r, c = pos
    
    # 1. One-hot position encoding
    pos_encoding = np.zeros(C.GRID_ROWS * C.GRID_COLS)
    pos_encoding[r * C.GRID_COLS + c] = 1
    
    # 2. Distance to nearest ball
    ball_dist, _ = nearest_ball(pos, balls)
    if ball_dist == float('inf'):
        ball_dist = C.GRID_ROWS + C.GRID_COLS
    normalized_ball_dist = ball_dist / (C.GRID_ROWS + C.GRID_COLS)
    
    # 3. Distance to nearest agent
    agent_dist, _ = nearest_agent(pos, other_positions)
    if agent_dist == float('inf'):
        agent_dist = C.GRID_ROWS + C.GRID_COLS
    normalized_agent_dist = agent_dist / (C.GRID_ROWS + C.GRID_COLS)
    
    # 4. Direction to nearest ball
    if balls:
        _, nearest_ball_pos = nearest_ball(pos, balls)
        dr = nearest_ball_pos[0] - r
        dc = nearest_ball_pos[1] - c
        # Normalize to [-1, 1]
        ball_dir = np.array([dr / C.GRID_ROWS, dc / C.GRID_COLS])
    else:
        ball_dir = np.zeros(2)
    
    # 5. Local obstacle map (5x5 grid around agent)
    obstacle_map = np.zeros(25)
    for i in range(-2, 4):
        for j in range(-2, 4):
            nr, nc = r + i, c + j
            if 0 <= nr < C.GRID_ROWS and 0 <= nc < C.GRID_COLS:
                if (nr, nc) in obstacles:
                    idx = (i + 2) * 5 + (j + 2)
                    if 0 <= idx < 25:
                        obstacle_map[idx] = 1
    
    # 6. Agent ID (one-hot)
    agent_id_encoding = np.zeros(num_agents)
    agent_id_encoding[agent_id] = 1
    
    # 7. Number of balls remaining
    balls_remaining = len(balls) / C.MAX_BALLS
    
    # Concatenate all features
    state_vector = np.concatenate([
        pos_encoding,
        [normalized_ball_dist],
        [normalized_agent_dist],
        ball_dir,
        obstacle_map,
        agent_id_encoding,
        [balls_remaining]
    ])
    
    return state_vector.astype(np.float32)


def log_episode(episode, steps, collisions, balls_collected, epsilon):
    """
    Log episode statistics to a CSV file.
    
    Args:
        episode: episode number
        steps: number of steps in the episode
        collisions: number of collisions in the episode
        balls_collected: number of balls collected
        epsilon: current epsilon value
    """
    import csv
    
    log_file = os.path.join(C.LOG_DIR, "training_log.csv")
    
    # Check if file exists to write header
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['episode', 'steps', 'collisions', 'balls_collected', 'epsilon', 'timestamp'])
        
        writer.writerow([
            episode, steps, collisions, balls_collected, epsilon,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])


def load_training_log():
    """
    Load the training log for analysis.
    
    Returns:
        list of dict: Training history
    """
    import csv
    
    log_file = os.path.join(C.LOG_DIR, "training_log.csv")
    
    if not os.path.isfile(log_file):
        return []
    
    history = []
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['episode'] = int(row['episode'])
            row['steps'] = int(row['steps'])
            row['collisions'] = int(row['collisions'])
            row['balls_collected'] = int(row['balls_collected'])
            row['epsilon'] = float(row['epsilon'])
            history.append(row)
    
    return history


def calculate_success_rate(history, window=100):
    """
    Calculate the rolling success rate.
    
    Args:
        history: training history from load_training_log()
        window: number of episodes to average over
    
    Returns:
        list: success rates
    """
    success_rates = []
    
    for i in range(len(history)):
        start = max(0, i - window + 1)
        window_data = history[start:i+1]
        successes = sum(1 for ep in window_data if ep['balls_collected'] == C.MAX_BALLS)
        success_rates.append(successes / len(window_data))
    
    return success_rates


def calculate_average_reward(history, window=100):
    """
    Calculate the rolling average reward.
    Note: Reward data would need to be logged separately.
    """
    # This is a placeholder - you'd need to log rewards separately
    pass


def is_valid_position(pos, obstacles, grid_rows, grid_cols):
    """
    Check if a position is valid (within bounds and not an obstacle).
    
    Args:
        pos: (row, col) tuple
        obstacles: set of obstacle positions
        grid_rows: number of grid rows
        grid_cols: number of grid columns
    
    Returns:
        bool: True if position is valid
    """
    r, c = pos
    return (0 <= r < grid_rows and 
            0 <= c < grid_cols and 
            pos not in obstacles)


def get_random_position(obstacles, grid_rows, grid_cols):
    """
    Get a random valid position.
    
    Args:
        obstacles: set of obstacle positions
        grid_rows: number of grid rows
        grid_cols: number of grid columns
    
    Returns:
        tuple: (row, col) random valid position
    """
    import random
    
    while True:
        pos = (random.randint(0, grid_rows - 1), random.randint(0, grid_cols - 1))
        if pos not in obstacles:
            return pos


def print_training_summary(history):
    """
    Print a summary of training progress.
    
    Args:
        history: training history from load_training_log()
    """
    if not history:
        print("No training data found.")
        return
    
    total_episodes = len(history)
    last_100 = history[-100:] if total_episodes >= 100 else history
    
    success_rate = sum(1 for ep in last_100 if ep['balls_collected'] == C.MAX_BALLS) / len(last_100) * 100
    avg_steps = sum(ep['steps'] for ep in last_100) / len(last_100)
    avg_collisions = sum(ep['collisions'] for ep in last_100) / len(last_100)
    avg_balls = sum(ep['balls_collected'] for ep in last_100) / len(last_100)
    
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total episodes: {total_episodes}")
    print(f"Success rate (last 100): {success_rate:.1f}%")
    print(f"Average steps (last 100): {avg_steps:.1f}")
    print(f"Average collisions (last 100): {avg_collisions:.1f}")
    print(f"Average balls collected (last 100): {avg_balls:.1f}/{C.MAX_BALLS}")
    
    if total_episodes >= 100:
        first_100 = history[:100]
        first_success = sum(1 for ep in first_100 if ep['balls_collected'] == C.MAX_BALLS) / 100 * 100
        print(f"\nImprovement: {first_success:.1f}% → {success_rate:.1f}% success rate")


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test distance functions
    assert manhattan_distance((0, 0), (3, 4)) == 7
    assert euclidean_distance((0, 0), (3, 4)) == 5.0
    
    # Test nearest_ball
    balls = [(1, 1), (5, 5), (2, 3)]
    dist, nearest = nearest_ball((0, 0), balls)
    assert dist == 2  # Manhattan distance to (1,1)
    assert nearest == (1, 1)
    
    print("All tests passed!")