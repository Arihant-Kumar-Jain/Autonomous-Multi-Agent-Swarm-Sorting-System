# train_headless.py — Headless training without Pygame rendering
# Run with: python train_headless.py

import sys
import os
import random
import time

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config as C
from env.environment import Environment
from agents.agent import Agent
from utils.utils import log_episode


# ═════════════════════════════════════════════════════════════════════════════
# HEADLESS TRAINING LOOP — No Pygame rendering
# ═════════════════════════════════════════════════════════════════════════════

def train_headless(num_episodes=5000, mode="rl", save_interval=100, 
                   verbose=True, print_frequency=10):
    """
    Train agents without Pygame rendering.
    
    Args:
        num_episodes: Number of episodes to train for
        mode: "rl" or "rule"
        save_interval: Save Q-tables every N episodes
        verbose: Print progress information
        print_frequency: Print stats every N episodes
    """
    env = Environment()
    
    # Instantiate agents
    agents = [Agent(i, mode=mode) for i in range(C.NUM_AGENTS)]
    
    # Try loading saved models
    for ag in agents:
        ag.load()
    
    episode = 1
    total_collisions = 0
    
    # Track training metrics
    episode_rewards = []
    episode_steps = []
    episode_collisions = []
    episode_balls_collected = []
    
    start_time = time.time()
    
    print("=" * 70)
    print(f"HEADLESS TRAINING STARTING")
    print(f"Mode: {mode.upper()}")
    print(f"Episodes: {num_episodes}")
    print(f"Agents: {C.NUM_AGENTS}")
    print(f"Save interval: {save_interval}")
    print("=" * 70)
    
    for episode in range(1, num_episodes + 1):
        # ── Episode start ─────────────────────────
        start_positions = env.reset()
        for i, ag in enumerate(agents):
            ag.reset(start_positions[i])
            ag.mode = mode
            # Initialise distance-to-ball for shaping
            from utils.utils import nearest_ball as _nb
            d, _ = _nb(ag.pos, env.balls)
            ag.prev_dist_to_ball = d if d != float("inf") else C.GRID_ROWS + C.GRID_COLS
        
        episode_done = False
        
        # ── Episode loop ─────────────────────────
        while not episode_done:
            # Collect actions
            proposed_positions = []
            proposed_actions = []
            other_positions = [ag.pos for ag in agents]
            
            for ag in agents:
                others = [p for j, p in enumerate(other_positions) 
                         if j != ag.agent_id]
                pos, action = ag.select_action(env.balls, others, env.obstacles)
                proposed_positions.append(pos)
                proposed_actions.append(action)
            
            # Environment step
            rewards, new_positions, episode_done, info = env.step(proposed_positions)
            total_collisions += info["collisions_step"]
            
            # Update agent positions & learn
            for i, ag in enumerate(agents):
                ag.pos = new_positions[i]
                if info["collected"][i]:
                    ag.balls_collected += 1
                
                others_new = [new_positions[j] for j in range(C.NUM_AGENTS) if j != i]
                ag.learn(rewards[i], env.balls, others_new, episode_done)
        
        # ── Episode end ─────────────────────────
        # Track metrics
        episode_rewards.append(sum(ag.episode_reward for ag in agents))
        episode_steps.append(env.step_count)
        episode_collisions.append(env.collision_count)
        episode_balls_collected.append(env.balls_collected)
        
        # Print progress
        if verbose and episode % print_frequency == 0:
            elapsed = time.time() - start_time
            avg_reward = sum(episode_rewards[-print_frequency:]) / print_frequency
            avg_steps = sum(episode_steps[-print_frequency:]) / print_frequency
            avg_collisions = sum(episode_collisions[-print_frequency:]) / print_frequency
            avg_balls = sum(episode_balls_collected[-print_frequency:]) / print_frequency
            
            print(f"Ep {episode:>6} | "
                  f"Steps: {env.step_count:>4} | "
                  f"Balls: {env.balls_collected:>2}/10 | "
                  f"Collisions: {env.collision_count:>3} | "
                  f"ε: {agents[0].epsilon:.4f} | "
                  f"Q-size: {agents[0].q_table_size:>6} | "
                  f"Avg Reward: {avg_reward:>6.1f}")
            
            # Log to file
            log_episode(episode, env.step_count, env.collision_count,
                       env.balls_collected, agents[0].epsilon)
        
        # Save Q-tables
        if episode % save_interval == 0:
            for ag in agents:
                ag.save()
            if verbose:
                print(f"\n[Save] Q-tables saved at episode {episode}\n")
        
        # End-of-episode bookkeeping
        for ag in agents:
            ag.end_episode()
    
    # ── Training complete ──────────────────────────
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total episodes: {num_episodes}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Average time per episode: {total_time/num_episodes:.2f} seconds")
    print(f"Final epsilon: {agents[0].epsilon:.6f}")
    print(f"Final Q-table size: {agents[0].q_table_size}")
    
    # Final statistics
    success_rate = sum(1 for balls in episode_balls_collected if balls == C.MAX_BALLS) / num_episodes * 100
    print(f"Success rate (collected all balls): {success_rate:.1f}%")
    
    # Save final models
    print("\nSaving final models...")
    for ag in agents:
        ag.save()
    
    return agents, {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'collisions': episode_collisions,
        'balls_collected': episode_balls_collected
    }


def train_continuous(episodes_between_saves=100, mode="rl"):
    """
    Train continuously until interrupted (Ctrl+C).
    
    Args:
        episodes_between_saves: Save Q-tables every N episodes
        mode: "rl" or "rule"
    """
    env = Environment()
    agents = [Agent(i, mode=mode) for i in range(C.NUM_AGENTS)]
    
    for ag in agents:
        ag.load()
    
    episode = 1
    total_collisions = 0
    
    print("=" * 70)
    print(f"CONTINUOUS TRAINING - Press Ctrl+C to stop and save")
    print(f"Mode: {mode.upper()}")
    print("=" * 70)
    
    try:
        while True:
            start_positions = env.reset()
            for i, ag in enumerate(agents):
                ag.reset(start_positions[i])
                ag.mode = mode
                from utils.utils import nearest_ball as _nb
                d, _ = _nb(ag.pos, env.balls)
                ag.prev_dist_to_ball = d if d != float("inf") else C.GRID_ROWS + C.GRID_COLS
            
            episode_done = False
            
            while not episode_done:
                proposed_positions = []
                other_positions = [ag.pos for ag in agents]
                
                for ag in agents:
                    others = [p for j, p in enumerate(other_positions) 
                             if j != ag.agent_id]
                    pos, action = ag.select_action(env.balls, others, env.obstacles)
                    proposed_positions.append(pos)
                
                rewards, new_positions, episode_done, info = env.step(proposed_positions)
                total_collisions += info["collisions_step"]
                
                for i, ag in enumerate(agents):
                    ag.pos = new_positions[i]
                    if info["collected"][i]:
                        ag.balls_collected += 1
                    
                    others_new = [new_positions[j] for j in range(C.NUM_AGENTS) if j != i]
                    ag.learn(rewards[i], env.balls, others_new, episode_done)
            
            # Episode end
            if episode % episodes_between_saves == 0:
                for ag in agents:
                    ag.save()
                print(f"Ep {episode:>6} | Steps: {env.step_count:>4} | "
                      f"Balls: {env.balls_collected:>2}/10 | "
                      f"ε: {agents[0].epsilon:.4f} | Saved")
            elif episode % 10 == 0:
                print(f"Ep {episode:>6} | Steps: {env.step_count:>4} | "
                      f"Balls: {env.balls_collected:>2}/10 | "
                      f"ε: {agents[0].epsilon:.4f}")
            
            log_episode(episode, env.step_count, env.collision_count,
                       env.balls_collected, agents[0].epsilon)
            
            for ag in agents:
                ag.end_episode()
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving models...")
        for ag in agents:
            ag.save()
        print("Models saved. Exiting.")
        
        return agents


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Choose training mode:
    
    # Option 1: Train for a fixed number of episodes
    trained_agents, metrics = train_headless(
        num_episodes=5000,      # Number of episodes
        mode="rl",              # "rl" or "rule"
        save_interval=100,      # Save every 100 episodes
        verbose=True,           # Print progress
        print_frequency=10      # Print every 10 episodes
    )
    
    # Option 2: Train continuously until Ctrl+C
    # trained_agents = train_continuous(
    #     episodes_between_saves=100,
    #     mode="rl"
    # )