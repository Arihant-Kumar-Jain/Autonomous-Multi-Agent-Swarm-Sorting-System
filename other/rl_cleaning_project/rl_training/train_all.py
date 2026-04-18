import numpy as np
import json
import torch
import os
from env import CleaningEnv
from baseline import GreedyBFSBaseline
from tabular_q import TabularQAgent
from dqn_pytorch import DQNAgent

def get_rewards(info, reward_type):
    cleaned = info['cleaned']
    collisions = info['collisions']
    num_robots = len(cleaned)
    rewards = np.zeros(num_robots)
    
    any_cleaned = np.any(cleaned)
    
    for i in range(num_robots):
        if reward_type == 'R1':
            rewards[i] = 10.0 if cleaned[i] else 0.0
        elif reward_type == 'R2':
            rewards[i] = -0.01
            if cleaned[i]: rewards[i] += 10.0
        elif reward_type == 'R3':
            rewards[i] = -0.01
            if cleaned[i]: rewards[i] += 10.0
            if collisions[i]: rewards[i] -= 5.0
            # Team reward
            if any_cleaned and not cleaned[i]:
                rewards[i] += 2.0
    return rewards

def train_agent(agent_type, reward_type, episodes=200):
    env = CleaningEnv(grid_size=50, num_robots=3, num_dirt=10, max_steps=400)
    
    if agent_type == 'Q':
        agent = TabularQAgent()
    elif agent_type == 'DQN':
        agent = DQNAgent()
        
    history_rewards = []
    history_steps = []
    history_collisions = []
    
    print(f"Training {agent_type} with {reward_type}...")
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        ep_collisions = 0
        
        while not done:
            actions = agent.act(state, explore=True)
            next_state, done, info = env.step(actions)
            
            rewards = get_rewards(info, reward_type)
            agent.learn(state, actions, rewards, next_state, done)
            
            state = next_state
            ep_reward += np.sum(rewards)
            ep_collisions += np.sum(info['collisions']) / 2.0 # avoid double count
            
        if agent_type == 'DQN':
            agent.update_target()
            agent.decay_epsilon()
            
        history_rewards.append(ep_reward)
        history_steps.append(env.steps)
        history_collisions.append(ep_collisions)
        
        if (ep+1) % 50 == 0:
            print(f"  Ep {ep+1}/{episodes} | Steps: {env.steps} | Rew: {ep_reward:.1f} | Col: {ep_collisions}")
            
    # Save models
    os.makedirs('models', exist_ok=True)
    if agent_type == 'Q':
        q_table_serializable = {str(k): list(v) for k, v in agent.q_table.items()}
        with open(f'models/q_table_{reward_type}.json', 'w') as f:
            json.dump(q_table_serializable, f)
    elif agent_type == 'DQN':
        torch.save(agent.q_network.state_dict(), f'models/dqn_{reward_type}.pth')
        
    return history_rewards, history_steps, history_collisions

def eval_baseline(episodes=50):
    env = CleaningEnv(grid_size=50, num_robots=3, num_dirt=10, max_steps=400)
    agent = GreedyBFSBaseline(env)
    
    history_steps = []
    history_collisions = []
    print("Evaluating Baseline...")
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_collisions = 0
        while not done:
            actions = agent.act(state)
            next_state, done, info = env.step(actions)
            state = next_state
            ep_collisions += np.sum(info['collisions']) / 2.0
            
        history_steps.append(env.steps)
        history_collisions.append(ep_collisions)
        if (ep+1) % 10 == 0:
            print(f"  Ep {ep+1}/{episodes} | Steps: {env.steps} | Col: {ep_collisions}")
        
    return np.mean(history_steps), np.mean(history_collisions)

if __name__ == '__main__':
    results = {}
    
    # Eval baseline
    b_steps, b_cols = eval_baseline(30)
    results['Baseline'] = {'steps': b_steps, 'cols': b_cols}
    
    # Train RL
    episodes = 250
    for agent_type in ['Q', 'DQN']:
        for rew_type in ['R1', 'R2', 'R3']:
            rews, steps, cols = train_agent(agent_type, rew_type, episodes)
            results[f'{agent_type}_{rew_type}'] = {
                'rewards': rews,
                'steps': steps,
                'cols': cols
            }
            
    with open('training_results.json', 'w') as f:
        json.dump(results, f)
