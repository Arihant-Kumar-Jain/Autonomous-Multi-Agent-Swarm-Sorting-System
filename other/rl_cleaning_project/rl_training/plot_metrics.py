import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    with open('training_results.json', 'r') as f:
        results = json.load(f)
        
    baseline_steps = results['Baseline']['steps']
    baseline_cols = results['Baseline']['cols']
    
    # 1. Training Convergence (Rewards)
    plt.figure(figsize=(10, 6))
    for agent_type in ['Q', 'DQN']:
        for rew_type in ['R1', 'R2', 'R3']:
            key = f"{agent_type}_{rew_type}"
            rews = smooth(results[key]['rewards'], 20)
            plt.plot(rews, label=key)
    plt.title('Training Convergence (Smoothed Rewards)')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_convergence.png')
    plt.close()
    
    # 2. Efficiency Comparison (Steps)
    plt.figure(figsize=(10, 6))
    plt.axhline(y=baseline_steps, color='r', linestyle='--', label='Baseline BFS')
    for agent_type in ['Q', 'DQN']:
        for rew_type in ['R3']: # Just plot R3 for clarity against baseline
            key = f"{agent_type}_{rew_type}"
            steps = smooth(results[key]['steps'], 20)
            plt.plot(steps, label=key)
    plt.title('Efficiency: Steps to Clean All Dirt')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficiency_comparison.png')
    plt.close()
    
    # 3. Safety Comparison (Collisions)
    plt.figure(figsize=(10, 6))
    plt.axhline(y=baseline_cols, color='r', linestyle='--', label='Baseline BFS')
    for rew_type in ['R1', 'R3']:
        key = f"DQN_{rew_type}"
        cols = smooth(results[key]['cols'], 20)
        plt.plot(cols, label=key)
    plt.title('Safety: Total Collisions (DQN R1 vs R3)')
    plt.xlabel('Episode')
    plt.ylabel('Collisions')
    plt.legend()
    plt.grid(True)
    plt.savefig('safety_comparison.png')
    plt.close()
    
    # 4. Final Evaluation Bar Chart
    labels = ['Baseline', 'Q_R1', 'Q_R2', 'Q_R3', 'DQN_R1', 'DQN_R2', 'DQN_R3']
    avg_steps = [baseline_steps]
    avg_cols = [baseline_cols]
    
    for agent_type in ['Q', 'DQN']:
        for rew_type in ['R1', 'R2', 'R3']:
            key = f"{agent_type}_{rew_type}"
            avg_steps.append(np.mean(results[key]['steps'][-30:]))
            avg_cols.append(np.mean(results[key]['cols'][-30:]))
            
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    rects1 = ax1.bar(x - width/2, avg_steps, width, label='Avg Steps', color='b', alpha=0.6)
    rects2 = ax2.bar(x + width/2, avg_cols, width, label='Avg Collisions', color='r', alpha=0.6)
    
    ax1.set_ylabel('Steps', color='b')
    ax2.set_ylabel('Collisions', color='r')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Final Evaluation Metrics')
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig('final_evaluation_bar.png')
    plt.close()
    print("Plots saved successfully.")

if __name__ == '__main__':
    main()
