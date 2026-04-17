import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    results_dir = '/home/aman/cs671_7/rl_eval_ws/results'
    algos = ['bfs', 'q', 'dqn']
    
    elapsed_times = []
    collisions = []
    valid_algos = []
    
    for algo in algos:
        filepath = os.path.join(results_dir, f'metrics_{algo}.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                elapsed_times.append(data.get('elapsed', 0))
                collisions.append(data.get('collisions', 0))
                valid_algos.append(algo.upper())
                
                # Plot Trajectory & Heatmap per algorithm
                plot_spatial(algo, data)
                
    if not valid_algos:
        print("No metrics JSON files found to plot. Run the evaluations first!")
        return
        
    # Plot 1: Efficiency (Makespan)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(valid_algos, elapsed_times, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(valid_algos)])
    plt.ylabel('Makespan / Time (seconds)')
    plt.title('Efficiency Comparison (Lower is Better)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}s', va='bottom', ha='center')
    plt.savefig(os.path.join(results_dir, 'efficiency_comparison.png'))
    plt.close()
    
    # Plot 2: Safety (Collisions)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(valid_algos, collisions, color=['#d62728', '#9467bd', '#8c564b'][:len(valid_algos)])
    plt.ylabel('Total Virtual/Physical Collisions')
    plt.title('Safety Comparison (Lower is Better)')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
    plt.savefig(os.path.join(results_dir, 'safety_comparison.png'))
    plt.close()

    print("Plots successfully generated in results folder.")

def draw_map_background():
    """Draws the obstacles onto the plot"""
    # Central pillar
    plt.gca().add_patch(plt.Rectangle((-0.75, -0.75), 1.5, 1.5, color='gray', alpha=0.5))
    # Corner wall 1
    plt.gca().add_patch(plt.Rectangle((1.0, 1.0), 1.0, 1.0, color='gray', alpha=0.5))
    # Corner wall 2
    plt.gca().add_patch(plt.Rectangle((-2.0, -2.0), 1.0, 1.0, color='gray', alpha=0.5))

def plot_spatial(algo, data):
    results_dir = '/home/aman/cs671_7/rl_eval_ws/results'
    trajs = data.get('trajectories', {})
    col_pts = data.get('collision_points', [])
    
    # Trajectory Plot
    plt.figure(figsize=(6, 6))
    draw_map_background()
    colors = ['r', 'g', 'b']
    for idx, (rob, pts) in enumerate(trajs.items()):
        if len(pts) > 0:
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            plt.plot(xs, ys, label=rob, color=colors[idx % len(colors)], linewidth=2)
            plt.scatter(xs[0], ys[0], marker='o', color=colors[idx % len(colors)]) # start
            plt.scatter(xs[-1], ys[-1], marker='x', s=100, color=colors[idx % len(colors)]) # end
            
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(f'Trajectory Map: {algo.upper()}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(results_dir, f'trajectory_{algo}.png'))
    plt.close()
    
    # Heatmap Plot (Scatter/Density)
    plt.figure(figsize=(6, 6))
    draw_map_background()
    if col_pts:
        cx = [p[0] for p in col_pts]
        cy = [p[1] for p in col_pts]
        # Use a 2D hexbin or scatter for heatmap
        plt.hexbin(cx, cy, gridsize=15, cmap='Reds', extent=[-3, 3, -3, 3], mincnt=1, alpha=0.8)
        plt.colorbar(label='Collision Count')
        plt.scatter(cx, cy, color='black', s=10, alpha=0.5) # Exact points
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.title(f'Collision Heatmap: {algo.upper()}')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(results_dir, f'heatmap_{algo}.png'))
    plt.close()

if __name__ == '__main__':
    generate_plots()
