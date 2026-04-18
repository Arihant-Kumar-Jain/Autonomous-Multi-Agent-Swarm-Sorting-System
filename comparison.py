#!/usr/bin/env python3
"""
Experiment Results Comparison and Analysis Tool
Combines all experiment outputs and generates comparison plots
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class ResultsAnalyzer:
    """Analyze and compare experiment results"""
    
    def __init__(self, experiments_root: str = "./experiments"):
        self.experiments_root = experiments_root
        self.results = {}
    
    def load_experiment_results(self):
        """Load training_scores.json from all experiments"""
        experiments_path = Path(self.experiments_root)
        
        if not experiments_path.exists():
            print(f"❌ Experiments directory not found: {self.experiments_root}")
            return
        
        for exp_dir in experiments_path.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Find latest run
            latest_run = None
            latest_time = None
            
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                scores_file = run_dir / "training_scores.json"
                if scores_file.exists():
                    run_time = run_dir.stat().st_mtime
                    if latest_time is None or run_time > latest_time:
                        latest_time = run_time
                        latest_run = run_dir
            
            if latest_run:
                scores_file = latest_run / "training_scores.json"
                try:
                    with open(scores_file, 'r') as f:
                        data = json.load(f)
                    self.results[exp_dir.name] = {
                        'path': str(latest_run),
                        'data': data,
                        'timestamp': data.get('timestamp', 'unknown'),
                        'status': data.get('status', 'unknown')
                    }
                    print(f"✅ Loaded: {exp_dir.name} ({data['num_episodes']} episodes)")
                except Exception as e:
                    print(f"⚠️  Failed to load {exp_dir.name}: {e}")
    
    def print_summary_table(self):
        """Print comparison table"""
        if not self.results:
            print("❌ No results to compare")
            return
        
        print("\n" + "="*80)
        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        
        print(f"{'Experiment':<25} {'Episodes':<12} {'Avg Score':<15} {'Best Score':<15} {'Status':<15}")
        print("-"*80)
        
        for exp_name in sorted(self.results.keys()):
            result = self.results[exp_name]
            data = result['data']
            print(f"{exp_name:<25} {data.get('num_episodes', 0):<12} "
                  f"{data.get('current_avg_score', 0):<15.2f} "
                  f"{data.get('best_score', 0):<15.2f} "
                  f"{data.get('status', 'unknown'):<15}")
        
        print("="*80)
    
    def generate_learning_curves(self, save_path: str = "comparison_learning_curves.png"):
        """Generate learning curves for all experiments"""
        if not self.results:
            print("❌ No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-Robot RL Experiments - Learning Curves', fontsize=16, fontweight='bold')
        
        # 1. Episode scores over time
        ax = axes[0, 0]
        for exp_name, result in self.results.items():
            scores = result['data'].get('episode_scores', [])
            if scores:
                ax.plot(scores, label=exp_name, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Episode Scores Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Smoothed learning curves (100-episode moving average)
        ax = axes[0, 1]
        for exp_name, result in self.results.items():
            scores = np.array(result['data'].get('episode_scores', []))
            if len(scores) > 100:
                smoothed = np.convolve(scores, np.ones(100)/100, mode='valid')
                ax.plot(smoothed, label=exp_name, linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Smoothed Score (100-ep MA)')
        ax.set_title('Smoothed Learning Curves (100-episode Moving Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Final scores comparison
        ax = axes[1, 0]
        exp_names = list(self.results.keys())
        final_scores = [self.results[exp]['data'].get('best_score', 0) for exp in exp_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
        bars = ax.bar(exp_names, final_scores, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Best Score')
        ax.set_title('Final Scores Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, final_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Convergence speed (episodes to reach 80% of best score)
        ax = axes[1, 1]
        convergence_episodes = []
        for exp_name, result in self.results.items():
            scores = np.array(result['data'].get('episode_scores', []))
            best = np.max(scores)
            threshold = best * 0.8
            
            # Find first time score exceeds threshold
            converged_at = np.where(scores >= threshold)[0]
            if len(converged_at) > 0:
                convergence_episodes.append(converged_at[0])
            else:
                convergence_episodes.append(len(scores))
        
        bars = ax.bar(exp_names, convergence_episodes, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Episodes')
        ax.set_title('Convergence Speed (Episodes to 80% of Best)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, ep in zip(bars, convergence_episodes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(ep)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Learning curves saved to: {save_path}")
        plt.close()
    
    def generate_comparison_table(self, save_path: str = "comparison_results.json"):
        """Generate detailed comparison JSON"""
        comparison = {
            'generated_at': datetime.now().isoformat(),
            'experiments': {}
        }
        
        for exp_name, result in self.results.items():
            data = result['data']
            scores = np.array(data.get('episode_scores', []))
            
            # Calculate metrics
            final_score = data.get('best_score', 0)
            avg_score = data.get('current_avg_score', 0)
            num_episodes = data.get('num_episodes', 0)
            
            # Convergence speed
            threshold = final_score * 0.8
            convergence = np.where(scores >= threshold)[0]
            convergence_ep = int(convergence[0]) if len(convergence) > 0 else num_episodes
            
            # Variance and improvement rate
            if len(scores) > 100:
                early_avg = np.mean(scores[:100])
                late_avg = np.mean(scores[-100:])
                improvement_rate = (late_avg - early_avg) / max(abs(early_avg), 1.0)
            else:
                improvement_rate = 0
            
            comparison['experiments'][exp_name] = {
                'final_score': float(final_score),
                'average_score': float(avg_score),
                'num_episodes': int(num_episodes),
                'convergence_episodes': int(convergence_ep),
                'improvement_rate': float(improvement_rate),
                'status': data.get('status', 'unknown'),
                'path': result['path'],
                'last_updated': result['timestamp']
            }
        
        # Add rankings
        sorted_exp = sorted(comparison['experiments'].items(),
                           key=lambda x: x[1]['final_score'], reverse=True)
        
        print("\n🏆 RANKINGS (by Final Score)")
        print("-" * 60)
        for rank, (exp_name, metrics) in enumerate(sorted_exp, 1):
            print(f"{rank}. {exp_name:<25} Score: {metrics['final_score']:.1f}")
        
        # Save
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n✅ Detailed comparison saved to: {save_path}")
    
    def generate_report(self):
        """Generate full analysis report"""
        print("\n" + "🔬 GENERATING COMPREHENSIVE ANALYSIS REPORT...")
        
        self.load_experiment_results()
        self.print_summary_table()
        self.generate_learning_curves()
        self.generate_comparison_table()
        
        print("\n✅ Analysis complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--root', default='./experiments', help='Experiments root directory')
    parser.add_argument('--output', default='comparison_results.json', help='Output file')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(experiments_root=args.root)
    analyzer.generate_report()
