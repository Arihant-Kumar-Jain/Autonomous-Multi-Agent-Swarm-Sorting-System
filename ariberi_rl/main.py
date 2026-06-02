"""
Main simulation runner — runs BFS / DQN / PPO with visualization.

Usage:
    python main.py --mode bfs
    python main.py --mode rl --model checkpoints/rl_best.pt
    python main.py --mode improved_rl --model checkpoints/improved_rl_best.pt
    python main.py --mode ppo --model checkpoints/ppo_best.pt
    python main.py --mode improved_ppo --model checkpoints/improved_ppo_best.pt
    python main.py --mode compare --runs 10
"""

import argparse
import json
import os
import numpy as np

import config as cfg
from warehouse_env import WarehouseEnv


def run_simulation(mode="bfs", model_path=None, visualize=True,
                   inject_failure=False, max_steps=cfg.MAX_STEPS_PER_EPISODE):
    """Run one full simulation episode."""
    use_congestion = ("improved" in mode)
    env = WarehouseEnv(use_congestion=use_congestion, mode=mode)

    # Load agent if needed
    agent = None
    agent_type = None

    if mode in ("rl", "improved_rl"):
        from dqn_agent import DQNAgent
        state_size = env.get_state_size()
        agent = DQNAgent(state_size=state_size)
        agent_type = "dqn"
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"  Loaded DQN model: {model_path}")
        else:
            print(f"  ⚠ No model at {model_path}")

    elif mode in ("ppo", "improved_ppo"):
        from ppo_agent import PPOAgent
        state_size = env.get_state_size()
        agent = PPOAgent(state_size=state_size)
        agent_type = "ppo"
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"  Loaded PPO model: {model_path}")
        else:
            print(f"  ⚠ No model at {model_path}")

    elif mode == "mappo":
        from mappo_agent import MAPPOAgent
        state_size = env.get_state_size()
        agent = MAPPOAgent(state_size=state_size, num_agents=cfg.NUM_ROBOTS)
        agent_type = "mappo"
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            print(f"  Loaded MAPPO model: {model_path}")
        else:
            print(f"  ⚠ No model at {model_path}")

    # Visualizer
    viz = None
    if visualize:
        try:
            from visualizer import WarehouseVisualizer
            label = mode.upper().replace("_", " ")
            viz = WarehouseVisualizer(env, mode_label=label)
        except Exception as e:
            print(f"  ⚠ Cannot init visualizer: {e}")
            visualize = False

    # Run
    observations = env.reset()
    env.allocate_tasks()
    if mode == "bfs":
        env.plan_bfs_paths()

    done = False
    step = 0

    print(f"\n  Running: {mode.upper()}")
    print(f"  {'─' * 40}")

    while not done and step < max_steps:
        step += 1

        actions = []
        for rid in range(cfg.NUM_ROBOTS):
            if env.robot_failed[rid] or env.robot_done[rid]:
                actions.append(4)
                continue

            if mode == "bfs":
                action = env.get_bfs_action(rid)
            elif agent_type == "dqn":
                obs = observations[rid]
                action = agent.select_action(obs, training=False)
            elif agent_type == "ppo":
                obs = observations[rid]
                action, _, _ = agent.select_action(obs, training=False)
            elif agent_type == "mappo":
                obs = observations[rid]
                action, _ = agent.select_action(obs, training=False)
            else:
                action = 4
            actions.append(action)

        observations, rewards, done, info = env.step(actions)

        if step % 5 == 0:
            env.allocate_tasks()
            if mode == "bfs":
                env.plan_bfs_paths()

        # Failure injection
        if inject_failure and step == 30:
            print(f"  ⚡ Injecting failure on Robot 1 at step {step}")
            env.simulate_failure(1)
            env.allocate_tasks()
            if mode == "bfs":
                env.plan_bfs_paths()
        if inject_failure and step == 50:
            print(f"  🔧 Recovering Robot 1 at step {step}")
            env.recover_robot(1)
            env.allocate_tasks()
            if mode == "bfs":
                env.plan_bfs_paths()

        if visualize and viz:
            if not viz.handle_events():
                break
            viz.draw(metrics=env.get_metrics())

    metrics = env.get_metrics()

    print(f"\n  Results ({mode.upper()}):")
    print(f"  Steps:      {metrics['steps']}")
    print(f"  Collisions: {metrics['collisions']}")
    print(f"  Pickups:    {metrics['pickups']}")
    print(f"  Deliveries: {metrics['deliveries']}")
    print(f"  Completion: {metrics['completion']:.0%}")

    if visualize and viz:
        print("\n  Press Q or close window to exit.")
        holding = True
        while holding:
            if not viz.handle_events():
                holding = False
            viz.draw(metrics=metrics)
        viz.close()

    return metrics, env.history


def run_comparison(visualize=False, num_runs=10):
    """Run all modes and compare."""
    modes = ["bfs", "rl", "improved_rl", "ppo", "improved_ppo", "mappo"]
    model_paths = {
        "rl": "checkpoints/rl_best.pt",
        "improved_rl": "checkpoints/improved_rl_best.pt",
        "ppo": "checkpoints/ppo_best.pt",
        "improved_ppo": "checkpoints/improved_ppo_best.pt",
        "mappo": "checkpoints/mappo_best.pt",
    }

    results = {m: [] for m in modes}

    print(f"\n{'='*65}")
    print(f"  MULTI-AGENT WAREHOUSE — FULL ALGORITHM COMPARISON")
    print(f"  {num_runs} runs per mode | 6 methods")
    print(f"{'='*65}")

    for mode in modes:
        mp = model_paths.get(mode)
        if mode != "bfs" and mp and not os.path.exists(mp):
            print(f"\n  ⚠ Skipping {mode} — no model at {mp}")
            continue

        print(f"\n  ── {mode.upper()} ──")
        for run in range(num_runs):
            metrics, _ = run_simulation(
                mode=mode,
                model_path=mp,
                visualize=False,
                max_steps=cfg.MAX_STEPS_PER_EPISODE,
            )
            results[mode].append(metrics)

    # Print comparison table
    print(f"\n{'='*72}")
    print(f"  {'Method':<18} {'Steps':>8} {'Collisions':>12} {'Completion':>12} {'Deliveries':>12}")
    print(f"  {'─'*62}")

    for mode in modes:
        if not results[mode]:
            continue
        avg_steps = np.mean([r["steps"] for r in results[mode]])
        avg_coll = np.mean([r["collisions"] for r in results[mode]])
        avg_comp = np.mean([r["completion"] for r in results[mode]])
        avg_del = np.mean([r["deliveries"] for r in results[mode]])
        print(f"  {mode.upper():<18} {avg_steps:>8.1f} {avg_coll:>12.1f} "
              f"{avg_comp:>11.0%} {avg_del:>12.1f}")

    print(f"{'='*72}")

    os.makedirs("results", exist_ok=True)
    with open("results/comparison.json", "w") as f:
        json.dump({m: results[m] for m in modes if results[m]}, f, indent=2)
    print(f"\n  Results saved to results/comparison.json")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Warehouse Simulation")
    parser.add_argument("--mode",
                        choices=["bfs", "rl", "improved_rl", "ppo", "improved_ppo", "mappo", "compare"],
                        default="bfs")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--failure", action="store_true")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(num_runs=args.runs)
    else:
        model = args.model
        if model is None and args.mode not in ("bfs",):
            model = f"checkpoints/{args.mode}_best.pt"
        run_simulation(
            mode=args.mode,
            model_path=model,
            visualize=not args.no_viz,
            inject_failure=args.failure,
        )
