"""
RL Trainer — trains DQN or PPO agents in the grid environment.

Usage:
    python trainer.py --mode rl              # DQN baseline
    python trainer.py --mode improved_rl     # DQN + congestion
    python trainer.py --mode ppo             # PPO
    python trainer.py --mode improved_ppo    # PPO + congestion (best)
    python trainer.py --mode all             # Train all variants
"""

import argparse
import os
import json
import shutil
import numpy as np
from datetime import datetime
from collections import deque

import config as cfg


def rotate_logs(variant_dir):
    """Archive existing logs/models before a new training run."""
    log_file = os.path.join(variant_dir, "training_log.json")
    if not os.path.exists(log_file):
        return  # nothing to rotate
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_dir = os.path.join(variant_dir, "history", ts)
    os.makedirs(history_dir, exist_ok=True)
    for f in os.listdir(variant_dir):
        fpath = os.path.join(variant_dir, f)
        if os.path.isfile(fpath):
            shutil.move(fpath, os.path.join(history_dir, f))
    print(f"  📦 Archived previous run to {history_dir}")
from warehouse_env import WarehouseEnv


def train_dqn(mode="rl", episodes=cfg.TRAIN_EPISODES, save_dir="checkpoints"):
    """Train DQN agents."""
    from dqn_agent import DQNAgent

    use_congestion = ("improved" in mode)
    env = WarehouseEnv(use_congestion=use_congestion, mode=mode)
    state_size = env.get_state_size()
    agent = DQNAgent(state_size=state_size)

    os.makedirs(save_dir, exist_ok=True)
    variant_dir = os.path.join(save_dir, mode)
    os.makedirs(variant_dir, exist_ok=True)
    rotate_logs(variant_dir)
    log_path = os.path.join(variant_dir, "training_log.json")

    episode_rewards = deque(maxlen=100)
    episode_completions = deque(maxlen=100)
    episode_collisions = deque(maxlen=100)
    logs = []

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  Training DQN: {mode.upper():>15}              ║")
    print(f"║  State: {state_size}  Episodes: {episodes:>5}  Device: {agent.device}  ║")
    print(f"╚══════════════════════════════════════════════╝")

    best_completion = 0.0

    for ep in range(1, episodes + 1):
        observations = env.reset()
        env.allocate_tasks()
        total_reward = 0.0
        done = False

        for step in range(cfg.MAX_STEPS_PER_EPISODE):
            actions = []
            old_obs = []

            for rid in range(cfg.NUM_ROBOTS):
                if env.robot_failed[rid] or env.robot_done[rid]:
                    actions.append(4)
                    old_obs.append(observations[rid])
                    continue
                obs = observations[rid]
                action = agent.select_action(obs, training=True)
                actions.append(action)
                old_obs.append(obs)

            observations, rewards, done, info = env.step(actions)

            if step % 10 == 0:
                env.allocate_tasks()

            for rid in range(cfg.NUM_ROBOTS):
                if not env.robot_failed[rid]:
                    agent.store_transition(
                        old_obs[rid], actions[rid], rewards[rid],
                        observations[rid], float(done))
                    total_reward += rewards[rid]

            agent.train_step()

            if done:
                break

            # Failure injection (10% at step 50)
            if step == 50 and np.random.random() < 0.1:
                env.simulate_failure(np.random.randint(0, cfg.NUM_ROBOTS))
            if step == 70:
                for rid in range(cfg.NUM_ROBOTS):
                    if env.robot_failed[rid]:
                        env.recover_robot(rid)
                        env.allocate_tasks()

        if ep % cfg.TARGET_UPDATE == 0:
            agent.update_target()

        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_completions.append(metrics["completion"])
        episode_collisions.append(metrics["collisions"])

        avg_reward = np.mean(episode_rewards)
        avg_completion = np.mean(episode_completions)
        avg_collisions = np.mean(episode_collisions)

        logs.append({
            "episode": ep, "reward": total_reward, "avg_reward": avg_reward,
            "completion": metrics["completion"], "avg_completion": avg_completion,
            "collisions": metrics["collisions"], "avg_collisions": avg_collisions,
            "steps": metrics["steps"], "epsilon": agent.get_epsilon(),
        })

        if ep % 50 == 0 or ep == 1:
            print(f"  Ep {ep:>5}/{episodes}  |  Reward: {avg_reward:>7.1f}  |  "
                  f"Completion: {avg_completion:.0%}  |  Collisions: {avg_collisions:.1f}  |  "
                  f"ε: {agent.get_epsilon():.3f}")

        if avg_completion > best_completion and ep > 100:
            best_completion = avg_completion
            agent.save(os.path.join(variant_dir, "best.pt"))

    agent.save(os.path.join(variant_dir, "latest.pt"))
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n  ✅ DQN training complete! Best: {best_completion:.0%}")
    return logs


def train_ppo(mode="ppo", episodes=cfg.PPO_EPISODES, save_dir="checkpoints"):
    """Train PPO agents."""
    from ppo_agent import PPOAgent

    use_congestion = ("improved" in mode)
    env = WarehouseEnv(use_congestion=use_congestion, mode=mode)
    state_size = env.get_state_size()

    agent = PPOAgent(
        state_size=state_size,
        action_size=cfg.NUM_ACTIONS,
        lr=cfg.PPO_LR,
        gamma=cfg.GAMMA,
        gae_lambda=cfg.GAE_LAMBDA,
        clip_eps=cfg.CLIP_EPS,
        entropy_coef=cfg.ENTROPY_COEF,
        value_coef=cfg.VALUE_COEF,
        ppo_epochs=cfg.PPO_EPOCHS,
        batch_size=cfg.PPO_BATCH_SIZE,
        rollout_length=cfg.ROLLOUT_LENGTH,
    )

    os.makedirs(save_dir, exist_ok=True)
    variant_dir = os.path.join(save_dir, mode)
    os.makedirs(variant_dir, exist_ok=True)
    rotate_logs(variant_dir)
    log_path = os.path.join(variant_dir, "training_log.json")

    episode_rewards = deque(maxlen=100)
    episode_completions = deque(maxlen=100)
    episode_collisions = deque(maxlen=100)
    logs = []

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Training PPO: {mode.upper():>15}                     ║")
    print(f"║  State: {state_size}  Episodes: {episodes:>5}  Device: {agent.device}    ║")
    print(f"║  Rollout: {cfg.ROLLOUT_LENGTH}  Batch: {cfg.PPO_BATCH_SIZE}  Epochs: {cfg.PPO_EPOCHS}          ║")
    print(f"╚══════════════════════════════════════════════════════╝")

    best_completion = 0.0
    global_step = 0

    for ep in range(1, episodes + 1):
        observations = env.reset()
        env.allocate_tasks()
        total_reward = 0.0
        done = False
        ep_update_info = None

        for step in range(cfg.MAX_STEPS_PER_EPISODE):
            actions = []
            action_data = []  # (action, log_prob, value) per robot

            for rid in range(cfg.NUM_ROBOTS):
                if env.robot_failed[rid] or env.robot_done[rid]:
                    actions.append(4)
                    action_data.append((4, 0.0, 0.0))
                    continue

                obs = observations[rid]
                action, log_prob, value = agent.select_action(obs, training=True)
                actions.append(action)
                action_data.append((action, log_prob, value))

            old_obs = [obs.copy() for obs in observations]
            observations, rewards, done, info = env.step(actions)

            if step % 10 == 0:
                env.allocate_tasks()

            # Store transitions for all active robots
            for rid in range(cfg.NUM_ROBOTS):
                if not env.robot_failed[rid]:
                    a, lp, v = action_data[rid]
                    agent.store_transition(
                        old_obs[rid], a, lp, rewards[rid], v, float(done))
                    total_reward += rewards[rid]
                    global_step += 1

            # PPO update when buffer is full
            if agent.should_update():
                # Use first active robot's observation as last_obs
                last_obs = observations[0]
                ep_update_info = agent.update(last_obs)

            if done:
                break

            # Failure injection
            if step == 50 and np.random.random() < 0.1:
                env.simulate_failure(np.random.randint(0, cfg.NUM_ROBOTS))
            if step == 70:
                for rid in range(cfg.NUM_ROBOTS):
                    if env.robot_failed[rid]:
                        env.recover_robot(rid)
                        env.allocate_tasks()

        # Flush remaining buffer
        if len(agent.buffer) > cfg.PPO_BATCH_SIZE:
            last_obs = observations[0]
            ep_update_info = agent.update(last_obs)

        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_completions.append(metrics["completion"])
        episode_collisions.append(metrics["collisions"])

        avg_reward = np.mean(episode_rewards)
        avg_completion = np.mean(episode_completions)
        avg_collisions = np.mean(episode_collisions)

        log_entry = {
            "episode": ep, "reward": total_reward, "avg_reward": avg_reward,
            "completion": metrics["completion"], "avg_completion": avg_completion,
            "collisions": metrics["collisions"], "avg_collisions": avg_collisions,
            "steps": metrics["steps"], "global_step": global_step,
        }
        if ep_update_info:
            log_entry.update({
                "pg_loss": ep_update_info["pg_loss"],
                "vf_loss": ep_update_info["vf_loss"],
                "entropy": ep_update_info["entropy"],
                "lr": ep_update_info["lr"],
            })
        logs.append(log_entry)

        if ep % 50 == 0 or ep == 1:
            ent = ep_update_info["entropy"] if ep_update_info else 0.0
            print(f"  Ep {ep:>5}/{episodes}  |  Reward: {avg_reward:>7.1f}  |  "
                  f"Completion: {avg_completion:.0%}  |  Collisions: {avg_collisions:.1f}  |  "
                  f"Entropy: {ent:.3f}")

        if avg_completion > best_completion and ep > 50:
            best_completion = avg_completion
            agent.save(os.path.join(variant_dir, "best.pt"))

    agent.save(os.path.join(variant_dir, "latest.pt"))
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n  ✅ PPO training complete! Best: {best_completion:.0%}")
    return logs


def train_mappo(mode="mappo", episodes=cfg.MAPPO_EPISODES, save_dir="checkpoints"):
    """Train MAPPO agent (centralized critic, decentralized actors)."""
    from mappo_agent import MAPPOAgent

    use_congestion = True  # MAPPO always uses congestion-aware state
    env = WarehouseEnv(use_congestion=use_congestion, mode=mode)
    state_size = env.get_state_size()

    agent = MAPPOAgent(
        state_size=state_size,
        action_size=cfg.NUM_ACTIONS,
        num_agents=cfg.NUM_ROBOTS,
        lr_actor=cfg.MAPPO_LR_ACTOR,
        lr_critic=cfg.MAPPO_LR_CRITIC,
        gamma=cfg.GAMMA,
        gae_lambda=cfg.GAE_LAMBDA,
        clip_eps=cfg.CLIP_EPS,
        entropy_coef=cfg.ENTROPY_COEF,
        value_coef=cfg.VALUE_COEF,
        ppo_epochs=cfg.MAPPO_EPOCHS,
        batch_size=cfg.MAPPO_BATCH_SIZE,
        rollout_length=cfg.MAPPO_ROLLOUT_LENGTH,
    )

    os.makedirs(save_dir, exist_ok=True)
    variant_dir = os.path.join(save_dir, mode)
    os.makedirs(variant_dir, exist_ok=True)
    rotate_logs(variant_dir)
    log_path = os.path.join(variant_dir, "training_log.json")

    episode_rewards = deque(maxlen=100)
    episode_completions = deque(maxlen=100)
    episode_collisions = deque(maxlen=100)
    logs = []

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Training MAPPO (CTDE)                               ║")
    print(f"║  State: {state_size}  Global: {state_size * cfg.NUM_ROBOTS}  Episodes: {episodes:>5}           ║")
    print(f"║  Device: {agent.device}                                       ║")
    print(f"╚══════════════════════════════════════════════════════╝")

    best_completion = 0.0

    for ep in range(1, episodes + 1):
        observations = env.reset()
        env.allocate_tasks()
        total_reward = 0.0
        done = False
        ep_update_info = None

        for step in range(cfg.MAX_STEPS_PER_EPISODE):
            actions = []
            action_data = []  # (action, log_prob) per robot

            for rid in range(cfg.NUM_ROBOTS):
                if env.robot_failed[rid] or env.robot_done[rid]:
                    actions.append(4)
                    action_data.append((4, 0.0))
                    continue
                obs = observations[rid]
                action, log_prob = agent.select_action(obs, training=True)
                actions.append(action)
                action_data.append((action, log_prob))

            # Compute global value (centralized critic)
            global_state = np.concatenate(observations)
            global_value = agent.get_global_value(observations)

            old_obs = [obs.copy() for obs in observations]
            observations, rewards, done, info = env.step(actions)

            # Mixed rewards: α * individual + (1-α) * team
            team_reward = sum(rewards) / cfg.NUM_ROBOTS
            alpha = cfg.MIXED_REWARD_ALPHA
            rewards = [alpha * r + (1 - alpha) * team_reward for r in rewards]

            if step % 10 == 0:
                env.allocate_tasks()

            # Store transitions for all active robots
            for rid in range(cfg.NUM_ROBOTS):
                if not env.robot_failed[rid]:
                    a, lp = action_data[rid]
                    agent.store_transition(
                        rid, old_obs[rid], a, lp, rewards[rid], float(done),
                        global_state=global_state if rid == 0 else None,
                        global_value=global_value if rid == 0 else None)
                    total_reward += rewards[rid]

            # MAPPO update when buffer is full
            if agent.should_update():
                ep_update_info = agent.update(observations)

            if done:
                break

            # Failure injection
            if step == 50 and np.random.random() < 0.1:
                env.simulate_failure(np.random.randint(0, cfg.NUM_ROBOTS))
            if step == 70:
                for rid in range(cfg.NUM_ROBOTS):
                    if env.robot_failed[rid]:
                        env.recover_robot(rid)
                        env.allocate_tasks()

        # Flush remaining buffer
        if len(agent.buffer) > cfg.MAPPO_BATCH_SIZE:
            ep_update_info = agent.update(observations)

        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_completions.append(metrics["completion"])
        episode_collisions.append(metrics["collisions"])

        avg_reward = np.mean(episode_rewards)
        avg_completion = np.mean(episode_completions)
        avg_collisions = np.mean(episode_collisions)

        log_entry = {
            "episode": ep, "reward": total_reward, "avg_reward": avg_reward,
            "completion": metrics["completion"], "avg_completion": avg_completion,
            "collisions": metrics["collisions"], "avg_collisions": avg_collisions,
            "steps": metrics["steps"],
        }
        if ep_update_info:
            log_entry.update({
                "pg_loss": ep_update_info["pg_loss"],
                "vf_loss": ep_update_info["vf_loss"],
                "entropy": ep_update_info["entropy"],
                "lr": ep_update_info["lr_actor"],
            })
        logs.append(log_entry)

        if ep % 50 == 0 or ep == 1:
            ent = ep_update_info["entropy"] if ep_update_info else 0.0
            print(f"  Ep {ep:>5}/{episodes}  |  Reward: {avg_reward:>7.1f}  |  "
                  f"Completion: {avg_completion:.0%}  |  Collisions: {avg_collisions:.1f}  |  "
                  f"Entropy: {ent:.3f}")

        if avg_completion > best_completion and ep > 50:
            best_completion = avg_completion
            agent.save(os.path.join(variant_dir, "best.pt"))

    agent.save(os.path.join(variant_dir, "latest.pt"))
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n  ✅ MAPPO training complete! Best: {best_completion:.0%}")
    return logs


def train_mappo_continuous(mode="mappo_continuous", episodes=cfg.MAPPO_EPISODES,
                           save_dir="checkpoints"):
    """Train MAPPO with continuous actions + pseudo-physics 2D world."""
    from mappo_continuous_agent import MAPPOContinuousAgent
    from warehouse_continuous_env import ContinuousWarehouseEnv

    env = ContinuousWarehouseEnv(use_congestion=True, mode=mode)
    state_size = env.get_state_size()

    agent = MAPPOContinuousAgent(
        state_size=state_size,
        num_agents=cfg.NUM_ROBOTS,
        lr_actor=cfg.MAPPO_LR_ACTOR,
        lr_critic=cfg.MAPPO_LR_CRITIC,
        gamma=cfg.GAMMA,
        gae_lambda=cfg.GAE_LAMBDA,
        clip_eps=cfg.CLIP_EPS,
        entropy_coef=0.05,  # higher entropy for continuous exploration
        value_coef=cfg.VALUE_COEF,
        ppo_epochs=cfg.MAPPO_EPOCHS,
        batch_size=cfg.MAPPO_BATCH_SIZE,
        rollout_length=cfg.MAPPO_ROLLOUT_LENGTH,
    )

    os.makedirs(save_dir, exist_ok=True)
    variant_dir = os.path.join(save_dir, mode)
    os.makedirs(variant_dir, exist_ok=True)
    rotate_logs(variant_dir)
    log_path = os.path.join(variant_dir, "training_log.json")

    episode_rewards = deque(maxlen=100)
    episode_completions = deque(maxlen=100)
    episode_collisions = deque(maxlen=100)
    logs = []

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Training MAPPO CONTINUOUS (Pseudo-Physics 2D)       ║")
    print(f"║  State: {state_size}  Global: {state_size * cfg.NUM_ROBOTS}  Episodes: {episodes:<5}           ║")
    print(f"║  Device: {agent.device}  Actions: (lin_vel, ang_vel)       ║")
    print(f"╚══════════════════════════════════════════════════════╝")

    best_completion = 0.0

    for ep in range(1, episodes + 1):
        observations = env.reset()
        env.allocate_tasks()
        total_reward = 0.0
        done = False
        ep_update_info = None

        for step in range(cfg.MAX_STEPS_PER_EPISODE):
            actions = []
            action_data = []

            for rid in range(cfg.NUM_ROBOTS):
                if env.robot_failed[rid] or env.robot_done[rid]:
                    actions.append([0.0, 0.0])
                    action_data.append(([0.0, 0.0], 0.0))
                    continue
                obs = observations[rid]
                action, log_prob = agent.select_action(obs, training=True)
                actions.append(action)
                action_data.append((action, log_prob))

            global_state = np.concatenate(observations)
            global_value = agent.get_global_value(observations)

            old_obs = [obs.copy() for obs in observations]
            observations, rewards, done, info = env.step(actions)

            # Mixed rewards
            team_reward = sum(rewards) / cfg.NUM_ROBOTS
            alpha = cfg.MIXED_REWARD_ALPHA
            rewards = [alpha * r + (1 - alpha) * team_reward for r in rewards]

            if step % 10 == 0:
                env.allocate_tasks()

            for rid in range(cfg.NUM_ROBOTS):
                if not env.robot_failed[rid]:
                    a, lp = action_data[rid]
                    agent.store_transition(
                        rid, old_obs[rid], a, lp, rewards[rid], float(done),
                        global_state=global_state if rid == 0 else None,
                        global_value=global_value if rid == 0 else None)
                    total_reward += rewards[rid]

            if agent.should_update():
                ep_update_info = agent.update(observations)

            if done:
                break

            if step == 50 and np.random.random() < 0.1:
                env.simulate_failure(np.random.randint(0, cfg.NUM_ROBOTS))
            if step == 70:
                for rid in range(cfg.NUM_ROBOTS):
                    if env.robot_failed[rid]:
                        env.recover_robot(rid)
                        env.allocate_tasks()

        if len(agent.buffer) > cfg.MAPPO_BATCH_SIZE:
            ep_update_info = agent.update(observations)

        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_completions.append(metrics["completion"])
        episode_collisions.append(metrics["collisions"])

        avg_reward = np.mean(episode_rewards)
        avg_completion = np.mean(episode_completions)
        avg_collisions = np.mean(episode_collisions)

        log_entry = {
            "episode": ep, "reward": total_reward, "avg_reward": avg_reward,
            "completion": metrics["completion"], "avg_completion": avg_completion,
            "collisions": metrics["collisions"], "avg_collisions": avg_collisions,
            "steps": metrics["steps"],
        }
        if ep_update_info:
            log_entry.update({
                "pg_loss": ep_update_info["pg_loss"],
                "vf_loss": ep_update_info["vf_loss"],
                "entropy": ep_update_info["entropy"],
                "lr": ep_update_info["lr_actor"],
                "log_std": ep_update_info["log_std"],
            })
        logs.append(log_entry)

        if ep % 50 == 0 or ep == 1:
            ent = ep_update_info["entropy"] if ep_update_info else 0.0
            log_std = ep_update_info["log_std"] if ep_update_info else [0, 0]
            print(f"  Ep {ep:>5}/{episodes}  |  Reward: {avg_reward:>7.1f}  |  "
                  f"Completion: {avg_completion:.0%}  |  Collisions: {avg_collisions:.1f}  |  "
                  f"Entropy: {ent:.3f}  |  σ: [{log_std[0]:.2f},{log_std[1]:.2f}]")

        if avg_completion > best_completion and ep > 50:
            best_completion = avg_completion
            agent.save(os.path.join(variant_dir, "best.pt"))

    agent.save(os.path.join(variant_dir, "latest.pt"))
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n  ✅ MAPPO Continuous training complete! Best: {best_completion:.0%}")
    return logs


def save_training_report(save_dir="checkpoints", results_dir="results"):
    """Generate a report-ready training summary from saved JSON logs."""
    import csv
    from datetime import datetime

    os.makedirs(results_dir, exist_ok=True)

    variants = ["rl", "improved_rl", "ppo", "improved_ppo", "mappo", "mappo_continuous"]
    variant_labels = {
        "rl": "DQN Baseline",
        "improved_rl": "DQN + Congestion",
        "ppo": "PPO Baseline",
        "improved_ppo": "PPO + Congestion",
        "mappo": "MAPPO (CTDE)",
    }

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  MULTI-AGENT WAREHOUSE — TRAINING REPORT")
    report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"  Grid: {cfg.GRID_ROWS}x{cfg.GRID_COLS} | Robots: {cfg.NUM_ROBOTS} | Objects: {cfg.NUM_OBJECTS}")
    report_lines.append(f"  Max steps: {cfg.MAX_STEPS_PER_EPISODE} | Sensor range: {cfg.SENSOR_RANGE} | Noise: {cfg.SENSOR_NOISE}")
    report_lines.append(f"  Randomized objects: {cfg.RANDOMIZE_OBJECTS} | Exploration: Yes")
    report_lines.append("=" * 70)

    summary_table = []

    for variant in variants:
        log_path = os.path.join(save_dir, variant, "training_log.json")
        if not os.path.exists(log_path):
            continue

        with open(log_path) as f:
            logs = json.load(f)

        if not logs:
            continue

        # Extract key stats
        best_ep = max(logs, key=lambda x: x.get("avg_completion", 0))
        final = logs[-1]

        report_lines.append(f"\n{'─' * 70}")
        report_lines.append(f"  {variant_labels.get(variant, variant)}")
        report_lines.append(f"{'─' * 70}")
        report_lines.append(f"  Episodes trained:    {len(logs)}")
        report_lines.append(f"  Best avg completion: {best_ep['avg_completion']:.0%} (ep {best_ep['episode']})")
        report_lines.append(f"  Best avg reward:     {best_ep['avg_reward']:.1f}")
        report_lines.append(f"  Best avg collisions: {best_ep['avg_collisions']:.1f}")
        report_lines.append(f"  Final avg completion:{final['avg_completion']:.0%}")
        report_lines.append(f"  Final avg reward:    {final['avg_reward']:.1f}")
        report_lines.append(f"  Final avg collisions:{final['avg_collisions']:.1f}")

        summary_table.append({
            "variant": variant_labels.get(variant, variant),
            "episodes": len(logs),
            "best_completion": best_ep["avg_completion"],
            "best_reward": best_ep["avg_reward"],
            "best_collisions": best_ep["avg_collisions"],
            "final_completion": final["avg_completion"],
            "final_reward": final["avg_reward"],
            "final_collisions": final["avg_collisions"],
        })

        # Save per-variant CSV (for LaTeX pgfplots)
        csv_path = os.path.join(results_dir, f"{variant}_curve.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            header = ["episode", "reward", "avg_reward", "completion", "avg_completion",
                       "collisions", "avg_collisions", "steps"]
            if "entropy" in logs[0]:
                header.append("entropy")
            if "epsilon" in logs[0]:
                header.append("epsilon")
            writer.writerow(header)
            for entry in logs:
                row = [entry.get(h, "") for h in header]
                writer.writerow(row)

    # Summary comparison table
    if summary_table:
        report_lines.append(f"\n{'=' * 70}")
        report_lines.append("  COMPARISON SUMMARY")
        report_lines.append(f"{'=' * 70}")
        report_lines.append(f"  {'Variant':<22} {'Best Compl':>10} {'Best Rew':>10} {'Collisions':>10}")
        report_lines.append(f"  {'─' * 52}")
        for row in summary_table:
            report_lines.append(
                f"  {row['variant']:<22} {row['best_completion']:>9.0%} {row['best_reward']:>10.1f} {row['best_collisions']:>10.1f}"
            )

    report_lines.append(f"\n{'=' * 70}\n")

    # Write report
    report_path = os.path.join(results_dir, "training_summary.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  📄 Training report saved to: {report_path}")
    print(f"  📊 CSV curves saved to: {results_dir}/")


def train_all(save_dir="checkpoints"):
    """Train all variants sequentially."""
    print("\n" + "=" * 60)
    print("  TRAINING ALL VARIANTS")
    print("=" * 60)

    modes = [
        ("mappo", train_mappo, cfg.MAPPO_EPISODES),
        ("rl", train_dqn, cfg.TRAIN_EPISODES),
        ("improved_rl", train_dqn, cfg.TRAIN_EPISODES),
        ("ppo", train_ppo, cfg.PPO_EPISODES),
        ("improved_ppo", train_ppo, cfg.PPO_EPISODES),
        ("mappo_continuous", train_mappo_continuous, cfg.MAPPO_EPISODES),
    ]

    for mode, train_fn, episodes in modes:
        print(f"\n{'─' * 60}")
        train_fn(mode=mode, episodes=episodes, save_dir=save_dir)

    print(f"\n{'=' * 60}")
    print("  ALL TRAINING COMPLETE!")
    print("  Generating training curves plot...")
    
    try:
        import plot_results
        plot_results.plot_training_curves()
    except Exception as e:
        print(f"  ⚠ Could not generate plots: {e}")

    # Generate report
    save_training_report(save_dir=save_dir)

    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["rl", "improved_rl", "ppo", "improved_ppo",
                                 "mappo", "mappo_continuous", "all"],
                        default="all")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "all":
        train_all()
    elif args.mode in ("rl", "improved_rl"):
        eps = args.episodes or cfg.TRAIN_EPISODES
        train_dqn(mode=args.mode, episodes=eps)
    elif args.mode == "mappo":
        eps = args.episodes or cfg.MAPPO_EPISODES
        train_mappo(mode=args.mode, episodes=eps)
    elif args.mode == "mappo_continuous":
        eps = args.episodes or cfg.MAPPO_EPISODES
        train_mappo_continuous(mode=args.mode, episodes=eps)
    else:
        eps = args.episodes or cfg.PPO_EPISODES
        train_ppo(mode=args.mode, episodes=eps)
