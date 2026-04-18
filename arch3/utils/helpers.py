"""
utils/helpers.py

Utility classes and functions:
  • RunningMeanStd  – online obs/reward normalization
  • EpisodeLogger   – CSV + tensorboard logging
  • CurriculumScheduler – phase control
"""

import os
import csv
import time
import numpy as np
from config import cfg


# ── Running statistics (Welford's algorithm) ─────────────────────────────────

class RunningMeanStd:
    """
    Tracks running mean and variance for normalization.
    Thread-safe for single-process use.
    """

    def __init__(self, shape=()):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4   # small initial count for numerical stability

    def update(self, x: np.ndarray):
        x     = np.asarray(x, dtype=np.float64)
        batch_mean = x
        batch_var  = np.zeros_like(x)
        batch_count = 1

        delta      = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean   = self.mean + delta * batch_count / total_count
        m_a        = self.var * self.count
        m_b        = batch_var * batch_count
        m2         = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var    = m2 / total_count

        self.mean  = new_mean
        self.var   = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)

    def normalize_reward(self, r: float) -> float:
        return float(r / (np.sqrt(self.var) + 1e-8))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]


# ── Episode logger ────────────────────────────────────────────────────────────

class EpisodeLogger:
    """
    Writes episode stats to a CSV file and optionally to TensorBoard.
    """

    def __init__(self, log_dir: str, writer=None):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "training_log.csv")
        self.writer   = writer
        self._file    = open(self.csv_path, "w", newline="")
        self._csv     = csv.writer(self._file)
        self._csv.writerow([
            "episode", "phase", "steps", "balls_remaining",
            "total_reward", "success", "elapsed_s"
        ])
        self._file.flush()
        self.t0 = time.time()

    def log(self, ep: int, phase: int, steps: int,
            balls_remaining: int, total_reward: float,
            success: bool):
        elapsed = time.time() - self.t0
        self._csv.writerow([ep, phase, steps, balls_remaining,
                            f"{total_reward:.2f}", int(success), f"{elapsed:.1f}"])
        self._file.flush()
        if self.writer is not None:
            self.writer.add_scalar("episode/total_reward", total_reward, ep)
            self.writer.add_scalar("episode/steps", steps, ep)
            self.writer.add_scalar("episode/balls_remaining", balls_remaining, ep)
            self.writer.add_scalar("episode/success", int(success), ep)

    def close(self):
        self._file.close()


# ── Curriculum scheduler ──────────────────────────────────────────────────────

class CurriculumScheduler:
    """
    Controls which curriculum phase is active based on episodes completed.

    Phase 0 → no obstacles   (easiest)
    Phase 1 → sparse obstacles
    Phase 2 → full obstacles  (hardest)
    """

    def __init__(self):
        self.phases = cfg.CURRICULUM_PHASES
        self.current_phase = 0
        self.episodes_in_phase = 0

    def step(self) -> tuple[int, int]:
        """
        Call once per episode.
        Returns (phase_id, num_obstacles).
        Advances phase when episode budget is consumed.
        """
        self.episodes_in_phase += 1
        phase_cfg = self.phases[self.current_phase]
        if (self.episodes_in_phase >= phase_cfg["episodes"] and
                self.current_phase < len(self.phases) - 1):
            self.current_phase += 1
            self.episodes_in_phase = 0
            print(f"\n[Curriculum] ── Advancing to Phase {self.current_phase} "
                  f"({self.phases[self.current_phase]['num_obstacles']} obstacles) ──\n")

        return self.current_phase, self.phases[self.current_phase]["num_obstacles"]

    def get_num_obstacles(self) -> int:
        return self.phases[self.current_phase]["num_obstacles"]


# ── Misc ──────────────────────────────────────────────────────────────────────

def pretty_metrics(ep: int, phase: int, steps: int, reward: float,
                   balls_left: int, losses: dict, fps: float) -> str:
    loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
    return (f"Ep {ep:5d} | Phase {phase} | "
            f"Steps {steps:4d} | Reward {reward:8.2f} | "
            f"Balls left {balls_left} | {loss_str} | FPS {fps:.0f}")