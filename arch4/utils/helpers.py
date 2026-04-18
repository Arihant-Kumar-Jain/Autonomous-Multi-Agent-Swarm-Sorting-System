"""
utils/helpers.py - Utilities for arch4

  RunningMeanStd     - online mean/variance for obs normalization
  PerformanceCurriculum - advances phase based on agent performance,
                         not a fixed episode count
  EpisodeLogger      - CSV + TensorBoard logging
"""

import os
import csv
import time
import numpy as np


# ── Running Mean/Std (observation normalization) ──────────────────────────────

class RunningMeanStd:
    """
    Welford's online algorithm for running mean and variance.
    Used to normalize observations before feeding to the network.
    """

    def __init__(self, shape: tuple, epsilon: float = 1e-4):
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape,  dtype=np.float64)
        self.count   = epsilon

    def update(self, x: np.ndarray):
        batch_mean  = x.astype(np.float64)
        batch_var   = np.zeros_like(batch_mean)
        batch_count = 1

        delta       = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean  += delta * batch_count / total_count
        m_a         = self.var  * self.count
        m_b         = batch_var * batch_count
        M2          = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var    = M2 / total_count
        self.count  = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean  = d["mean"]
        self.var   = d["var"]
        self.count = d["count"]


# ── Performance-Based Curriculum ──────────────────────────────────────────────

class PerformanceCurriculum:
    """
    Advances the obstacle curriculum based on ACTUAL agent performance,
    not a fixed episode count.

    Arch3 problem: phase advanced at episode 300 regardless of whether
    agents had learned anything (they still had 10/10 balls left at ep 290).

    This curriculum tracks the rolling average of balls delivered per episode
    and only advances when the agents consistently meet the threshold.
    """

    def __init__(self, phases: dict = None):
        from config import cfg
        self.phases      = phases or cfg.CURRICULUM_PHASES
        self.phase       = 0
        self.window      = self.phases[0].get("window", 50)
        self.history     = []   # recent balls_delivered values
        self.ep_count    = 0
        self._announced  = False

    @property
    def num_obstacles(self) -> int:
        return self.phases[self.phase]["num_obstacles"]

    @property
    def advance_threshold(self) -> float:
        return self.phases[self.phase].get("advance_threshold", 9999)

    def record(self, balls_delivered: int) -> int:
        """
        Record result of one episode. Returns current phase.
        Call this at the end of each episode.
        """
        self.ep_count += 1
        self.history.append(balls_delivered)
        # Keep only the last `window` results
        if len(self.history) > self.window:
            self.history.pop(0)

        # Check if we should advance phase
        max_phase = max(self.phases.keys())
        if (self.phase < max_phase and
                len(self.history) >= self.window // 2 and  # need at least half-window
                np.mean(self.history) >= self.advance_threshold):
            self.phase    += 1
            self.window    = self.phases[self.phase].get("window", 50)
            self.history   = []
            self._announced = True

        return self.phase

    def pop_announcement(self) -> str | None:
        """Returns a phase-change message if one just happened, else None."""
        if self._announced:
            self._announced = False
            p = self.phases[self.phase]
            return (f"\n[Curriculum] ── Advancing to Phase {self.phase} "
                    f"({p['num_obstacles']} obstacles) ──\n")
        return None


# ── Episode Logger ────────────────────────────────────────────────────────────

class EpisodeLogger:
    """Logs episode metrics to CSV and optionally TensorBoard."""

    def __init__(self, log_dir: str, writer=None):
        os.makedirs(log_dir, exist_ok=True)
        self.writer   = writer
        csv_path      = os.path.join(log_dir, "training_log.csv")
        self._file    = open(csv_path, "w", newline="")
        self._csv     = csv.writer(self._file)
        self._csv.writerow(["episode", "phase", "steps", "balls_delivered",
                             "reward", "success"])

    def log(self, episode: int, phase: int, steps: int,
            balls_delivered: int, reward: float, success: bool):
        self._csv.writerow([episode, phase, steps, balls_delivered,
                            f"{reward:.3f}", int(success)])
        self._file.flush()
        if self.writer is not None:
            self.writer.add_scalar("ep/reward",          reward,          episode)
            self.writer.add_scalar("ep/balls_delivered", balls_delivered, episode)
            self.writer.add_scalar("ep/success",         int(success),    episode)
            self.writer.add_scalar("ep/phase",           phase,           episode)

    def close(self):
        self._file.close()


# ── Pretty metrics printer ────────────────────────────────────────────────────

def pretty_metrics(episode, phase, steps, reward, balls_delivered,
                   losses: dict, fps: float) -> str:
    loss_str = "  ".join(
        f"{k}={v:7.4f}" for k, v in losses.items()
    ) if losses else "no update yet"
    return (
        f"Ep {episode:5d} | Phase {phase} | Steps {steps:4d} | "
        f"Reward {reward:8.2f} | Delivered {balls_delivered:2d}/10 | "
        f"{loss_str} | FPS {fps:.0f}"
    )
