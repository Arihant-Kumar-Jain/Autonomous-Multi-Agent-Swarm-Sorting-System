# utils/utils.py — Helper functions: BFS, state encoding, distance metrics
# These are pure utility functions with no side effects.

from collections import deque
import math
import sys
import os

# Allow imports from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as C


# ─────────────────────────────────────────────
# DISTANCE UTILITIES
# ─────────────────────────────────────────────

def manhattan_distance(a, b):
    """Manhattan distance between two (row, col) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_ball(pos, balls):
    """
    Returns (distance, nearest_ball_pos) for the closest ball.
    Returns (inf, None) if no balls remain.
    """
    if not balls:
        return float("inf"), None
    best_dist = float("inf")
    best_ball = None
    for b in balls:
        d = manhattan_distance(pos, b)
        if d < best_dist:
            best_dist = d
            best_ball = b
    return best_dist, best_ball


def direction_sign(a, b):
    """
    Returns a compact 2-tuple of signs: (row_sign, col_sign)
    indicating the direction from a to b.
    Each component is -1, 0, or +1.
    """
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (
        0 if dr == 0 else (1 if dr > 0 else -1),
        0 if dc == 0 else (1 if dc > 0 else -1),
    )


# ─────────────────────────────────────────────
# BFS — SHORTEST PATH
# ─────────────────────────────────────────────

def bfs_next_step(start, goal, obstacles, grid_rows, grid_cols):
    """
    BFS from start → goal on a grid with obstacles.
    Returns the first step (next_row, next_col) to take, or None if unreachable.
    Used by the rule-based greedy baseline.
    """
    if start == goal:
        return start

    visited = {start}
    # Queue stores (current_pos, first_step_taken)
    queue = deque()
    queue.append((start, None))

    while queue:
        pos, first_step = queue.popleft()
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            npos = (nr, nc)
            if (
                0 <= nr < grid_rows and
                0 <= nc < grid_cols and
                npos not in obstacles and
                npos not in visited
            ):
                step = first_step if first_step is not None else npos
                if npos == goal:
                    return step
                visited.add(npos)
                queue.append((npos, step))
    return None   # Unreachable


def bfs_distance(start, goal, obstacles, grid_rows, grid_cols):
    """
    BFS distance from start to goal. Returns inf if unreachable.
    Slower than Manhattan but correct when obstacles exist.
    """
    if start == goal:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        pos, dist = queue.popleft()
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            npos = (nr, nc)
            if (
                0 <= nr < grid_rows and
                0 <= nc < grid_cols and
                npos not in obstacles and
                npos not in visited
            ):
                if npos == goal:
                    return dist + 1
                visited.add(npos)
                queue.append((npos, dist + 1))
    return float("inf")


# ─────────────────────────────────────────────
# STATE ENCODING — compact tuple for Q-table key
# ─────────────────────────────────────────────

def coarse_pos(pos, rows=C.GRID_ROWS, cols=C.GRID_COLS, bins=C.COARSE_BINS):
    """
    Discretize (r, c) into a coarse grid position.
    Reduces the state space enormously while preserving spatial structure.
    E.g., bins=5 → 5×5=25 possible position buckets on a 20×20 grid.
    """
    r_bin = int(pos[0] * bins / rows)
    c_bin = int(pos[1] * bins / cols)
    return (min(r_bin, bins - 1), min(c_bin, bins - 1))


def encode_state(agent_pos, balls, other_agent_positions):
    """
    Encode the state into a compact hashable tuple for the Q-table.

    Components:
    1. Coarse agent position         → coarse (row_bin, col_bin)
    2. Direction to nearest ball     → (dr_sign, dc_sign)
    3. Distance bucket to nearest    → 0..4
    4. Direction to 2nd nearest ball → (dr_sign, dc_sign)  [or (0,0)]
    5. Relative direction to each other agent → list of (dr_sign, dc_sign)
    6. Whether nearest ball is within OBS_RADIUS → bool

    Keeping it compact is critical: Q-table size = states × actions.
    """
    cp = coarse_pos(agent_pos)

    # Nearest ball info
    ball_list = list(balls)
    dist1, b1 = nearest_ball(agent_pos, ball_list)

    if b1 is not None:
        dir1 = direction_sign(agent_pos, b1)
        # Bucket distance into 5 levels (0=same cell, 4=far)
        dist_bucket = min(int(dist1 / 4), 4)
        nearby = int(dist1 <= C.OBS_RADIUS)
    else:
        dir1 = (0, 0)
        dist_bucket = 4
        nearby = 0

    # Second nearest ball direction (helps avoid redundant chasing)
    remaining = [b for b in ball_list if b != b1]
    _, b2 = nearest_ball(agent_pos, remaining)
    dir2 = direction_sign(agent_pos, b2) if b2 is not None else (0, 0)

    # Other agents' relative directions
    other_dirs = tuple(
        direction_sign(agent_pos, op) for op in sorted(other_agent_positions)
    )

    return (cp, dir1, dist_bucket, dir2, other_dirs, nearby)


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def log_episode(episode, steps, collisions, balls_collected, epsilon, path=C.LOG_PATH):
    """Append one row to the CSV training log."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("episode,steps,collisions,balls_collected,epsilon\n")
        f.write(f"{episode},{steps},{collisions},{balls_collected},{epsilon:.4f}\n")
