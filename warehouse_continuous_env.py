"""
Continuous 2D Warehouse Environment — Pseudo-Physics for Gazebo Transfer.

Replaces grid-based movement with unicycle dynamics:
    x += v * cos(θ) * dt
    y += v * sin(θ) * dt
    θ += ω * dt

Walls are still defined by the grid layout, but robots move in continuous
(x, y, θ) space. Collision detection is geometric (circle-rectangle).

This is the BEST training environment for Gazebo transfer because:
1. No grid snapping — smooth motion like real robots
2. Heading is continuous angle, not 4 discrete directions
3. Actions (v, ω) map directly to cmd_vel
4. Physics are identical to Gazebo's differential drive
"""

import math
import random
import numpy as np
from collections import deque
import config as cfg
from pathfinding import manhattan_distance


# ─── Constants ───────────────────────────────────────────────────────

ROBOT_RADIUS = 0.35          # robot radius in grid-cell units
PICKUP_RADIUS = 1.5          # generous radius for continuous navigation
DELIVERY_RADIUS = 2.0        # generous delivery zone
MAX_LINEAR_VEL = 1.0         # cells/step max speed
MAX_ANGULAR_VEL = math.pi/3  # rad/step max turn (60°/step, smoother)
DT = 1.0                     # timestep
COLLISION_PENALTY = -3.0     # softer than grid (-10) for continuous physics


class ContinuousWarehouseEnv:
    """Warehouse with continuous 2D physics."""

    def __init__(self, use_congestion=True, mode="mappo_continuous"):
        self.mode = mode
        self.use_congestion = use_congestion
        self.grid = [row[:] for row in cfg.WAREHOUSE_MAP]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_robots = cfg.NUM_ROBOTS

        # Precompute wall cells for collision
        self.wall_cells = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == cfg.WALL:
                    self.wall_cells.add((r, c))

        # Drop zone center (continuous)
        self.drop_zone_center = (float(cfg.DROP_ZONE_CENTER[0]),
                                 float(cfg.DROP_ZONE_CENTER[1]))
        self.drop_zone_cells = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == cfg.DROP_ZONE:
                    self.drop_zone_cells.add((r, c))

        self._total_explorable = sum(1 for r in range(self.rows)
                                     for c in range(self.cols)
                                     if self.grid[r][c] != cfg.WALL)

    def reset(self):
        """Reset environment to initial state."""
        # Robot state: (x, y, theta) — continuous
        spawn = cfg.ROBOT_SPAWN_POSITIONS
        self.robot_x = [float(s[1]) + 0.5 for s in spawn]  # col → x (center of cell)
        self.robot_y = [float(s[0]) + 0.5 for s in spawn]  # row → y (center of cell)
        self.robot_theta = [math.pi / 2] * self.num_robots  # face down (+y direction)

        self.robot_carrying = [False] * self.num_robots
        self.robot_failed = [False] * self.num_robots
        self.robot_done = [False] * self.num_robots

        # Objects: continuous positions (center of cell)
        if cfg.RANDOMIZE_OBJECTS:
            walkable = [(r, c) for r in range(self.rows) for c in range(self.cols)
                        if self.grid[r][c] == cfg.EMPTY
                        and (r, c) not in set(cfg.ROBOT_SPAWN_POSITIONS)
                        and (r, c) not in self.drop_zone_cells]
            grid_objs = random.sample(walkable, min(cfg.NUM_OBJECTS, len(walkable)))
        else:
            grid_objs = list(cfg.OBJECT_POSITIONS)
        self.object_positions = [(float(c) + 0.5, float(r) + 0.5) for r, c in grid_objs]
        self.objects_collected = [False] * len(self.object_positions)
        self.objects_delivered = [False] * len(self.object_positions)
        self.objects_discovered = set()

        # Exploration map
        self.explored_map = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Task assignments: {robot_id: object_index}
        self.assignments = {}

        # Metrics
        self.step_count = 0
        self.total_collisions = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.total_steps_global = getattr(self, 'total_steps_global', 0)  # across episodes

        # Frame stacking
        self._raw_obs_size = None
        self._obs_history = [deque(maxlen=cfg.FRAME_STACK) for _ in range(self.num_robots)]

        return self._get_all_observations()

    # ─── Physics ─────────────────────────────────────────────────────

    def _check_wall_collision(self, x, y):
        """Check if position (x, y) collides with any wall."""
        # Check grid cells that the robot circle overlaps
        min_c = max(0, int(x - ROBOT_RADIUS))
        max_c = min(self.cols - 1, int(x + ROBOT_RADIUS))
        min_r = max(0, int(y - ROBOT_RADIUS))
        max_r = min(self.rows - 1, int(y + ROBOT_RADIUS))

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) in self.wall_cells:
                    # Circle-rectangle collision: find closest point on cell to robot
                    closest_x = max(float(c), min(x, float(c + 1)))
                    closest_y = max(float(r), min(y, float(r + 1)))
                    dx = x - closest_x
                    dy = y - closest_y
                    if dx * dx + dy * dy < ROBOT_RADIUS * ROBOT_RADIUS:
                        return True

        # Boundary collision
        if (x - ROBOT_RADIUS < 0 or x + ROBOT_RADIUS > self.cols or
                y - ROBOT_RADIUS < 0 or y + ROBOT_RADIUS > self.rows):
            return True

        return False

    def _check_robot_collision(self, rid, x, y):
        """Check if robot rid at (x,y) collides with any other robot."""
        for i in range(self.num_robots):
            if i == rid or self.robot_failed[i]:
                continue
            dx = x - self.robot_x[i]
            dy = y - self.robot_y[i]
            if dx * dx + dy * dy < (2 * ROBOT_RADIUS) ** 2:
                return i
        return -1

    def _euclidean_dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ─── Observation (ego-centric) ───────────────────────────────────

    def get_observation(self, robot_id):
        """Ego-centric observation for continuous env.

        All vectors rotated into robot's heading frame.
        State: [ego_fwd, ego_left, dist, carrying, has_target,
                other_robot_fwd_1, other_robot_left_1,
                other_robot_fwd_2, other_robot_left_2,
                obs_fwd, obs_back, obs_left, obs_right,
                sin(θ), cos(θ), frac_explored,
                robot_onehot × 3,
                (congestion, density)]
        """
        x, y, theta = self.robot_x[robot_id], self.robot_y[robot_id], self.robot_theta[robot_id]
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        target = self._get_current_target(robot_id)
        has_target = target is not None
        if target is None:
            target = (x, y)  # no target → zero direction

        # Goal direction (ego-centric)
        gdx = (target[0] - x) / max(self.cols, 1)
        gdy = (target[1] - y) / max(self.rows, 1)
        ego_fwd = gdx * cos_t + gdy * sin_t
        ego_left = -gdx * sin_t + gdy * cos_t
        dist = self._euclidean_dist(x, y, target[0], target[1]) / (self.rows + self.cols)

        carrying = 1.0 if self.robot_carrying[robot_id] else 0.0
        has_target_flag = 1.0 if has_target else 0.0

        # Other robots (ego-centric)
        other_dists = []
        for i in range(self.num_robots):
            if i == robot_id:
                continue
            odx = (self.robot_x[i] - x) / self.cols
            ody = (self.robot_y[i] - y) / self.rows
            d = self._euclidean_dist(x, y, self.robot_x[i], self.robot_y[i])
            if d <= cfg.VISIBILITY_RADIUS:
                of = odx * cos_t + ody * sin_t
                ol = -odx * sin_t + ody * cos_t
                other_dists.extend([of, ol])
            else:
                other_dists.extend([0.0, 0.0])

        # Ray-cast obstacle sensors (ego-centric directions)
        # 4 rays: forward, backward, left, right
        obstacles = []
        ray_angles = [0, math.pi, math.pi / 2, -math.pi / 2]  # fwd, back, left, right
        for ray_offset in ray_angles:
            ray_angle = theta + ray_offset
            ray_cos = math.cos(ray_angle)
            ray_sin = math.sin(ray_angle)
            hit_dist = 0.0
            for step in range(1, cfg.SENSOR_RANGE * 2 + 1):  # finer resolution
                rx = x + ray_cos * step * 0.5
                ry = y + ray_sin * step * 0.5
                # Check bounds
                if not (0 <= rx < self.cols and 0 <= ry < self.rows):
                    hit_dist = 1.0 - (step - 1) / (cfg.SENSOR_RANGE * 2)
                    break
                # Check wall
                gc, gr = int(rx), int(ry)
                if (gr, gc) in self.wall_cells:
                    hit_dist = 1.0 - (step - 1) / (cfg.SENSOR_RANGE * 2)
                    break
                # Check other robots
                for i in range(self.num_robots):
                    if i == robot_id:
                        continue
                    if self._euclidean_dist(rx, ry, self.robot_x[i], self.robot_y[i]) < ROBOT_RADIUS:
                        hit_dist = 1.0 - (step - 1) / (cfg.SENSOR_RANGE * 2)
                        break
                if hit_dist > 0:
                    break
            # Sensor noise
            if cfg.SENSOR_NOISE > 0 and np.random.random() < cfg.SENSOR_NOISE:
                hit_dist = 1.0 - hit_dist
            obstacles.append(hit_dist)

        # Heading encoding
        sin_h, cos_h = math.sin(theta), math.cos(theta)

        # Exploration fraction
        frac_explored = np.sum(self.explored_map) / max(self._total_explorable, 1)

        state = [ego_fwd, ego_left, dist, carrying, has_target_flag] + \
                other_dists + obstacles + [sin_h, cos_h, frac_explored]

        # Robot identity
        robot_onehot = [0.0] * self.num_robots
        robot_onehot[robot_id] = 1.0
        state.extend(robot_onehot)

        if self.use_congestion:
            cong = sum(1.0 / max(self._euclidean_dist(x, y, self.robot_x[i], self.robot_y[i]), 0.1)
                       for i in range(self.num_robots) if i != robot_id
                       and self._euclidean_dist(x, y, self.robot_x[i], self.robot_y[i]) <= cfg.CONGESTION_RADIUS)
            density = sum(1 for i in range(self.num_robots) if i != robot_id
                         and self._euclidean_dist(x, y, self.robot_x[i], self.robot_y[i]) <= cfg.CONGESTION_RADIUS)
            density /= (self.num_robots - 1)
            state.extend([cong, density])

        return np.array(state, dtype=np.float32)

    def _get_stacked_observation(self, robot_id):
        raw = self.get_observation(robot_id)
        if self._raw_obs_size is None:
            self._raw_obs_size = len(raw)
        self._obs_history[robot_id].append(raw)
        frames = list(self._obs_history[robot_id])
        while len(frames) < cfg.FRAME_STACK:
            frames.insert(0, np.zeros(self._raw_obs_size, dtype=np.float32))
        return np.concatenate(frames)

    def _get_all_observations(self):
        return [self._get_stacked_observation(i) for i in range(self.num_robots)]

    def get_state_size(self):
        raw = 19  # 14 base + 2 heading + 3 robot_id
        if self.use_congestion:
            raw = 21
        return raw * cfg.FRAME_STACK

    # ─── Target logic ────────────────────────────────────────────────

    def _get_current_target(self, robot_id):
        """Get navigation target as continuous (x, y)."""
        if self.robot_carrying[robot_id]:
            return self.drop_zone_center
        if robot_id in self.assignments:
            obj_idx = self.assignments[robot_id]
            if not self.objects_collected[obj_idx] and obj_idx in self.objects_discovered:
                return self.object_positions[obj_idx]
        return None

    def allocate_tasks(self):
        """Assign discovered objects to free robots."""
        free_robots = {}
        for i in range(self.num_robots):
            if (not self.robot_failed[i] and not self.robot_carrying[i]
                    and not self.robot_done[i] and i not in self.assignments):
                free_robots[i] = (self.robot_x[i], self.robot_y[i])

        available = []
        assigned_objs = set(self.assignments.values())
        for idx, pos in enumerate(self.object_positions):
            if not self.objects_collected[idx] and idx not in assigned_objs and idx in self.objects_discovered:
                available.append((idx, pos))

        if not free_robots or not available:
            return

        # Greedy closest-first
        costs = []
        for rid, rpos in free_robots.items():
            for idx, opos in available:
                d = self._euclidean_dist(rpos[0], rpos[1], opos[0], opos[1])
                costs.append((d, rid, idx))
        costs.sort()

        assigned_r = set()
        assigned_o = set()
        for d, rid, idx in costs:
            if rid in assigned_r or idx in assigned_o:
                continue
            self.assignments[rid] = idx
            assigned_r.add(rid)
            assigned_o.add(idx)

    # ─── Step ────────────────────────────────────────────────────────

    def step(self, actions):
        """Step with continuous actions: list of (linear_vel, angular_vel) per robot.

        Uses unicycle dynamics:
            x += v * cos(θ) * dt
            y += v * sin(θ) * dt
            θ += ω * dt
        """
        self.step_count += 1
        self.total_steps_global += 1
        rewards = [0.0] * self.num_robots
        info = {"collisions": 0, "pickups": 0, "deliveries": 0, "discoveries": 0}

        old_x = list(self.robot_x)
        old_y = list(self.robot_y)

        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue

            lin_vel, ang_vel = actions[rid]
            # Clamp velocities
            lin_vel = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, float(lin_vel)))
            ang_vel = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, float(ang_vel)))

            # Update heading
            self.robot_theta[rid] += ang_vel * DT
            # Normalize theta to [-π, π]
            self.robot_theta[rid] = (self.robot_theta[rid] + math.pi) % (2 * math.pi) - math.pi

            # Compute new position
            new_x = self.robot_x[rid] + lin_vel * math.cos(self.robot_theta[rid]) * DT
            new_y = self.robot_y[rid] + lin_vel * math.sin(self.robot_theta[rid]) * DT

            # Wall collision check
            if self._check_wall_collision(new_x, new_y):
                rewards[rid] += cfg.REWARD_WALL
                new_x, new_y = self.robot_x[rid], self.robot_y[rid]  # revert

            self.robot_x[rid] = new_x
            self.robot_y[rid] = new_y

        # ─── Robot-robot collision ───────────────────────────────
        for i in range(self.num_robots):
            if self.robot_failed[i] or self.robot_done[i]:
                continue
            for j in range(i + 1, self.num_robots):
                if self.robot_failed[j] or self.robot_done[j]:
                    continue
                if self._euclidean_dist(self.robot_x[i], self.robot_y[i],
                                        self.robot_x[j], self.robot_y[j]) < 2 * ROBOT_RADIUS:
                    info["collisions"] += 1
                    self.total_collisions += 1
                    rewards[i] += COLLISION_PENALTY * 0.5  # symmetric penalty
                    rewards[j] += COLLISION_PENALTY * 0.5

                    # Push apart along collision axis instead of reverting
                    dx = self.robot_x[i] - self.robot_x[j]
                    dy = self.robot_y[i] - self.robot_y[j]
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-6
                    overlap = (2 * ROBOT_RADIUS - dist) / 2.0 + 0.01
                    nx, ny = dx / dist, dy / dist

                    new_ix = self.robot_x[i] + nx * overlap
                    new_iy = self.robot_y[i] + ny * overlap
                    new_jx = self.robot_x[j] - nx * overlap
                    new_jy = self.robot_y[j] - ny * overlap

                    if not self._check_wall_collision(new_ix, new_iy):
                        self.robot_x[i], self.robot_y[i] = new_ix, new_iy
                    else:
                        self.robot_x[i], self.robot_y[i] = old_x[i], old_y[i]

                    if not self._check_wall_collision(new_jx, new_jy):
                        self.robot_x[j], self.robot_y[j] = new_jx, new_jy
                    else:
                        self.robot_x[j], self.robot_y[j] = old_x[j], old_y[j]

        # ─── Distance reward (closer/farther to target) ──────────
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            target = self._get_current_target(rid)

            if target is not None:
                old_dist = self._euclidean_dist(old_x[rid], old_y[rid], target[0], target[1])
                new_dist = self._euclidean_dist(self.robot_x[rid], self.robot_y[rid],
                                                target[0], target[1])
                delta = old_dist - new_dist  # positive = getting closer

                # Continuous shaping: no deadband, always differentiable
                if delta >= 0:
                    rewards[rid] += delta * 2.0
                else:
                    rewards[rid] += delta * 0.3  # softer retreat penalty

                # Heading alignment bonus: reward facing the target
                tx, ty = target
                target_angle = math.atan2(ty - self.robot_y[rid], tx - self.robot_x[rid])
                heading_err = abs(math.atan2(
                    math.sin(target_angle - self.robot_theta[rid]),
                    math.cos(target_angle - self.robot_theta[rid])))
                alignment_bonus = 0.15 * max(0.0, 1.0 - heading_err / math.pi)
                rewards[rid] += alignment_bonus
            else:
                # No target yet — reward forward movement (exploration)
                lin_vel = abs(float(actions[rid][0]))
                if lin_vel > 0.3:
                    rewards[rid] += 0.1

            # Step penalty (lighter for continuous)
            if abs(actions[rid][0]) < 0.1:
                rewards[rid] += cfg.REWARD_WAIT
            else:
                rewards[rid] += cfg.REWARD_STEP * 0.5

            # Proximity penalty — warmup: disabled for first ~200 episodes
            if self.use_congestion and self.total_steps_global > 200 * cfg.MAX_STEPS_PER_EPISODE:
                for other in range(self.num_robots):
                    if other == rid or self.robot_failed[other] or self.robot_done[other]:
                        continue
                    if self._euclidean_dist(self.robot_x[rid], self.robot_y[rid],
                                            self.robot_x[other], self.robot_y[other]) <= cfg.CONGESTION_RADIUS:
                        rewards[rid] += cfg.REWARD_PROXIMITY_PENALTY

        # ─── Exploration map update + frontier reward ────────────
        frontier_scale = max(0.0, 1.0 - self.step_count / (cfg.MAX_STEPS_PER_EPISODE * 0.4))
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            cx, cy = self.robot_x[rid], self.robot_y[rid]
            new_cells = 0
            for dr in range(-cfg.SENSOR_RANGE, cfg.SENSOR_RANGE + 1):
                for dc in range(-cfg.SENSOR_RANGE, cfg.SENSOR_RANGE + 1):
                    nr, nc = int(cy) + dr, int(cx) + dc
                    if (0 <= nr < self.rows and 0 <= nc < self.cols
                            and self.explored_map[nr][nc] == 0
                            and self.grid[nr][nc] != cfg.WALL):
                        self.explored_map[nr][nc] = 1.0
                        new_cells += 1
            if new_cells > 0:
                rewards[rid] += cfg.FRONTIER_REWARD * new_cells * frontier_scale

        # ─── Discovery (find hidden objects within sensor range) ──
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            for obj_idx, obj_pos in enumerate(self.object_positions):
                if obj_idx in self.objects_discovered or self.objects_collected[obj_idx]:
                    continue
                if self._euclidean_dist(self.robot_x[rid], self.robot_y[rid],
                                        obj_pos[0], obj_pos[1]) <= cfg.SENSOR_RANGE:
                    self.objects_discovered.add(obj_idx)
                    rewards[rid] += cfg.REWARD_DISCOVERY
                    info["discoveries"] += 1

        # ─── Pickup ──────────────────────────────────────────────
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid] or self.robot_carrying[rid]:
                continue
            if rid not in self.assignments:
                continue
            obj_idx = self.assignments[rid]
            if self.objects_collected[obj_idx]:
                continue
            obj_pos = self.object_positions[obj_idx]
            if self._euclidean_dist(self.robot_x[rid], self.robot_y[rid],
                                    obj_pos[0], obj_pos[1]) <= PICKUP_RADIUS:
                self.objects_collected[obj_idx] = True
                self.robot_carrying[rid] = True
                del self.assignments[rid]
                rewards[rid] += cfg.REWARD_PICKUP
                self.total_pickups += 1
                info["pickups"] += 1

        # ─── Delivery ───────────────────────────────────────────
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid] or not self.robot_carrying[rid]:
                continue
            if self._euclidean_dist(self.robot_x[rid], self.robot_y[rid],
                                    self.drop_zone_center[0], self.drop_zone_center[1]) <= DELIVERY_RADIUS:
                # Find which object this robot has
                for idx in range(len(self.objects_delivered)):
                    if self.objects_collected[idx] and not self.objects_delivered[idx]:
                        self.objects_delivered[idx] = True
                        break
                self.robot_carrying[rid] = False
                rewards[rid] += cfg.REWARD_GOAL
                self.total_deliveries += 1
                info["deliveries"] += 1

                # Check if robot should be marked done
                remaining = sum(1 for i in range(len(self.objects_delivered))
                               if not self.objects_delivered[i])
                if remaining == 0:
                    self.robot_done[rid] = True

        # Done condition
        done = all(self.objects_delivered) or self.step_count >= cfg.MAX_STEPS_PER_EPISODE

        observations = self._get_all_observations()
        return observations, rewards, done, info

    # ─── Failure / recovery ──────────────────────────────────────

    def simulate_failure(self, robot_id):
        self.robot_failed[robot_id] = True
        if robot_id in self.assignments:
            del self.assignments[robot_id]

    def recover_robot(self, robot_id):
        self.robot_failed[robot_id] = False

    # ─── Metrics ─────────────────────────────────────────────────

    def get_metrics(self):
        return {
            "steps": self.step_count,
            "collisions": self.total_collisions,
            "pickups": self.total_pickups,
            "deliveries": self.total_deliveries,
            "completion": sum(self.objects_delivered) / len(self.object_positions),
            "all_done": all(self.objects_delivered),
        }
