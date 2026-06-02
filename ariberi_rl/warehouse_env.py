"""
Multi-Agent Warehouse Environment.

Gym-like interface for 3 robots in a warehouse grid.
Supports BFS baseline, RL, and improved RL (congestion-aware) modes.
"""

import copy
import random
import numpy as np
from collections import deque
import math
import config as cfg
from pathfinding import bfs, manhattan_distance
from task_allocator import allocate_tasks_greedy, compute_congestion, reallocate_on_failure
from mappo_continuous_agent import heading_to_sincos, rotate_to_ego, HEADING_DELTAS


class WarehouseEnv:
    """Multi-agent warehouse environment."""

    def __init__(self, use_congestion=False, mode="bfs"):
        """
        Args:
            use_congestion: include congestion in state/reward (improved RL)
            mode: 'bfs' | 'rl' | 'improved_rl'
        """
        self.mode = mode
        self.use_congestion = use_congestion
        self.grid = [row[:] for row in cfg.WAREHOUSE_MAP]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # Robots
        self.num_robots = cfg.NUM_ROBOTS
        self.robot_positions = list(cfg.ROBOT_SPAWN_POSITIONS)
        self.robot_carrying = [False] * self.num_robots
        self.robot_failed = [False] * self.num_robots
        self.robot_done = [False] * self.num_robots  # reached drop zone with object

        # Objects
        self.objects = list(cfg.OBJECT_POSITIONS)
        self.objects_collected = [False] * len(self.objects)
        self.objects_delivered = [False] * len(self.objects)

        # Task assignments: {robot_id: object_index}
        self.assignments = {}

        # BFS paths (for BFS mode)
        self.bfs_paths = {}

        # Metrics
        self.step_count = 0
        self.total_collisions = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.history = []  # list of snapshots for replay

        # Drop zone cells
        self.drop_zone_cells = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == cfg.DROP_ZONE:
                    self.drop_zone_cells.add((r, c))

    def reset(self):
        """Reset environment to initial state."""
        self.grid = [row[:] for row in cfg.WAREHOUSE_MAP]
        self.robot_positions = list(cfg.ROBOT_SPAWN_POSITIONS)
        self.robot_carrying = [False] * self.num_robots
        self.robot_failed = [False] * self.num_robots
        self.robot_done = [False] * self.num_robots
        self.objects = list(cfg.OBJECT_POSITIONS)
        # Randomize object positions if enabled
        if cfg.RANDOMIZE_OBJECTS:
            walkable = [(r, c) for r in range(self.rows) for c in range(self.cols)
                        if self.grid[r][c] == cfg.EMPTY
                        and (r, c) not in set(cfg.ROBOT_SPAWN_POSITIONS)
                        and (r, c) not in self.drop_zone_cells]
            self.objects = random.sample(walkable, min(cfg.NUM_OBJECTS, len(walkable)))
        self.objects_collected = [False] * len(self.objects)
        self.objects_delivered = [False] * len(self.objects)
        # Exploration: objects only discovered when within SENSOR_RANGE
        self.objects_discovered = set()
        # Shared exploration map: which cells have been seen by any robot
        self.explored_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._total_explorable = sum(1 for r in range(self.rows) for c in range(self.cols)
                                     if self.grid[r][c] != cfg.WALL)
        self.assignments = {}
        self.bfs_paths = {}
        self.step_count = 0
        self.total_collisions = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.history = []
        # Frame stacking: maintain deque of last N raw observations per robot
        self._raw_obs_size = None  # set on first get_observation
        self._obs_history = [deque(maxlen=cfg.FRAME_STACK) for _ in range(self.num_robots)]
        # Continuous mode: robot headings (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        self.robot_headings = [2] * self.num_robots  # face DOWN initially (toward objects)
        self._raw_obs_size_cont = None
        self._obs_history_cont = [deque(maxlen=cfg.FRAME_STACK) for _ in range(self.num_robots)]
        self._save_snapshot()
        return self._get_all_observations()

    def _save_snapshot(self):
        """Save current state for replay."""
        self.history.append({
            "positions": list(self.robot_positions),
            "carrying": list(self.robot_carrying),
            "objects": list(self.objects),
            "collected": list(self.objects_collected),
            "delivered": list(self.objects_delivered),
            "assignments": dict(self.assignments),
            "failed": list(self.robot_failed),
            "done": list(self.robot_done),
            "step": self.step_count,
        })

    # ─── Observation ────────────────────────────────────────────────

    def get_observation(self, robot_id):
        """Get observation for a single robot.
        
        State vector (12 dims, 14 for improved RL):
            [0-1]  goal_dx, goal_dy (normalized)
            [2]    distance_to_goal (normalized)
            [3]    carrying_object (0/1)
            [4-5]  dist to robot 1 (normalized)
            [6-7]  dist to robot 2 (normalized)
            [8-11] obstacle sensors (up, down, left, right) — 0=free, 1=blocked
            [12]   congestion (improved RL only)
            [13]   local_density (improved RL only)
        """
        pos = self.robot_positions[robot_id]
        target = self._get_current_target(robot_id)

        has_target = target is not None
        if target is None:
            # No discovered object, not carrying — must explore
            target = pos  # self-position → (0,0,0) direction

        # Direction to target (normalized)
        dr = (target[0] - pos[0]) / max(self.rows, 1)
        dc = (target[1] - pos[1]) / max(self.cols, 1)
        dist = manhattan_distance(pos, target) / (self.rows + self.cols)

        carrying = 1.0 if self.robot_carrying[robot_id] else 0.0
        has_target_flag = 1.0 if has_target else 0.0

        # Distance to other robots (limited by visibility radius)
        other_dists = []
        for i in range(self.num_robots):
            if i == robot_id:
                continue
            opos = self.robot_positions[i]
            dist_to_other = manhattan_distance(pos, opos)
            if dist_to_other <= cfg.VISIBILITY_RADIUS:
                other_dists.append((opos[0] - pos[0]) / self.rows)
                other_dists.append((opos[1] - pos[1]) / self.cols)
            else:
                # Can't see this robot — zero out
                other_dists.append(0.0)
                other_dists.append(0.0)

        # Obstacle sensors — ray-cast up to SENSOR_RANGE cells
        obstacles = []
        for action_id in range(4):  # UP, DOWN, LEFT, RIGHT
            ddr, ddc = cfg.ACTIONS[action_id]
            hit_dist = 0.0  # 0 = no obstacle within range
            for step in range(1, cfg.SENSOR_RANGE + 1):
                nr, nc = pos[0] + ddr * step, pos[1] + ddc * step
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    hit_dist = 1.0 - (step - 1) / cfg.SENSOR_RANGE  # boundary wall
                    break
                if self.grid[nr][nc] == cfg.WALL:
                    hit_dist = 1.0 - (step - 1) / cfg.SENSOR_RANGE  # closer = higher
                    break
                for i in range(self.num_robots):
                    if i != robot_id and self.robot_positions[i] == (nr, nc):
                        hit_dist = 1.0 - (step - 1) / cfg.SENSOR_RANGE
                        break
                if hit_dist > 0:
                    break
            # Add sensor noise
            if cfg.SENSOR_NOISE > 0 and np.random.random() < cfg.SENSOR_NOISE:
                hit_dist = 1.0 - hit_dist  # flip reading
            obstacles.append(hit_dist)

        state = [dr, dc, dist, carrying, has_target_flag] + other_dists + obstacles

        # Fraction of map explored (shared knowledge)
        explored_count = np.sum(self.explored_map)
        frac_explored = explored_count / max(self._total_explorable, 1)
        state.append(frac_explored)

        # Robot identity (one-hot)
        robot_onehot = [0.0] * self.num_robots
        robot_onehot[robot_id] = 1.0
        state.extend(robot_onehot)

        if self.use_congestion:
            cong = compute_congestion(pos, self.robot_positions, robot_id)
            density = sum(1 for i in range(self.num_robots)
                         if i != robot_id
                         and manhattan_distance(pos, self.robot_positions[i]) <= cfg.CONGESTION_RADIUS)
            density /= (self.num_robots - 1)
            state.extend([cong, density])

        raw_obs = np.array(state, dtype=np.float32)
        return raw_obs

    def _get_stacked_observation(self, robot_id):
        """Get frame-stacked observation for a robot."""
        raw = self.get_observation(robot_id)
        if self._raw_obs_size is None:
            self._raw_obs_size = len(raw)
        # Push to history
        self._obs_history[robot_id].append(raw)
        # Pad with zeros if fewer than FRAME_STACK frames
        frames = list(self._obs_history[robot_id])
        while len(frames) < cfg.FRAME_STACK:
            frames.insert(0, np.zeros(self._raw_obs_size, dtype=np.float32))
        return np.concatenate(frames)

    def _get_all_observations(self):
        """Get observations for all robots (frame-stacked)."""
        return [self._get_stacked_observation(i) for i in range(self.num_robots)]

    def _get_current_target(self, robot_id):
        """Get current navigation target for robot.
        
        Only returns targets the robot actually knows about:
        - Carrying → drop zone (always known)
        - Has assignment to a DISCOVERED object → that object
        - Otherwise → None (must explore)
        """
        if self.robot_carrying[robot_id]:
            return cfg.DROP_ZONE_CENTER
        if robot_id in self.assignments:
            obj_idx = self.assignments[robot_id]
            if not self.objects_collected[obj_idx] and obj_idx in self.objects_discovered:
                return self.objects[obj_idx]
        return None  # must explore

    # ─── Task Allocation ────────────────────────────────────────────

    def allocate_tasks(self):
        """Assign available objects to free robots."""
        free_robots = {}
        for i in range(self.num_robots):
            if (not self.robot_failed[i]
                    and not self.robot_carrying[i]
                    and not self.robot_done[i]
                    and i not in self.assignments):
                free_robots[i] = self.robot_positions[i]

        available_objects = []
        assigned_obj_indices = set(self.assignments.values())
        for idx, obj in enumerate(self.objects):
            if (not self.objects_collected[idx]
                    and idx not in assigned_obj_indices
                    and idx in self.objects_discovered):
                available_objects.append((idx, obj))

        if not free_robots or not available_objects:
            return

        # Use task allocator
        robot_pos_dict = free_robots
        obj_positions = [obj for _, obj in available_objects]
        obj_indices = [idx for idx, _ in available_objects]

        result = allocate_tasks_greedy(
            robot_pos_dict, obj_positions,
            self.robot_positions, self.use_congestion
        )

        for rid, target_pos in result.items():
            # Map position back to object index
            for idx, pos in available_objects:
                if pos == target_pos:
                    self.assignments[rid] = idx
                    break

    # ─── BFS Planning ───────────────────────────────────────────────

    def plan_bfs_paths(self):
        """Compute BFS paths for all assigned robots."""
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            target = self._get_current_target(rid)
            if target is None:
                self.bfs_paths.pop(rid, None)
                continue

            # Blocked cells: other robot positions (except self)
            blocked = set()
            for i in range(self.num_robots):
                if i != rid:
                    blocked.add(self.robot_positions[i])

            path = bfs(self.grid, self.robot_positions[rid], target, blocked)
            if path and len(path) > 1:
                self.bfs_paths[rid] = path[1:]  # skip current position
            else:
                self.bfs_paths[rid] = []

    def get_bfs_action(self, robot_id):
        """Get next action from BFS path."""
        if robot_id not in self.bfs_paths or not self.bfs_paths[robot_id]:
            # No path — explore randomly (pick a valid direction)
            curr = self.robot_positions[robot_id]
            valid_actions = []
            for aid in range(4):  # UP, DOWN, LEFT, RIGHT
                ddr, ddc = cfg.ACTIONS[aid]
                nr, nc = curr[0] + ddr, curr[1] + ddc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and self.grid[nr][nc] != cfg.WALL):
                    valid_actions.append(aid)
            if valid_actions:
                return random.choice(valid_actions)
            return 4  # truly stuck

        # Pop any waypoints the robot has already reached
        curr_pos = self.robot_positions[robot_id]
        while self.bfs_paths[robot_id] and self.bfs_paths[robot_id][0] == curr_pos:
            self.bfs_paths[robot_id].pop(0)

        if not self.bfs_paths[robot_id]:
            return 4  # WAIT — at destination

        next_pos = self.bfs_paths[robot_id][0]
        dr = next_pos[0] - curr_pos[0]
        dc = next_pos[1] - curr_pos[1]

        # Check if next position is occupied by another robot
        for i in range(self.num_robots):
            if i != robot_id and self.robot_positions[i] == next_pos:
                # Priority-based: higher priority robot goes, lower waits
                if cfg.PRIORITY_ORDER.index(robot_id) > cfg.PRIORITY_ORDER.index(i):
                    return 4  # WAIT
                break

        for action_id, (adr, adc) in cfg.ACTIONS.items():
            if adr == dr and adc == dc:
                return action_id
        return 4

    # ─── Step ───────────────────────────────────────────────────────

    def step(self, actions):
        """Execute one step for all robots.
        
        Args:
            actions: list of action_ids, one per robot
        
        Returns:
            observations, rewards, done, info
        """
        self.step_count += 1
        rewards = [0.0] * self.num_robots
        info = {"collisions": 0, "pickups": 0, "deliveries": 0, "failures": 0, "discoveries": 0}

        # Compute proposed positions
        proposed = []
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                proposed.append(self.robot_positions[rid])
                continue

            action = actions[rid]
            dr, dc = cfg.ACTIONS[action]
            nr = self.robot_positions[rid][0] + dr
            nc = self.robot_positions[rid][1] + dc

            # Boundary / wall check
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != cfg.WALL:
                proposed.append((nr, nc))
            else:
                proposed.append(self.robot_positions[rid])
                rewards[rid] += cfg.REWARD_WALL

        # ─── Collision detection & resolution ───────────────────────
        # Priority-based: detect conflicts, lower priority waits
        final_positions = list(proposed)
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if self.robot_failed[i] or self.robot_failed[j]:
                    continue
                if self.robot_done[i] or self.robot_done[j]:
                    continue
                if final_positions[i] == final_positions[j]:
                    # Collision! Lower priority robot reverts
                    info["collisions"] += 1
                    self.total_collisions += 1
                    pi = cfg.PRIORITY_ORDER.index(i)
                    pj = cfg.PRIORITY_ORDER.index(j)
                    if pi > pj:
                        final_positions[i] = self.robot_positions[i]
                        rewards[i] += cfg.REWARD_COLLISION
                    else:
                        final_positions[j] = self.robot_positions[j]
                        rewards[j] += cfg.REWARD_COLLISION

                    # Congestion penalty for improved RL
                    if self.use_congestion:
                        rewards[i] += cfg.REWARD_CONGESTION_PENALTY
                        rewards[j] += cfg.REWARD_CONGESTION_PENALTY

        # ─── Swap collision detection ───────────────────────────────
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if self.robot_failed[i] or self.robot_failed[j]:
                    continue
                if (final_positions[i] == self.robot_positions[j] and
                        final_positions[j] == self.robot_positions[i]):
                    # Swap collision
                    info["collisions"] += 1
                    self.total_collisions += 1
                    pi = cfg.PRIORITY_ORDER.index(i)
                    pj = cfg.PRIORITY_ORDER.index(j)
                    if pi > pj:
                        final_positions[i] = self.robot_positions[i]
                        rewards[i] += cfg.REWARD_COLLISION
                    else:
                        final_positions[j] = self.robot_positions[j]
                        rewards[j] += cfg.REWARD_COLLISION

        # ─── Apply movement ─────────────────────────────────────────
        old_positions = list(self.robot_positions)
        for rid in range(self.num_robots):
            self.robot_positions[rid] = final_positions[rid]

        # ─── Rewards: closer/farther to target ──────────────────────
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            target = self._get_current_target(rid)
            if target is None:
                continue
            old_dist = manhattan_distance(old_positions[rid], target)
            new_dist = manhattan_distance(self.robot_positions[rid], target)
            if new_dist < old_dist:
                rewards[rid] += cfg.REWARD_CLOSER
            elif new_dist > old_dist:
                rewards[rid] += cfg.REWARD_FARTHER

            # Step penalty
            if actions[rid] == 4:
                rewards[rid] += cfg.REWARD_WAIT
            else:
                rewards[rid] += cfg.REWARD_STEP

            # Per-step proximity penalty (improved RL only)
            if self.use_congestion:
                for other_rid in range(self.num_robots):
                    if other_rid == rid or self.robot_failed[other_rid] or self.robot_done[other_rid]:
                        continue
                    if manhattan_distance(self.robot_positions[rid], self.robot_positions[other_rid]) <= cfg.CONGESTION_RADIUS:
                        rewards[rid] += cfg.REWARD_PROXIMITY_PENALTY

        # ─── Exploration map update + frontier reward (decaying) ────────
        # Frontier reward decays within the episode: full strength early, zero late
        # This prevents over-exploration and transitions to task focus
        frontier_scale = max(0.0, 1.0 - self.step_count / (cfg.MAX_STEPS_PER_EPISODE * 0.4))
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            pos = self.robot_positions[rid]
            new_cells = 0
            for dr in range(-cfg.SENSOR_RANGE, cfg.SENSOR_RANGE + 1):
                for dc in range(-cfg.SENSOR_RANGE, cfg.SENSOR_RANGE + 1):
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if (0 <= nr < self.rows and 0 <= nc < self.cols
                            and self.explored_map[nr][nc] == 0
                            and self.grid[nr][nc] != cfg.WALL):
                        self.explored_map[nr][nc] = 1.0
                        new_cells += 1
            if new_cells > 0:
                rewards[rid] += cfg.FRONTIER_REWARD * new_cells * frontier_scale

        # ─── Discovery scan ──────────────────────────────────────────
        # Each robot discovers objects within SENSOR_RANGE
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            for obj_idx, obj_pos in enumerate(self.objects):
                if obj_idx in self.objects_discovered or self.objects_collected[obj_idx]:
                    continue
                if manhattan_distance(self.robot_positions[rid], obj_pos) <= cfg.SENSOR_RANGE:
                    self.objects_discovered.add(obj_idx)
                    rewards[rid] += cfg.REWARD_DISCOVERY
                    info["discoveries"] += 1

        # ─── Pickup check ───────────────────────────────────────────
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid] or self.robot_carrying[rid]:
                continue
            if rid in self.assignments:
                obj_idx = self.assignments[rid]
                if (not self.objects_collected[obj_idx] and
                        self.robot_positions[rid] == self.objects[obj_idx]):
                    self.objects_collected[obj_idx] = True
                    self.robot_carrying[rid] = True
                    rewards[rid] += cfg.REWARD_PICKUP
                    info["pickups"] += 1
                    self.total_pickups += 1
                    # Remove assignment (now heading to drop zone)
                    del self.assignments[rid]

        # ─── Delivery check ─────────────────────────────────────────
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                continue
            if self.robot_carrying[rid]:
                if self.robot_positions[rid] in self.drop_zone_cells:
                    self.robot_carrying[rid] = False
                    # Mark corresponding object as delivered
                    for idx in range(len(self.objects)):
                        if self.objects_collected[idx] and not self.objects_delivered[idx]:
                            self.objects_delivered[idx] = True
                            break
                    rewards[rid] += cfg.REWARD_GOAL
                    info["deliveries"] += 1
                    self.total_deliveries += 1
                    # Robot is now free for new tasks (NOT marked done)
                    # It will be re-assigned in the next allocate_tasks() call
                    # Only mark done when ALL objects are delivered
                    if all(self.objects_delivered):
                        self.robot_done[rid] = True

        # ─── Check completion ───────────────────────────────────────
        all_delivered = all(self.objects_delivered)
        no_more_tasks = not any(
            not self.objects_collected[i] for i in range(len(self.objects))
        ) and not any(self.robot_carrying)
        done = all_delivered or self.step_count >= cfg.MAX_STEPS_PER_EPISODE

        self._save_snapshot()

        observations = self._get_all_observations()
        info["step"] = self.step_count
        info["all_delivered"] = all_delivered

        return observations, rewards, done, info

    # ─── Failure simulation ─────────────────────────────────────────

    def simulate_failure(self, robot_id):
        """Simulate a robot failure — reassign its task."""
        self.robot_failed[robot_id] = True
        if robot_id in self.assignments:
            obj_idx = self.assignments.pop(robot_id)
            # Reassign to remaining robots
            free_robots = {}
            for i in range(self.num_robots):
                if not self.robot_failed[i] and not self.robot_done[i] and i not in self.assignments:
                    free_robots[i] = self.robot_positions[i]
            if free_robots:
                best_rid = min(free_robots.keys(),
                               key=lambda r: manhattan_distance(free_robots[r], self.objects[obj_idx]))
                self.assignments[best_rid] = obj_idx

    def recover_robot(self, robot_id):
        """Recover a failed robot."""
        self.robot_failed[robot_id] = False

    # ─── Info ───────────────────────────────────────────────────────

    def get_metrics(self):
        """Get current metrics."""
        return {
            "steps": self.step_count,
            "collisions": self.total_collisions,
            "pickups": self.total_pickups,
            "deliveries": self.total_deliveries,
            "completion": sum(self.objects_delivered) / len(self.objects),
            "all_done": all(self.objects_delivered),
        }

    def get_state_size(self):
        """Get observation size (with frame stacking)."""
        raw = 17  # 14 base + 3 robot_id one-hot (has_target_flag + fraction_explored)
        if self.use_congestion:
            raw = 19  # + congestion + density
        return raw * cfg.FRAME_STACK  # frame-stacked

    def get_continuous_state_size(self):
        """Get observation size for continuous (ego-centric) mode."""
        # Same as discrete but +2 for heading (sin, cos)
        raw = 19  # 17 base + 2 heading
        if self.use_congestion:
            raw = 21
        return raw * cfg.FRAME_STACK

    # ─── Continuous mode (ego-centric) ───────────────────────────────

    def get_continuous_observation(self, robot_id):
        """Get ego-centric observation for continuous MAPPO.

        All direction vectors are rotated into the robot's heading frame:
          - 'forward' means the direction the robot is facing
          - 'left' means 90° left of heading
        This is critical for Gazebo transfer — the robot thinks in
        body-frame coordinates, not world coordinates.

        Additional dims vs discrete: +2 for heading (sin, cos).
        """
        pos = self.robot_positions[robot_id]
        heading = self.robot_headings[robot_id]
        target = self._get_current_target(robot_id)

        has_target = target is not None
        if target is None:
            target = pos

        # Direction to target — ego-centric
        dx = (target[0] - pos[0]) / max(self.rows, 1)
        dy = (target[1] - pos[1]) / max(self.cols, 1)
        ego_fwd, ego_left = rotate_to_ego(dx, dy, heading)
        dist = manhattan_distance(pos, target) / (self.rows + self.cols)

        carrying = 1.0 if self.robot_carrying[robot_id] else 0.0
        has_target_flag = 1.0 if has_target else 0.0

        # Distance to other robots — ego-centric
        other_dists = []
        for i in range(self.num_robots):
            if i == robot_id:
                continue
            opos = self.robot_positions[i]
            dist_to_other = manhattan_distance(pos, opos)
            if dist_to_other <= cfg.VISIBILITY_RADIUS:
                odx = (opos[0] - pos[0]) / self.rows
                ody = (opos[1] - pos[1]) / self.cols
                ego_f, ego_l = rotate_to_ego(odx, ody, heading)
                other_dists.extend([ego_f, ego_l])
            else:
                other_dists.extend([0.0, 0.0])

        # Obstacle sensors — ego-centric ray-cast
        # Reorder: FORWARD, BACKWARD, LEFT, RIGHT (relative to heading)
        obstacles = []
        # Sensor directions in ego frame → map to world heading offsets
        sensor_offsets = [0, 2, 3, 1]  # fwd, back, left, right relative to heading
        for offset in sensor_offsets:
            world_dir = (heading + offset) % 4
            ddr, ddc = HEADING_DELTAS[world_dir]
            hit_dist = 0.0
            for step in range(1, cfg.SENSOR_RANGE + 1):
                nr, nc = pos[0] + ddr * step, pos[1] + ddc * step
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    hit_dist = 1.0 - (step - 1) / cfg.SENSOR_RANGE
                    break
                if self.grid[nr][nc] == cfg.WALL:
                    hit_dist = 1.0 - (step - 1) / cfg.SENSOR_RANGE
                    break
                for i in range(self.num_robots):
                    if i != robot_id and self.robot_positions[i] == (nr, nc):
                        hit_dist = 1.0 - (step - 1) / cfg.SENSOR_RANGE
                        break
                if hit_dist > 0:
                    break
            if cfg.SENSOR_NOISE > 0 and np.random.random() < cfg.SENSOR_NOISE:
                hit_dist = 1.0 - hit_dist
            obstacles.append(hit_dist)

        # Heading encoding
        sin_h, cos_h = heading_to_sincos(heading)

        # Fraction explored
        frac_explored = np.sum(self.explored_map) / max(self._total_explorable, 1)

        state = [ego_fwd, ego_left, dist, carrying, has_target_flag] + \
                other_dists + obstacles + [sin_h, cos_h, frac_explored]

        # Robot identity one-hot
        robot_onehot = [0.0] * self.num_robots
        robot_onehot[robot_id] = 1.0
        state.extend(robot_onehot)

        if self.use_congestion:
            cong = compute_congestion(pos, self.robot_positions, robot_id)
            density = sum(1 for i in range(self.num_robots)
                         if i != robot_id
                         and manhattan_distance(pos, self.robot_positions[i]) <= cfg.CONGESTION_RADIUS)
            density /= (self.num_robots - 1)
            state.extend([cong, density])

        return np.array(state, dtype=np.float32)

    def _get_stacked_continuous_observation(self, robot_id):
        """Frame-stacked ego-centric observation."""
        raw = self.get_continuous_observation(robot_id)
        if self._raw_obs_size_cont is None:
            self._raw_obs_size_cont = len(raw)
        self._obs_history_cont[robot_id].append(raw)
        frames = list(self._obs_history_cont[robot_id])
        while len(frames) < cfg.FRAME_STACK:
            frames.insert(0, np.zeros(self._raw_obs_size_cont, dtype=np.float32))
        return np.concatenate(frames)

    def _get_all_continuous_observations(self):
        """Get frame-stacked ego-centric observations for all robots."""
        return [self._get_stacked_continuous_observation(i) for i in range(self.num_robots)]

    def step_continuous(self, continuous_actions):
        """Step with continuous actions: list of [linear_vel, angular_vel] per robot.

        Grid mapping:
          angular_vel < -0.3 → turn left 90°
          angular_vel >  0.3 → turn right 90°
          linear_vel  >  0.3 → move 1 cell forward
          linear_vel  < -0.3 → move 1 cell backward
          else → stay in place

        Returns: (observations, rewards, done, info) — same as step()
        """
        # Convert continuous actions to discrete for the grid
        discrete_actions = []
        for rid in range(self.num_robots):
            if self.robot_failed[rid] or self.robot_done[rid]:
                discrete_actions.append(4)  # WAIT
                continue

            lin_vel, ang_vel = continuous_actions[rid]

            # Apply rotation first
            if ang_vel > 0.3:
                self.robot_headings[rid] = (self.robot_headings[rid] + 1) % 4  # turn right
            elif ang_vel < -0.3:
                self.robot_headings[rid] = (self.robot_headings[rid] - 1) % 4  # turn left

            # Then apply linear movement in heading direction
            if lin_vel > 0.3:
                # Move forward
                heading = self.robot_headings[rid]
                dr, dc = HEADING_DELTAS[heading]
                nr, nc = self.robot_positions[rid][0] + dr, self.robot_positions[rid][1] + dc
                # Map heading to discrete action: UP=0, DOWN=1, LEFT=2, RIGHT=3
                heading_to_action = {0: 0, 2: 1, 3: 2, 1: 3}
                discrete_actions.append(heading_to_action[heading])
            elif lin_vel < -0.3:
                # Move backward (opposite heading)
                back_heading = (self.robot_headings[rid] + 2) % 4
                heading_to_action = {0: 0, 2: 1, 3: 2, 1: 3}
                discrete_actions.append(heading_to_action[back_heading])
            else:
                discrete_actions.append(4)  # WAIT

        # Delegate to the existing discrete step
        obs_discrete, rewards, done, info = self.step(discrete_actions)

        # Return continuous (ego-centric) observations instead
        obs_continuous = self._get_all_continuous_observations()
        return obs_continuous, rewards, done, info
