"""
Multi-Agent Warehouse Environment.

Gym-like interface for 3 robots in a warehouse grid.
Supports BFS baseline, RL, and improved RL (congestion-aware) modes.
"""

import copy
import random
import numpy as np
import config as cfg
from pathfinding import bfs, manhattan_distance
from task_allocator import allocate_tasks_greedy, compute_congestion, reallocate_on_failure


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
        self.objects_collected = [False] * len(self.objects)
        self.objects_delivered = [False] * len(self.objects)
        self.assignments = {}
        self.bfs_paths = {}
        self.step_count = 0
        self.total_collisions = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.history = []
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

        if target is None:
            target = cfg.DROP_ZONE_CENTER

        # Direction to target (normalized)
        dr = (target[0] - pos[0]) / max(self.rows, 1)
        dc = (target[1] - pos[1]) / max(self.cols, 1)
        dist = manhattan_distance(pos, target) / (self.rows + self.cols)

        carrying = 1.0 if self.robot_carrying[robot_id] else 0.0

        # Distance to other robots
        other_dists = []
        for i in range(self.num_robots):
            if i == robot_id:
                continue
            opos = self.robot_positions[i]
            other_dists.append((opos[0] - pos[0]) / self.rows)
            other_dists.append((opos[1] - pos[1]) / self.cols)

        # Obstacle sensors (4 directions)
        obstacles = []
        for action_id in range(4):  # UP, DOWN, LEFT, RIGHT
            ddr, ddc = cfg.ACTIONS[action_id]
            nr, nc = pos[0] + ddr, pos[1] + ddc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                blocked = 1.0 if self.grid[nr][nc] == cfg.WALL else 0.0
                # Also check if another robot is there
                for i in range(self.num_robots):
                    if i != robot_id and self.robot_positions[i] == (nr, nc):
                        blocked = 1.0
                obstacles.append(blocked)
            else:
                obstacles.append(1.0)  # out of bounds = wall

        state = [dr, dc, dist, carrying] + other_dists + obstacles

        # Robot identity (one-hot)
        robot_onehot = [0.0] * self.num_robots
        robot_onehot[robot_id] = 1.0
        state.extend(robot_onehot)

        if self.use_congestion:
            cong = compute_congestion(pos, self.robot_positions, robot_id)
            # Local density: robots within radius
            density = sum(1 for i in range(self.num_robots)
                         if i != robot_id
                         and manhattan_distance(pos, self.robot_positions[i]) <= cfg.CONGESTION_RADIUS)
            density /= (self.num_robots - 1)
            state.extend([cong, density])

        return np.array(state, dtype=np.float32)

    def _get_all_observations(self):
        """Get observations for all robots."""
        return [self.get_observation(i) for i in range(self.num_robots)]

    def _get_current_target(self, robot_id):
        """Get current navigation target for robot."""
        if self.robot_carrying[robot_id]:
            return cfg.DROP_ZONE_CENTER
        if robot_id in self.assignments:
            obj_idx = self.assignments[robot_id]
            if not self.objects_collected[obj_idx]:
                return self.objects[obj_idx]
        return None

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
            if not self.objects_collected[idx] and idx not in assigned_obj_indices:
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
            return 4  # WAIT

        next_pos = self.bfs_paths[robot_id][0]
        curr_pos = self.robot_positions[robot_id]
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
        info = {"collisions": 0, "pickups": 0, "deliveries": 0, "failures": 0}

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
        """Get observation size."""
        base = 15  # 12 original + 3 robot_id one-hot
        if self.use_congestion:
            base = 17  # + congestion + density
        return base
