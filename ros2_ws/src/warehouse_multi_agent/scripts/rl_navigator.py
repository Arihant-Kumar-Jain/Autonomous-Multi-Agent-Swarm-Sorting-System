#!/usr/bin/env python3
"""
RL / BFS Navigator Node — per-robot navigation controller.

This node is the BRIDGE between:
  - Trained RL policy (from Pygame grid training) → Gazebo robot control
  - OR BFS pathfinding → Gazebo robot control

It translates:
  - Continuous odom → normalized state vector (same format as training)
  - Discrete action (UP/DOWN/LEFT/RIGHT/WAIT) → /cmd_vel Twist

Subscribed Topics:
  /<ns>/odom                    (nav_msgs/Odometry)
  /coordinator/assignments      (std_msgs/String)

Published Topics:
  /<ns>/cmd_vel                 (geometry_msgs/Twist)
  /<ns>/status                  (std_msgs/String)
"""

import json
import math
import os
import sys
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ─── Grid configuration (must match training config) ────────────
GRID_ROWS = 15
GRID_COLS = 15
NUM_ROBOTS = 3
PICKUP_DISTANCE = 0.5   # meters — close enough to "pick up"
DELIVERY_DISTANCE = 1.0  # meters — close enough to drop zone
LINEAR_SPEED = 0.22      # m/s (TurtleBot3 burger max ~0.22)
ANGULAR_SPEED = 1.5      # rad/s
CONGESTION_RADIUS = 3.0

# Known warehouse map for wall detection (matches training grid)
WAREHOUSE_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,1,1,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,2,2,2,0,0,0,0,0,1],
    [1,0,0,0,0,0,2,2,2,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

# Action mapping: discrete → velocity
ACTION_MAP = {
    0: (LINEAR_SPEED, 0.0),     # UP    → forward
    1: (-LINEAR_SPEED, 0.0),    # DOWN  → backward
    2: (0.0, ANGULAR_SPEED),    # LEFT  → turn left
    3: (0.0, -ANGULAR_SPEED),   # RIGHT → turn right
    4: (0.0, 0.0),              # WAIT  → stop
}


class RLNavigator(Node):
    def __init__(self):
        super().__init__('rl_navigator')

        # Parameters
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('robot_name', 'robot0')
        self.declare_parameter('mode', 'bfs')
        self.declare_parameter('model_path', '')
        self.declare_parameter('grid_scale', 1.0)

        self.robot_id = self.get_parameter('robot_id').value
        self.robot_name = self.get_parameter('robot_name').value
        self.mode = self.get_parameter('mode').value
        self.model_path = self.get_parameter('model_path').value
        self.grid_scale = self.get_parameter('grid_scale').value

        self.get_logger().info(
            f'Navigator started | robot={self.robot_name} id={self.robot_id} mode={self.mode}')

        # State
        self.position = None    # (x, y)
        self.yaw = 0.0
        self.target = None      # (x, y)
        self.task = 'idle'
        self.object_id = -1
        self.carrying = False
        self.other_robots = {}  # {robot_id: (x, y)}

        # RL model (loaded if mode is rl/improved_rl)
        self.rl_model = None
        self.use_congestion = (self.mode == 'improved_rl')
        if self.mode in ('rl', 'improved_rl') and self.model_path:
            self._load_model()

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, 10)
        self.assignment_sub = self.create_subscription(
            String, '/coordinator/assignments', self._assignment_callback, 10)

        # Subscribe to other robots' odom for collision avoidance
        for i in range(NUM_ROBOTS):
            if i != self.robot_id:
                self.create_subscription(
                    Odometry, f'/robot{i}/odom',
                    lambda msg, rid=i: self._other_odom_callback(msg, rid), 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'status', 10)

        # Control timer (10 Hz)
        self.timer = self.create_timer(0.1, self._control_loop)

        # BFS path (for BFS mode)
        self._bfs_waypoints = deque()

    def _load_model(self):
        """Load trained DQN model."""
        try:
            import torch
            # Add path to find dqn_agent module
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, project_dir)

            from dqn_agent import DQNAgent
            state_size = 14 if self.use_congestion else 12
            self.rl_model = DQNAgent(state_size=state_size)

            if os.path.exists(self.model_path):
                self.rl_model.load(self.model_path)
                self.get_logger().info(f'Loaded RL model from {self.model_path}')
            else:
                self.get_logger().warn(f'Model not found at {self.model_path}, using random policy')
        except ImportError as e:
            self.get_logger().error(f'Cannot load RL model: {e}')

    def _odom_callback(self, msg):
        """Update own position and orientation."""
        self.position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _other_odom_callback(self, msg, robot_id):
        """Track other robots' positions for collision avoidance."""
        self.other_robots[robot_id] = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def _assignment_callback(self, msg):
        """Receive task assignment from coordinator."""
        data = json.loads(msg.data)
        my_assignment = data.get(str(self.robot_id))

        if my_assignment is None:
            return

        new_task = my_assignment.get('task', 'idle')
        new_target = my_assignment.get('target')
        new_obj_id = my_assignment.get('object_id', -1)

        if new_target is not None:
            self.target = tuple(new_target)
        else:
            self.target = None

        if new_task != self.task:
            self.task = new_task
            self.object_id = new_obj_id
            self._bfs_waypoints.clear()  # replan on task change

    def _control_loop(self):
        """Main control loop — compute action, send velocity."""
        if self.position is None or self.target is None or self.task == 'idle':
            self._stop()
            return

        # Check if reached target
        dist_to_target = self._distance(self.position, self.target)

        if self.task == 'pickup' and dist_to_target < PICKUP_DISTANCE:
            self._stop()
            self.carrying = True
            self._publish_status('picked_up', self.object_id)
            self.task = 'idle'  # wait for new assignment
            return

        if self.task == 'deliver' and dist_to_target < DELIVERY_DISTANCE:
            self._stop()
            self.carrying = False
            self._publish_status('delivered', self.object_id)
            self.task = 'idle'
            return

        # Collision avoidance — check if any robot is too close ahead
        if self._collision_risk():
            self._stop()
            return

        # Compute action based on mode
        if self.mode == 'bfs':
            self._bfs_navigate()
        elif self.rl_model is not None:
            self._rl_navigate()
        else:
            # Fallback: simple proportional control
            self._proportional_navigate()

    def _bfs_navigate(self):
        """Navigate using simple proportional control toward target."""
        # For Gazebo, BFS on continuous space = go-to-goal with obstacle avoidance
        self._proportional_navigate()

    def _rl_navigate(self):
        """Navigate using trained RL policy."""
        state = self._build_state()
        action = self.rl_model.select_action(state, training=False)
        self._execute_action(action)

    def _proportional_navigate(self):
        """Simple proportional navigation toward target."""
        if self.target is None:
            self._stop()
            return

        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        target_angle = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - self.yaw)

        cmd = Twist()

        if abs(angle_diff) > 0.3:
            # Turn toward target
            cmd.angular.z = ANGULAR_SPEED if angle_diff > 0 else -ANGULAR_SPEED
            cmd.linear.x = 0.05  # slow forward while turning
        else:
            # Go straight
            cmd.linear.x = min(LINEAR_SPEED, self._distance(self.position, self.target) * 0.5)
            cmd.angular.z = angle_diff * 2.0  # proportional heading correction

        self.cmd_pub.publish(cmd)

    def _build_state(self):
        """Build state vector matching training format."""
        if self.position is None or self.target is None:
            return np.zeros(14 if self.use_congestion else 12, dtype=np.float32)

        # Direction to target (normalized)
        dr = (self.target[1] - self.position[1]) / GRID_ROWS
        dc = (self.target[0] - self.position[0]) / GRID_COLS
        dist = self._distance(self.position, self.target) / (GRID_ROWS + GRID_COLS)
        carrying = 1.0 if self.carrying else 0.0

        # Other robot relative positions
        other_dists = []
        sorted_others = sorted(self.other_robots.keys())
        for rid in sorted_others[:2]:  # max 2 other robots
            opos = self.other_robots[rid]
            other_dists.append((opos[1] - self.position[1]) / GRID_ROWS)
            other_dists.append((opos[0] - self.position[0]) / GRID_COLS)
        while len(other_dists) < 4:
            other_dists.append(0.0)

        # Simple obstacle sensors (walls from map + nearby robots)
        obstacles = [0.0, 0.0, 0.0, 0.0]  # up, down, left, right
        # Check walls from known map
        grid_r = int(round(self.position[1]))
        grid_c = int(round(self.position[0]))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for i, (ddr, ddc) in enumerate(directions):
            nr, nc = grid_r + ddr, grid_c + ddc
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                if WAREHOUSE_MAP[nr][nc] == 1:  # WALL
                    obstacles[i] = 1.0
            else:
                obstacles[i] = 1.0  # out of bounds
        # Also check nearby robots
        for rid, opos in self.other_robots.items():
            rdist = self._distance(self.position, opos)
            if rdist < 1.0:
                dx = opos[0] - self.position[0]
                dy = opos[1] - self.position[1]
                if abs(dy) > abs(dx):
                    if dy > 0:
                        obstacles[1] = 1.0  # down
                    else:
                        obstacles[0] = 1.0  # up
                else:
                    if dx > 0:
                        obstacles[3] = 1.0  # right
                    else:
                        obstacles[2] = 1.0  # left

        state = [dr, dc, dist, carrying] + other_dists + obstacles

        # Robot identity (one-hot, matches training)
        robot_onehot = [0.0] * NUM_ROBOTS
        robot_onehot[self.robot_id] = 1.0
        state.extend(robot_onehot)

        if self.use_congestion:
            cong = self._compute_congestion()
            density = sum(1 for opos in self.other_robots.values()
                         if self._distance(self.position, opos) <= CONGESTION_RADIUS)
            density /= max(len(self.other_robots), 1)
            state.extend([cong, density])

        return np.array(state, dtype=np.float32)

    def _execute_action(self, action):
        """Convert discrete action to Twist and publish."""
        # Map discrete action to desired direction, then use proportional control
        cmd = Twist()

        if action == 4:  # WAIT
            self.cmd_pub.publish(cmd)
            return

        # Use proportional navigation but bias toward the RL-suggested direction
        linear, angular = ACTION_MAP[action]
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)

    def _collision_risk(self):
        """Check if another robot is dangerously close."""
        if not self.other_robots:
            return False

        for rid, opos in self.other_robots.items():
            dist = self._distance(self.position, opos)
            if dist < 0.5:  # 0.5m = danger zone
                # Priority: lower ID has priority
                if self.robot_id > rid:
                    return True  # we yield
        return False

    def _compute_congestion(self):
        """Compute local congestion."""
        cong = 0.0
        for opos in self.other_robots.values():
            dist = self._distance(self.position, opos)
            if dist <= CONGESTION_RADIUS:
                cong += 1.0 / max(dist, 0.1)
        return cong

    def _stop(self):
        """Publish zero velocity."""
        self.cmd_pub.publish(Twist())

    def _publish_status(self, status, object_id=-1):
        """Publish robot status."""
        msg = String()
        msg.data = json.dumps({'status': status, 'object_id': object_id})
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status} | object: {object_id}')

    @staticmethod
    def _distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
