#!/usr/bin/env python3
"""
BFS Navigator Node — rule-based navigation using BFS pathfinding.

Provides a cleaner baseline for comparison:
  - Discretizes continuous space into grid cells
  - Runs BFS on the grid
  - Converts grid path to waypoints
  - Uses proportional control to follow waypoints

Subscribed Topics:
  /<ns>/odom                    (nav_msgs/Odometry)
  /coordinator/assignments      (std_msgs/String)

Published Topics:
  /<ns>/cmd_vel                 (geometry_msgs/Twist)
  /<ns>/status                  (std_msgs/String)
"""

import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ─── Warehouse grid (same as config.py) ─────────────────────────
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

LINEAR_SPEED = 0.2
ANGULAR_SPEED = 1.5
WAYPOINT_TOLERANCE = 0.4  # meters
PICKUP_DISTANCE = 0.5
DELIVERY_DISTANCE = 1.0
NUM_ROBOTS = 3


def bfs_grid(grid, start, goal, blocked=None):
    """BFS on grid. Returns list of (row, col)."""
    if start == goal:
        return [start]
    blocked = blocked or set()
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr][nc] != 1 and (nr, nc) not in blocked:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    if (nr, nc) == goal:
                        path = []
                        node = goal
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]
                    queue.append((nr, nc))
    return []


class BFSNavigator(Node):
    def __init__(self):
        super().__init__('bfs_navigator')

        self.declare_parameter('robot_id', 0)
        self.declare_parameter('robot_name', 'robot0')

        self.robot_id = self.get_parameter('robot_id').value
        self.robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(f'BFS Navigator | robot={self.robot_name} id={self.robot_id}')

        # State
        self.position = None
        self.yaw = 0.0
        self.target = None
        self.task = 'idle'
        self.object_id = -1
        self.carrying = False
        self.other_robots = {}
        self.waypoints = deque()

        # Subs
        self.create_subscription(Odometry, 'odom', self._odom_cb, 10)
        self.create_subscription(String, '/coordinator/assignments', self._assign_cb, 10)
        for i in range(NUM_ROBOTS):
            if i != self.robot_id:
                self.create_subscription(
                    Odometry, f'/robot{i}/odom',
                    lambda msg, rid=i: self._other_odom_cb(msg, rid), 10)

        # Pubs
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'status', 10)

        # Control at 10 Hz
        self.create_timer(0.1, self._control)
        # Replan at 1 Hz
        self.create_timer(1.0, self._replan)

    def _odom_cb(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _other_odom_cb(self, msg, rid):
        self.other_robots[rid] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _assign_cb(self, msg):
        data = json.loads(msg.data)
        my = data.get(str(self.robot_id))
        if my is None:
            return
        new_task = my.get('task', 'idle')
        new_target = my.get('target')
        if new_task != self.task or (new_target and self.target and
                                      tuple(new_target) != self.target):
            self.task = new_task
            self.target = tuple(new_target) if new_target else None
            self.object_id = my.get('object_id', -1)
            self.waypoints.clear()
            self._replan()

    def _pos_to_grid(self, pos):
        """Convert continuous position to grid cell."""
        return (int(round(pos[1])), int(round(pos[0])))  # (row, col) = (y, x)

    def _grid_to_pos(self, cell):
        """Convert grid cell to continuous position."""
        return (float(cell[1]), float(cell[0]))  # (x, y) = (col, row)

    def _replan(self):
        """Recompute BFS path."""
        if self.position is None or self.target is None or self.task == 'idle':
            return

        start = self._pos_to_grid(self.position)
        goal = self._pos_to_grid(self.target)

        # Block cells occupied by other robots
        blocked = set()
        for rid, opos in self.other_robots.items():
            blocked.add(self._pos_to_grid(opos))

        path = bfs_grid(WAREHOUSE_MAP, start, goal, blocked)
        self.waypoints = deque()
        for cell in path[1:]:  # skip current cell
            self.waypoints.append(self._grid_to_pos(cell))

    def _control(self):
        if self.position is None or self.task == 'idle':
            self.cmd_pub.publish(Twist())
            return

        # Check arrival
        if self.target:
            d = self._dist(self.position, self.target)
            if self.task == 'pickup' and d < PICKUP_DISTANCE:
                self.cmd_pub.publish(Twist())
                self.carrying = True
                self._pub_status('picked_up', self.object_id)
                self.task = 'idle'
                return
            if self.task == 'deliver' and d < DELIVERY_DISTANCE:
                self.cmd_pub.publish(Twist())
                self.carrying = False
                self._pub_status('delivered', self.object_id)
                self.task = 'idle'
                return

        # Collision check
        for rid, opos in self.other_robots.items():
            if self._dist(self.position, opos) < 0.5 and self.robot_id > rid:
                self.cmd_pub.publish(Twist())
                return

        # Follow waypoints
        if not self.waypoints:
            if self.target:
                self._go_to(self.target)
            return

        wp = self.waypoints[0]
        if self._dist(self.position, wp) < WAYPOINT_TOLERANCE:
            self.waypoints.popleft()
            if not self.waypoints:
                return
            wp = self.waypoints[0]

        self._go_to(wp)

    def _go_to(self, target):
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        angle = math.atan2(dy, dx)
        diff = self._norm_angle(angle - self.yaw)
        cmd = Twist()
        if abs(diff) > 0.3:
            cmd.angular.z = ANGULAR_SPEED if diff > 0 else -ANGULAR_SPEED
            cmd.linear.x = 0.05
        else:
            cmd.linear.x = min(LINEAR_SPEED, self._dist(self.position, target) * 0.5)
            cmd.angular.z = diff * 2.0
        self.cmd_pub.publish(cmd)

    def _pub_status(self, status, obj_id=-1):
        msg = String()
        msg.data = json.dumps({'status': status, 'object_id': obj_id})
        self.status_pub.publish(msg)

    @staticmethod
    def _dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    @staticmethod
    def _norm_angle(a):
        while a > math.pi: a -= 2*math.pi
        while a < -math.pi: a += 2*math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = BFSNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
