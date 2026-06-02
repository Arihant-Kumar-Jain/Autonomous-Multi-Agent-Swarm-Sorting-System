#!/usr/bin/env python3
"""
Task Coordinator Node — central coordination for multi-agent warehouse.

Responsibilities:
  - Tracks all robot positions via /robotX/odom
  - Maintains global object list
  - Assigns objects to robots (greedy + congestion-aware)
  - Publishes task assignments to each robot
  - Handles failure detection and reassignment

Published Topics:
  /coordinator/assignments   (std_msgs/String)  — JSON task assignments
  /coordinator/status        (std_msgs/String)  — system status

Subscribed Topics:
  /robotX/odom               (nav_msgs/Odometry) — robot positions
  /robotX/status             (std_msgs/String)   — robot status updates
"""

import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry


# ─── Configuration (matches grid config.py) ────────────────────
OBJECT_POSITIONS = [
    (4.0, 2.0), (10.0, 5.0), (4.0, 8.0), (7.0, 4.0), (1.0, 10.0)
]
DROP_ZONE_CENTER = (7.5, 13.0)
CONGESTION_RADIUS = 3.0
CONGESTION_WEIGHT = 0.5
NUM_ROBOTS = 3
REASSIGN_TIMEOUT = 30.0  # seconds without progress → assume failure


class TaskCoordinator(Node):
    def __init__(self):
        super().__init__('task_coordinator')

        # Parameters
        self.declare_parameter('num_robots', NUM_ROBOTS)
        self.declare_parameter('mode', 'bfs')
        self.num_robots = self.get_parameter('num_robots').value
        self.mode = self.get_parameter('mode').value
        self.use_congestion = (self.mode == 'improved_rl')

        self.get_logger().info(f'Task Coordinator started | mode={self.mode} | robots={self.num_robots}')

        # State
        self.robot_positions = {i: None for i in range(self.num_robots)}
        self.robot_carrying = {i: False for i in range(self.num_robots)}
        self.robot_status = {i: 'idle' for i in range(self.num_robots)}
        self.robot_last_progress = {i: self.get_clock().now() for i in range(self.num_robots)}

        self.objects_available = list(range(len(OBJECT_POSITIONS)))
        self.objects_collected = set()
        self.objects_delivered = set()
        self.assignments = {}  # {robot_id: object_index}

        # Subscribers — robot odometry
        self.odom_subs = []
        for i in range(self.num_robots):
            sub = self.create_subscription(
                Odometry,
                f'/robot{i}/odom',
                lambda msg, rid=i: self._odom_callback(msg, rid),
                10
            )
            self.odom_subs.append(sub)

        # Subscribers — robot status
        self.status_subs = []
        for i in range(self.num_robots):
            sub = self.create_subscription(
                String,
                f'/robot{i}/status',
                lambda msg, rid=i: self._status_callback(msg, rid),
                10
            )
            self.status_subs.append(sub)

        # Publishers
        self.assignment_pub = self.create_publisher(String, '/coordinator/assignments', 10)
        self.system_status_pub = self.create_publisher(String, '/coordinator/status', 10)

        # Timer — periodic task allocation (2 Hz)
        self.timer = self.create_timer(0.5, self._coordination_loop)

        self.get_logger().info(f'Objects: {len(OBJECT_POSITIONS)} | Drop zone: {DROP_ZONE_CENTER}')

    def _odom_callback(self, msg, robot_id):
        """Update robot position from odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_positions[robot_id] = (x, y)

    def _status_callback(self, msg, robot_id):
        """Handle status updates from robots."""
        data = json.loads(msg.data)
        status = data.get('status', 'idle')
        old_status = self.robot_status[robot_id]
        self.robot_status[robot_id] = status

        if status != old_status:
            self.robot_last_progress[robot_id] = self.get_clock().now()

        if status == 'picked_up':
            obj_idx = data.get('object_id')
            if obj_idx is not None:
                self.objects_collected.add(obj_idx)
                self.robot_carrying[robot_id] = True
                if robot_id in self.assignments:
                    del self.assignments[robot_id]
                self.get_logger().info(f'Robot {robot_id} picked up object {obj_idx}')

        elif status == 'delivered':
            obj_idx = data.get('object_id')
            if obj_idx is not None:
                self.objects_delivered.add(obj_idx)
            self.robot_carrying[robot_id] = False
            self.robot_status[robot_id] = 'idle'
            self.get_logger().info(f'Robot {robot_id} delivered object')

        elif status == 'failed':
            self._handle_failure(robot_id)

    def _coordination_loop(self):
        """Main coordination loop — allocate tasks, detect failures, publish."""
        # Check if all robots have positions
        if any(pos is None for pos in self.robot_positions.values()):
            return

        # Detect stuck robots (failure)
        now = self.get_clock().now()
        for rid in range(self.num_robots):
            if self.robot_status[rid] not in ('idle', 'done', 'failed'):
                elapsed = (now - self.robot_last_progress[rid]).nanoseconds / 1e9
                if elapsed > REASSIGN_TIMEOUT:
                    self.get_logger().warn(f'Robot {rid} stuck for {elapsed:.0f}s — treating as failure')
                    self._handle_failure(rid)

        # Allocate tasks to idle robots
        self._allocate_tasks()

        # Publish assignments
        self._publish_assignments()

        # Check completion
        if len(self.objects_delivered) == len(OBJECT_POSITIONS):
            self.get_logger().info('🎉 ALL OBJECTS DELIVERED — TASK COMPLETE!')
            status_msg = String()
            status_msg.data = json.dumps({'status': 'complete', 'deliveries': len(self.objects_delivered)})
            self.system_status_pub.publish(status_msg)

    def _allocate_tasks(self):
        """Assign available objects to free robots."""
        # Available objects (not collected, not assigned)
        assigned_objects = set(self.assignments.values())
        available = [idx for idx in range(len(OBJECT_POSITIONS))
                     if idx not in self.objects_collected and idx not in assigned_objects]

        # Free robots
        free_robots = [rid for rid in range(self.num_robots)
                       if (self.robot_status[rid] == 'idle'
                           and not self.robot_carrying[rid]
                           and rid not in self.assignments)]

        if not available or not free_robots:
            return

        # Compute costs and assign
        for obj_idx in available:
            if not free_robots:
                break

            obj_pos = OBJECT_POSITIONS[obj_idx]
            best_rid = None
            best_cost = float('inf')

            for rid in free_robots:
                rpos = self.robot_positions[rid]
                if rpos is None:
                    continue
                dist = math.sqrt((rpos[0] - obj_pos[0])**2 + (rpos[1] - obj_pos[1])**2)
                cong = 0.0
                if self.use_congestion:
                    cong = self._compute_congestion(rid)
                cost = dist + CONGESTION_WEIGHT * cong

                if cost < best_cost:
                    best_cost = cost
                    best_rid = rid

            if best_rid is not None:
                self.assignments[best_rid] = obj_idx
                free_robots.remove(best_rid)
                self.get_logger().info(
                    f'Assigned object {obj_idx} at {obj_pos} to robot {best_rid} (cost={best_cost:.2f})')

    def _compute_congestion(self, robot_id):
        """Compute congestion around a robot."""
        rpos = self.robot_positions[robot_id]
        if rpos is None:
            return 0.0
        congestion = 0.0
        for i in range(self.num_robots):
            if i == robot_id:
                continue
            opos = self.robot_positions[i]
            if opos is None:
                continue
            dist = math.sqrt((rpos[0] - opos[0])**2 + (rpos[1] - opos[1])**2)
            if dist <= CONGESTION_RADIUS:
                congestion += 1.0 / max(dist, 0.1)
        return congestion

    def _handle_failure(self, robot_id):
        """Handle robot failure — reassign its task."""
        self.get_logger().warn(f'Handling failure for robot {robot_id}')
        self.robot_status[robot_id] = 'failed'

        if robot_id in self.assignments:
            obj_idx = self.assignments.pop(robot_id)
            self.get_logger().info(f'Object {obj_idx} freed for reassignment')
            # Will be picked up next cycle

    def _publish_assignments(self):
        """Publish current assignments and robot targets."""
        assignments_data = {}
        for rid in range(self.num_robots):
            if self.robot_carrying[rid]:
                # Navigate to drop zone
                assignments_data[str(rid)] = {
                    'target': list(DROP_ZONE_CENTER),
                    'task': 'deliver',
                    'object_id': -1,
                }
            elif rid in self.assignments:
                obj_idx = self.assignments[rid]
                obj_pos = OBJECT_POSITIONS[obj_idx]
                assignments_data[str(rid)] = {
                    'target': list(obj_pos),
                    'task': 'pickup',
                    'object_id': obj_idx,
                }
            else:
                assignments_data[str(rid)] = {
                    'target': None,
                    'task': 'idle',
                    'object_id': -1,
                }

        msg = String()
        msg.data = json.dumps(assignments_data)
        self.assignment_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TaskCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
