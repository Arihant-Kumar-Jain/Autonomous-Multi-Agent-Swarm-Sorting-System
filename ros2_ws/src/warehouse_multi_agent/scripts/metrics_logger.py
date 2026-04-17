#!/usr/bin/env python3
"""
Metrics Logger Node — tracks and records simulation metrics.

Subscribes to all robot status updates and coordinator,
logs metrics to CSV for post-analysis and comparison plots.

Published Topics:
  /metrics/summary (std_msgs/String) — periodic JSON summary
"""

import json
import os
import time
import csv

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import math

NUM_ROBOTS = 3


class MetricsLogger(Node):
    def __init__(self):
        super().__init__('metrics_logger')

        self.declare_parameter('mode', 'bfs')
        self.mode = self.get_parameter('mode').value

        self.get_logger().info(f'Metrics Logger started | mode={self.mode}')

        # Metrics
        self.start_time = self.get_clock().now()
        self.total_collisions = 0
        self.total_pickups = 0
        self.total_deliveries = 0
        self.near_misses = 0
        self.robot_positions = {}
        self.robot_distances = {i: 0.0 for i in range(NUM_ROBOTS)}
        self.robot_last_pos = {}

        # CSV log
        os.makedirs('results', exist_ok=True)
        self.csv_path = f'results/metrics_{self.mode}_{int(time.time())}.csv'
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'elapsed_s', 'collisions', 'pickups',
            'deliveries', 'near_misses', 'total_distance'
        ])

        # Subscribers
        for i in range(NUM_ROBOTS):
            self.create_subscription(
                Odometry, f'/robot{i}/odom',
                lambda msg, rid=i: self._odom_cb(msg, rid), 10)
            self.create_subscription(
                String, f'/robot{i}/status',
                lambda msg, rid=i: self._status_cb(msg, rid), 10)

        self.create_subscription(
            String, '/coordinator/status', self._coordinator_cb, 10)

        # Publisher
        self.summary_pub = self.create_publisher(String, '/metrics/summary', 10)

        # Timers
        self.create_timer(1.0, self._check_collisions)
        self.create_timer(5.0, self._log_metrics)

    def _odom_cb(self, msg, rid):
        pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        # Track distance traveled
        if rid in self.robot_last_pos:
            last = self.robot_last_pos[rid]
            d = math.sqrt((pos[0]-last[0])**2 + (pos[1]-last[1])**2)
            self.robot_distances[rid] += d
        self.robot_last_pos[rid] = pos
        self.robot_positions[rid] = pos

    def _status_cb(self, msg, rid):
        data = json.loads(msg.data)
        status = data.get('status')
        if status == 'picked_up':
            self.total_pickups += 1
        elif status == 'delivered':
            self.total_deliveries += 1

    def _coordinator_cb(self, msg):
        data = json.loads(msg.data)
        if data.get('status') == 'complete':
            self._log_final()

    def _check_collisions(self):
        """Check for near-misses and collisions between robots."""
        positions = list(self.robot_positions.items())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                rid_a, pos_a = positions[i]
                rid_b, pos_b = positions[j]
                d = math.sqrt((pos_a[0]-pos_b[0])**2 + (pos_a[1]-pos_b[1])**2)
                if d < 0.3:
                    self.total_collisions += 1
                    self.get_logger().warn(
                        f'COLLISION: robot{rid_a} ↔ robot{rid_b} (dist={d:.2f}m)')
                elif d < 0.8:
                    self.near_misses += 1

    def _log_metrics(self):
        """Periodic metrics logging."""
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9
        total_dist = sum(self.robot_distances.values())

        self.csv_writer.writerow([
            time.time(), f'{elapsed:.1f}', self.total_collisions,
            self.total_pickups, self.total_deliveries,
            self.near_misses, f'{total_dist:.1f}'
        ])
        self.csv_file.flush()

        # Publish summary
        summary = {
            'mode': self.mode,
            'elapsed_s': round(elapsed, 1),
            'collisions': self.total_collisions,
            'pickups': self.total_pickups,
            'deliveries': self.total_deliveries,
            'near_misses': self.near_misses,
            'total_distance': round(total_dist, 1),
        }
        msg = String()
        msg.data = json.dumps(summary)
        self.summary_pub.publish(msg)

        self.get_logger().info(
            f'[{self.mode.upper()}] t={elapsed:.0f}s | '
            f'collisions={self.total_collisions} | '
            f'pickups={self.total_pickups} | '
            f'deliveries={self.total_deliveries} | '
            f'dist={total_dist:.1f}m')

    def _log_final(self):
        """Log final results."""
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        total_dist = sum(self.robot_distances.values())
        self.get_logger().info(
            f'\n{"="*50}\n'
            f'  FINAL RESULTS ({self.mode.upper()})\n'
            f'  Time: {elapsed:.1f}s\n'
            f'  Collisions: {self.total_collisions}\n'
            f'  Pickups: {self.total_pickups}\n'
            f'  Deliveries: {self.total_deliveries}\n'
            f'  Near Misses: {self.near_misses}\n'
            f'  Total Distance: {total_dist:.1f}m\n'
            f'{"="*50}')

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
