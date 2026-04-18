#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.msg import ContactsState
import time
import json
import os

class MetricsNode(Node):
    def __init__(self):
        super().__init__('metrics_node')
        self.declare_parameter('algo', 'unknown')
        self.algo = self.get_parameter('algo').value
        
        self.start_time = time.time()
        self.robots = ['robot_0', 'robot_1', 'robot_2']
        self.collisions = {r: 0 for r in self.robots}
        self.in_collision = {r: False for r in self.robots}
        self.task_status = {r: {'state': 'UNKNOWN', 'end': None} for r in self.robots}
        
        # Spatial Tracking
        self.trajectories = {r: [] for r in self.robots}
        self.collision_points = []  # List of (x, y) coordinates
        self.current_poses = {r: (0.0, 0.0) for r in self.robots}
        
        self._subs = []
        for r in self.robots:
            sub = self.create_subscription(ContactsState, f'/{r}/bumper_states', 
                                           lambda msg, name=r: self._col_cb(msg, name), 10)
            self._subs.append(sub)
            
            sub = self.create_subscription(String, f'/{r}/fleet_status',
                                           lambda msg, name=r: self._stat_cb(msg, name), 10)
            self._subs.append(sub)
            
        self.create_timer(0.5, self._track_trajectories)
        self.create_timer(1.0, self._check_done)
        
    def _col_cb(self, msg, name):
        valid = [s for s in msg.states if 'floor' not in s.collision2_name and 'ground' not in s.collision2_name]
        is_col = len(valid) > 0
        if is_col and not self.in_collision[name]:
            self.collisions[name] += 1
            self.in_collision[name] = True
            # Log spatial collision point
            self.collision_points.append(self.current_poses[name])
            self.get_logger().info(f"Collision on {name} at {self.current_poses[name]}")
        elif not is_col:
            self.in_collision[name] = False
            
    def _stat_cb(self, msg, name):
        data = json.loads(msg.data)
        state = data.get('state', 'UNKNOWN')
        x = data.get('x', 0.0)
        y = data.get('y', 0.0)
        self.current_poses[name] = (x, y)
        
        if state == 'DONE' and self.task_status[name]['state'] != 'DONE':
            self.task_status[name]['end'] = time.time()
            self.get_logger().info(f"{name} finished!")
        self.task_status[name]['state'] = state
        
    def _track_trajectories(self):
        t = time.time() - self.start_time
        for r in self.robots:
            if self.task_status[r]['state'] != 'DONE':
                self.trajectories[r].append((t, self.current_poses[r][0], self.current_poses[r][1]))

    def _check_done(self):
        all_done = all(self.task_status[r]['state'] == 'DONE' for r in self.robots)
        # Timeout safety (e.g. 60 seconds)
        if all_done or (time.time() - self.start_time > 60.0):
            elapsed = time.time() - self.start_time
            metrics = {
                'algo': self.algo,
                'elapsed': elapsed,
                'collisions': sum(self.collisions.values()),
                'trajectories': self.trajectories,
                'collision_points': self.collision_points
            }
            res_dir = '/home/aman/cs671_7/rl_eval_ws/results'
            os.makedirs(res_dir, exist_ok=True)
            with open(f'{res_dir}/metrics_{self.algo}.json', 'w') as f:
                json.dump(metrics, f)
            self.get_logger().info(f"EVAL DONE in {elapsed:.2f}s. Saved spatial metrics.")
            import sys
            sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
