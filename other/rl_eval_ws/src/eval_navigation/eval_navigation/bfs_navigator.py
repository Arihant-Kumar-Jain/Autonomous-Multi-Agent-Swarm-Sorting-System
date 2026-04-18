#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import json
import numpy as np

class BFSNavigator(Node):
    def __init__(self):
        super().__init__('bfs_navigator')
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        
        self.robot_id = self.get_parameter('robot_id').value
        self.target_x = self.get_parameter('target_x').value
        self.target_y = self.get_parameter('target_y').value
        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.has_odom = False
        self.is_done = False
        
        # Grid parameters for BFS
        # 10m x 10m arena (-5 to +5)
        self.res = 0.25 # 25cm per cell
        self.grid_size = int(10.0 / self.res) 
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Proper Roads Obstacles (3.5x3.5 boxes in corners)
        self._add_obstacle(-3.25, -3.25, 3.5, 3.5) # Top Left
        self._add_obstacle(3.25, -3.25, 3.5, 3.5)  # Top Right
        self._add_obstacle(-3.25, 3.25, 3.5, 3.5)  # Bot Left
        self._add_obstacle(3.25, 3.25, 3.5, 3.5)   # Bot Right
        
        self.path = []
        self.path_planned = False
        
        namespace = f'/robot_{self.robot_id}'
        self.create_subscription(Odometry, f'{namespace}/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, f'{namespace}/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, f'{namespace}/fleet_status', 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
    def _add_obstacle(self, cx, cy, w, h):
        """Add obstacle to grid with inflation padding (0.2m)"""
        pad = 0.2
        w += pad * 2
        h += pad * 2
        min_x = max(-5.0, cx - w/2.0)
        max_x = min(5.0, cx + w/2.0)
        min_y = max(-5.0, cy - h/2.0)
        max_y = min(5.0, cy + h/2.0)
        
        for x in np.arange(min_x, max_x, self.res):
            for y in np.arange(min_y, max_y, self.res):
                gx, gy = self._world_to_grid(x, y)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.grid[gx, gy] = 1

    def _world_to_grid(self, wx, wy):
        gx = int((wx + 5.0) / self.res)
        gy = int((wy + 5.0) / self.res)
        return gx, gy

    def _grid_to_world(self, gx, gy):
        wx = (gx * self.res) - 5.0 + (self.res/2.0)
        wy = (gy * self.res) - 5.0 + (self.res/2.0)
        return wx, wy

    def _plan_bfs(self):
        sx, sy = self._world_to_grid(self.x, self.y)
        gx, gy = self._world_to_grid(self.target_x, self.target_y)
        
        # Clip to bounds
        sx = max(0, min(self.grid_size-1, sx))
        sy = max(0, min(self.grid_size-1, sy))
        gx = max(0, min(self.grid_size-1, gx))
        gy = max(0, min(self.grid_size-1, gy))
        
        if self.grid[sx, sy] == 1 or self.grid[gx, gy] == 1:
            self.get_logger().warning("Start or Goal is inside an obstacle!")
            self.grid[sx, sy] = 0
            self.grid[gx, gy] = 0
            
        queue = [(sx, sy)]
        came_from = {(sx, sy): None}
        found = False
        
        while queue:
            cx, cy = queue.pop(0)
            if cx == gx and cy == gy:
                found = True
                break
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx, ny] == 0 and (nx, ny) not in came_from:
                        queue.append((nx, ny))
                        came_from[(nx, ny)] = (cx, cy)
                        
        if found:
            p = []
            c = (gx, gy)
            while c is not None:
                p.append(c)
                c = came_from[c]
            p.reverse()
            self.path = [self._grid_to_world(px, py) for px, py in p]
            self.get_logger().info(f"BFS Path found: {len(self.path)} waypoints")
        else:
            self.get_logger().error("BFS: NO PATH FOUND")
            self.path = [(self.target_x, self.target_y)]
            
        self.path_planned = True

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_odom = True
        
    def control_loop(self):
        if not self.has_odom: return
        
        if not self.path_planned:
            self._plan_bfs()
            
        dist_to_final = math.hypot(self.target_x - self.x, self.target_y - self.y)
        status_msg = String()
        status_msg.data = json.dumps({'state': 'DONE' if dist_to_final < 0.3 else 'MOVING', 'x': self.x, 'y': self.y})
        self.status_pub.publish(status_msg)
        
        twist = Twist()
        if dist_to_final < 0.3:
            self.cmd_pub.publish(twist)
            self.is_done = True
            return
            
        # Path following logic
        if len(self.path) > 0:
            tx, ty = self.path[0]
            dist_to_wp = math.hypot(tx - self.x, ty - self.y)
            if dist_to_wp < 0.3:
                self.path.pop(0)
                if len(self.path) > 0:
                    tx, ty = self.path[0]
                else:
                    tx, ty = self.target_x, self.target_y
                    
            dx = tx - self.x
            dy = ty - self.y
            target_yaw = math.atan2(dy, dx)
            err_yaw = target_yaw - self.yaw
            while err_yaw > math.pi: err_yaw -= 2*math.pi
            while err_yaw < -math.pi: err_yaw += 2*math.pi
            
            if abs(err_yaw) > 0.3:
                twist.angular.z = 1.0 * err_yaw
                twist.linear.x = 0.0
            else:
                twist.angular.z = 0.5 * err_yaw
                twist.linear.x = 0.5
                
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = BFSNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
