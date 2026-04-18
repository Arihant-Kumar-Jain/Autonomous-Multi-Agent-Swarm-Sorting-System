#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import json
import numpy as np
import os
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNNavigator(Node):
    def __init__(self):
        super().__init__('dqn_navigator')
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
        self.fov = 11
        self.device = torch.device('cpu')
        
        # Load DQN
        model_path = f'/home/aman/cs671_7/rl_cleaning_project/rl_training/models/dqn_R{self.robot_id+1}.pth'
        self.model = DQN(self.fov*self.fov + 2, 5).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            self.get_logger().error(f"DQN not found at {model_path}")
            
        namespace = f'/robot_{self.robot_id}'
        self.create_subscription(Odometry, f'{namespace}/odom', self.odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, f'{namespace}/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, f'{namespace}/fleet_status', 10)
        
        self.timer = self.create_timer(0.1, self.control_loop)
        
    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_odom = True
        
    def get_action(self, dx, dy):
        norm = math.sqrt(dx*dx + dy*dy) + 1e-5
        dx /= norm
        dy /= norm
        local_grid = np.zeros((self.fov, self.fov))
        flat_state = np.concatenate([local_grid.flatten(), [dy, dx]])
        s_tensor = torch.FloatTensor(flat_state).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.model(s_tensor)
        return torch.argmax(q_vals).item()
        
    def control_loop(self):
        if not self.has_odom:
            return
            
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        status_msg = String()
        status_msg.data = json.dumps({'state': 'DONE' if dist < 0.2 else 'MOVING', 'x': self.x, 'y': self.y})
        self.status_pub.publish(status_msg)
        
        twist = Twist()
        if dist < 0.2:
            self.cmd_pub.publish(twist)
            self.is_done = True
            return
            
        action = self.get_action(dx, dy)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        d_ax, d_ay = moves[action]
        
        t_x = self.x + d_ay * 0.5
        t_y = self.y + d_ax * 0.5
        
        t_dx = t_x - self.x
        t_dy = t_y - self.y
        target_yaw = math.atan2(t_dy, t_dx)
        
        err_yaw = target_yaw - self.yaw
        while err_yaw > math.pi: err_yaw -= 2*math.pi
        while err_yaw < -math.pi: err_yaw += 2*math.pi
        
        if abs(err_yaw) > 0.2:
            twist.angular.z = 2.0 * err_yaw
            twist.linear.x = 0.0
        else:
            twist.angular.z = 0.0
            twist.linear.x = 0.5
            
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = DQNNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
