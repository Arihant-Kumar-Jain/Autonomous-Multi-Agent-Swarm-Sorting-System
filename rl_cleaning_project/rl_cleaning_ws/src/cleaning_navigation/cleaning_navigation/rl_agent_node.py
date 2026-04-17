import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import math
import numpy as np
import torch
import torch.nn as nn
import json

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

class RLAgentNode(Node):
    def __init__(self):
        super().__init__('rl_agent_node')
        
        self.declare_parameter('robot_id', 0)
        self.declare_parameter('model_path', '')
        
        self.robot_id = self.get_parameter('robot_id').value
        self.model_path = self.get_parameter('model_path').value
        
        self.grid_size = 50
        self.cell_size = 0.2
        self.fov = 11
        
        self.device = torch.device('cpu')
        
        self.is_q_table = self.model_path.endswith('.json')
        if self.is_q_table:
            with open(self.model_path, 'r') as f:
                self.q_table = json.load(f)
        else:
            self.model = DQN(self.fov*self.fov + 2, 5).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        self.target_x = None
        self.target_y = None
        
        namespace = f'/robot_{self.robot_id}'
        self.odom_sub = self.create_subscription(Odometry, f'{namespace}/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, f'{namespace}/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f'RL Agent {self.robot_id} started.')

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def _get_grid_pos(self):
        c = int((self.x - self.cell_size/2.0) / self.cell_size + self.grid_size/2.0)
        r = int(self.grid_size/2.0 - (self.y + self.cell_size/2.0) / self.cell_size)
        return r, c

    def _get_physical_pos(self, r, c):
        x = (c - self.grid_size/2.0) * self.cell_size + self.cell_size/2.0
        y = (self.grid_size/2.0 - r) * self.cell_size - self.cell_size/2.0
        return x, y

    def _get_rl_action(self):
        r, c = self._get_grid_pos()
        dx, dy = 0.0 - self.x, 0.0 - self.y
        norm = math.sqrt(dx*dx + dy*dy) + 1e-5
        
        if self.is_q_table:
            # Tabular Q state extraction
            angle = math.atan2(dy, dx)
            dirt_dir = int(np.round(angle / (np.pi/4))) % 8 + 1
            dist = abs(0.0 - self.x) + abs(0.0 - self.y)
            if dist <= 1.0: dirt_dist = 1
            elif dist <= 3.0: dirt_dist = 2
            else: dirt_dist = 3
            
            # Dummy nearest robot (assuming no robots nearby for testing single agent behavior)
            rob_dir = 0
            rob_dist = 0
            
            state_key = str((dirt_dir, dirt_dist, rob_dir, rob_dist))
            if state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            else:
                return 4 # stay
        else:
            local_grid = np.zeros((self.fov, self.fov))
            dx /= norm
            dy /= norm
            
            flat_state = np.concatenate([local_grid.flatten(), [dy, dx]])
            s_tensor = torch.FloatTensor(flat_state).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.model(s_tensor)
            return torch.argmax(q_vals).item()

    def control_loop(self):
        if self.target_x is None:
            action = self._get_rl_action()
            r, c = self._get_grid_pos()
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
            dr, dc = moves[action]
            nr, nc = r + dr, c + dc
            self.target_x, self.target_y = self._get_physical_pos(nr, nc)
            
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < 0.05:
            self.target_x = None
            twist = Twist()
            self.cmd_pub.publish(twist)
            return
            
        target_yaw = math.atan2(dy, dx)
        err_yaw = target_yaw - self.yaw
        while err_yaw > math.pi: err_yaw -= 2*math.pi
        while err_yaw < -math.pi: err_yaw += 2*math.pi
        
        twist = Twist()
        if abs(err_yaw) > 0.2:
            twist.angular.z = 2.0 * err_yaw
            twist.linear.x = 0.0
        else:
            twist.angular.z = 1.0 * err_yaw
            twist.linear.x = 0.5
            
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RLAgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
