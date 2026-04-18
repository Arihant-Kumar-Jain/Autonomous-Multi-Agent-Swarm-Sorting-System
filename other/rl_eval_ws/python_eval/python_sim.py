#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import json
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.fc(x)

class Robot:
    def __init__(self, r_id, sx, sy, tx, ty, algo, model_path=None):
        self.id = r_id
        self.x = sx
        self.y = sy
        self.yaw = 0.0
        self.target_x = tx
        self.target_y = ty
        self.algo = algo
        
        self.grid_target_x = None
        self.grid_target_y = None
        
        self.is_done = False
        self.radius = 0.3
        self.history = [(self.x, self.y)]
        self.collisions = 0
        self.collision_points = []
        
        self.fov = 11
        self.path = []
        self.yield_timer = 0.0
        
        if algo == 'q':
            self.q_table = {}
            if os.path.exists(model_path):
                with open(model_path, 'r') as f:
                    self.q_table = json.load(f)
        elif algo == 'dqn':
            self.dqn = DQN(self.fov*self.fov + 2, 5)
            if os.path.exists(model_path):
                self.dqn.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.dqn.eval()
                
    def _get_direction_bin(self, r1, c1, r2, c2):
        if r1 == r2 and c1 == c2: return 0
        angle = np.arctan2(r2 - r1, c2 - c1)
        bin_idx = int(np.round(angle / (np.pi/4))) % 8
        return bin_idx + 1
        
    def _get_dist_bin(self, r1, c1, r2, c2):
        dist = abs(r1 - r2) + abs(c1 - c2)
        if dist == 0: return 0
        if dist <= 5: return 1
        if dist <= 15: return 2
        return 3

    def get_action(self, sim):
        # Convert continuous coordinates to roughly discrete grid units for the RL models
        # Grid is 10m / 50 cells = 0.2m per cell
        r = int(self.grid_target_y/0.2) if self.grid_target_y else int(self.y/0.2)
        c = int(self.grid_target_x/0.2) if self.grid_target_x else int(self.x/0.2)
        
        tr = int(self.target_y/0.2)
        tc = int(self.target_x/0.2)
        
        if self.algo == 'q':
            dirt_dir = self._get_direction_bin(r, c, tr, tc)
            dirt_dist = self._get_dist_bin(r, c, tr, tc)
            
            other_robots = [rob for rob in sim.robots if rob.id != self.id]
            closest_rob = min(other_robots, key=lambda rob: abs(self.x-rob.x) + abs(self.y-rob.y))
            cr = int(closest_rob.y/0.2)
            cc = int(closest_rob.x/0.2)
            
            rob_dir = self._get_direction_bin(r, c, cr, cc)
            dist_to_rob = abs(r-cr) + abs(c-cc)
            rob_dist = 1 if dist_to_rob <= 3 else 2 if dist_to_rob <= 10 else 0
            
            state_key = str((dirt_dir, dirt_dist, rob_dir, rob_dist))
            if state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            return 4 # Stay
            
        elif self.algo == 'dqn':
            half_fov = self.fov // 2
            local_grid = np.full((self.fov, self.fov), -1.0) # default wall
            
            # Map sim grid (-1 walls, 0 free)
            for i in range(self.fov):
                for j in range(self.fov):
                    gr = r - half_fov + i
                    gc = c - half_fov + j
                    # Mapping to sim.grid coords (0 to 50)
                    sim_gr = int(gr + 25)
                    sim_gc = int(gc + 25)
                    if 0 <= sim_gr < sim.gs and 0 <= sim_gc < sim.gs:
                        local_grid[i, j] = 0.0 if sim.grid[sim_gr, sim_gc] == 0 else -1.0
            
            # Mark other robots
            for rob in sim.robots:
                if rob.id != self.id:
                    ogr = int(rob.y/0.2)
                    ogc = int(rob.x/0.2)
                    if abs(ogr - r) <= half_fov and abs(ogc - c) <= half_fov:
                        local_grid[ogr - r + half_fov, ogc - c + half_fov] = 2.0
                        
            dy = tr - r
            dx = tc - c
            norm = np.sqrt(dx**2 + dy**2) + 1e-5
            dx /= norm
            dy /= norm
            
            flat_state = np.concatenate([local_grid.flatten(), [dy, dx]])
            s_tensor = torch.FloatTensor(flat_state).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.dqn(s_tensor)
            return torch.argmax(q_vals).item()
            
        return 4

    def update(self, dt, sim):
        if self.is_done: return
        
        dist = math.hypot(self.target_x - self.x, self.target_y - self.y)
        if dist < 0.3:
            self.is_done = True
            return
            
        if self.algo == 'bfs':
            if not self.path:
                self.path = sim.plan_bfs(self.x, self.y, self.target_x, self.target_y)
            if self.path:
                tx, ty = self.path[0]
                if math.hypot(tx - self.x, ty - self.y) < 0.3:
                    self.path.pop(0)
                    if self.path: tx, ty = self.path[0]
                    else: tx, ty = self.target_x, self.target_y
                self.grid_target_x = tx
                self.grid_target_y = ty
        else:
            if self.grid_target_x is None:
                action = self.get_action(sim)
                moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)] # dr, dc -> y, x
                d_ay, d_ax = moves[action]
                self.grid_target_x = self.x + d_ax * 0.4
                self.grid_target_y = self.y + d_ay * 0.4
        
        # Low level controller
        if self.grid_target_x is not None:
            tx = self.grid_target_x
            ty = self.grid_target_y
            dist_to_gt = math.hypot(tx - self.x, ty - self.y)
            if dist_to_gt < 0.1 and self.algo != 'bfs':
                self.grid_target_x = None
            
            dx = tx - self.x
            dy = ty - self.y
            t_yaw = math.atan2(dy, dx)
            
            e_yaw = t_yaw - self.yaw
            while e_yaw > math.pi: e_yaw -= 2*math.pi
            while e_yaw < -math.pi: e_yaw += 2*math.pi
            
            v = 0.0
            w = 0.0
            
            # BFS Fleet Yielding
            if self.algo == 'bfs':
                yield_needed = False
                for rob in sim.robots:
                    if rob.id < self.id and math.hypot(rob.x - self.x, rob.y - self.y) < 1.5 and not rob.is_done:
                        yield_needed = True
                if yield_needed:
                    self.yield_timer += dt
                    if self.yield_timer > 5.0:
                        self.path = [] # force replan
                        self.yield_timer = 0.0
                else:
                    self.yield_timer = 0.0
                    if abs(e_yaw) > 0.2: w = np.sign(e_yaw) * 2.0
                    else: w = 1.0 * e_yaw; v = 0.8
            else:
                if abs(e_yaw) > 0.2: w = np.sign(e_yaw) * 2.0
                else: w = 1.0 * e_yaw; v = 0.8
                
            # Predict next pos
            nx = self.x + v * math.cos(self.yaw) * dt
            ny = self.y + v * math.sin(self.yaw) * dt
            
            # Collision check
            if sim.is_collision(nx, ny, self.radius, self.id):
                self.collisions += 1
                self.collision_points.append((self.x, self.y))
                # Stop if collision
                v = 0.0
            else:
                self.x = nx
                self.y = ny
            self.yaw += w * dt
            self.history.append((self.x, self.y))

class Simulator:
    def __init__(self, algo):
        self.algo = algo
        self.robots = []
        self.walls = [
            (-5.0, -5.0, 3.5, 3.5), # Top Left
            (1.5, -5.0, 3.5, 3.5),  # Top Right
            (-5.0, 1.5, 3.5, 3.5),  # Bot Left
            (1.5, 1.5, 3.5, 3.5)    # Bot Right
        ]
        
        model_dir = '/home/aman/cs671_7/rl_cleaning_project/rl_training/models'
        
        self.robots.append(Robot(0, -4.0, 0.0, 4.0, 0.0, algo, f'{model_dir}/{algo}_R1.{ "json" if algo=="q" else "pth"}'))
        self.robots.append(Robot(1, 0.0, -4.0, 0.0, 4.0, algo, f'{model_dir}/{algo}_R2.{ "json" if algo=="q" else "pth"}'))
        self.robots.append(Robot(2, 4.0, 0.0, -4.0, 0.0, algo, f'{model_dir}/{algo}_R3.{ "json" if algo=="q" else "pth"}'))
        
        self.res = 0.2
        self.gs = int(10.0 / self.res)
        self.grid = np.zeros((self.gs, self.gs))
        for wx, wy, ww, wh in self.walls:
            min_x = int((wx + 5.0)/self.res)
            max_x = int((wx + ww + 5.0)/self.res)
            min_y = int((wy + 5.0)/self.res)
            max_y = int((wy + wh + 5.0)/self.res)
            self.grid[min_x:max_x, min_y:max_y] = 1
            
        self.time = 0.0

    def is_collision(self, x, y, r, r_id):
        for wx, wy, ww, wh in self.walls:
            if wx-r < x < wx+ww+r and wy-r < y < wy+wh+r:
                return True
        for rob in self.robots:
            if rob.id != r_id and not rob.is_done:
                if math.hypot(rob.x - x, rob.y - y) < r * 2:
                    return True
        return False

    def plan_bfs(self, sx, sy, tx, ty):
        stx = max(0, min(self.gs-1, int((sx+5.0)/self.res)))
        sty = max(0, min(self.gs-1, int((sy+5.0)/self.res)))
        gtx = max(0, min(self.gs-1, int((tx+5.0)/self.res)))
        gty = max(0, min(self.gs-1, int((ty+5.0)/self.res)))
        
        q = [(stx, sty)]
        came_from = {(stx, sty): None}
        found = False
        while q:
            cx, cy = q.pop(0)
            if cx == gtx and cy == gty:
                found = True
                break
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nx, ny = cx+dx, cy+dy
                if 0<=nx<self.gs and 0<=ny<self.gs and self.grid[nx,ny]==0:
                    if (nx, ny) not in came_from:
                        q.append((nx, ny))
                        came_from[(nx, ny)] = (cx, cy)
        path = []
        if found:
            curr = (gtx, gty)
            while curr is not None:
                path.append(((curr[0]*self.res)-5.0, (curr[1]*self.res)-5.0))
                curr = came_from[curr]
            path.reverse()
        return path

    def step(self, dt):
        for r in self.robots:
            r.update(dt, self)
        self.time += dt
        return all(r.is_done for r in self.robots) or self.time > 40.0

def run_experiment(algo):
    print(f"Running Experiment: {algo.upper()}...")
    sim = Simulator(algo)
    dt = 0.1
    
    fig, ax = plt.subplots(figsize=(6,6))
    frames = []
    
    while not sim.step(dt):
        if int(sim.time * 10) % 5 == 0:
            ax.clear()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            for wx, wy, ww, wh in sim.walls:
                ax.add_patch(plt.Rectangle((wx, wy), ww, wh, color='black', alpha=0.5))
            colors = ['red', 'green', 'blue']
            for i, r in enumerate(sim.robots):
                circle = plt.Circle((r.x, r.y), r.radius, color=colors[i])
                ax.add_patch(circle)
                ax.scatter(r.target_x, r.target_y, marker='x', color=colors[i], s=100)
            ax.set_title(f"{algo.upper()} - Time: {sim.time:.1f}s")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
    
    plt.close(fig)
    
    os.makedirs('/home/aman/cs671_7/rl_eval_ws/python_eval/results', exist_ok=True)
    gif_path = f'/home/aman/cs671_7/rl_eval_ws/python_eval/results/{algo}_eval.gif'
    
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    if imgs:
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"Saved GIF: {gif_path}")
    
    plt.figure(figsize=(6,6))
    for wx, wy, ww, wh in sim.walls:
        plt.gca().add_patch(plt.Rectangle((wx, wy), ww, wh, color='black', alpha=0.5))
        
    colors = ['r', 'g', 'b']
    all_cx, all_cy = [], []
    for i, r in enumerate(sim.robots):
        hx = [p[0] for p in r.history]
        hy = [p[1] for p in r.history]
        plt.plot(hx, hy, color=colors[i], label=f'Robot {i}')
        if r.collision_points:
            cx = [p[0] for p in r.collision_points]
            cy = [p[1] for p in r.collision_points]
            all_cx.extend(cx)
            all_cy.extend(cy)
            plt.scatter(cx, cy, color='black', s=20, marker='x')
    
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(f"{algo.upper()} Trajectories (Total Collisions: {sum(r.collisions for r in sim.robots)})")
    plt.legend()
    plt.savefig(f'/home/aman/cs671_7/rl_eval_ws/python_eval/results/{algo}_trajectory.png')
    plt.close()
    
    plt.figure(figsize=(6,6))
    for wx, wy, ww, wh in sim.walls:
        plt.gca().add_patch(plt.Rectangle((wx, wy), ww, wh, color='black', alpha=0.5))
    if all_cx:
        plt.hexbin(all_cx, all_cy, gridsize=15, cmap='Reds', extent=[-5, 5, -5, 5], mincnt=1, alpha=0.8)
        plt.colorbar(label='Collisions')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.title(f"{algo.upper()} Collision Heatmap")
    plt.savefig(f'/home/aman/cs671_7/rl_eval_ws/python_eval/results/{algo}_heatmap.png')
    plt.close()

if __name__ == '__main__':
    for a in ['bfs', 'q', 'dqn']:
        run_experiment(a)
