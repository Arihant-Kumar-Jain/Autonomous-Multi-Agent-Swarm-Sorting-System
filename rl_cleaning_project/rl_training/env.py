import numpy as np

class CleaningEnv:
    def __init__(self, grid_size=50, num_robots=3, num_dirt=10, max_steps=400):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.num_dirt = num_dirt
        self.max_steps = max_steps
        
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.robot_positions = []
        self.dirt_positions = set()
        self.steps = 0
        
        self._build_roads()

    def _build_roads(self):
        # Create a grid of roads (0=road, -1=wall)
        self.grid.fill(-1)
        # Every 5th row and column is a road (2 cells wide)
        for i in range(self.grid_size):
            if i % 5 == 0 or i % 5 == 1:
                self.grid[i, :] = 0
                self.grid[:, i] = 0

    def reset(self):
        self.steps = 0
        self.robot_positions = []
        self.dirt_positions = set()
        
        # Get road coordinates
        road_coords = np.argwhere(self.grid == 0)
        
        # Spawn robots
        idx = np.random.choice(len(road_coords), self.num_robots, replace=False)
        for i in idx:
            self.robot_positions.append(tuple(road_coords[i]))
            
        # Spawn dirt
        road_coords_list = [tuple(c) for c in road_coords if tuple(c) not in self.robot_positions]
        np.random.shuffle(road_coords_list)
        for i in range(self.num_dirt):
            self.dirt_positions.add(road_coords_list[i])
            
        return self._get_state()

    def _get_state(self):
        return {
            'robots': list(self.robot_positions),
            'dirt': list(self.dirt_positions),
            'grid': self.grid.copy() # read-only copy of the base grid
        }

    def step(self, actions):
        # Actions: list of int (0: up, 1: down, 2: left, 3: right, 4: stay)
        self.steps += 1
        collisions = np.zeros(self.num_robots, dtype=bool)
        cleaned_this_step = np.zeros(self.num_robots, dtype=bool)
        
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        new_positions = []
        
        # Determine new proposed positions
        for i, action in enumerate(actions):
            r, c = self.robot_positions[i]
            dr, dc = moves[action]
            nr, nc = r + dr, c + dc
            
            # Check bounds and walls
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size and self.grid[nr, nc] == 0:
                new_positions.append((nr, nc))
            else:
                new_positions.append((r, c)) # Stay if blocked
                
        # Handle collisions
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                if new_positions[i] == new_positions[j]:
                    collisions[i] = True
                    collisions[j] = True
                    new_positions[i] = self.robot_positions[i]
                    new_positions[j] = self.robot_positions[j]
                # Swap collision
                elif new_positions[i] == self.robot_positions[j] and new_positions[j] == self.robot_positions[i]:
                    collisions[i] = True
                    collisions[j] = True
                    new_positions[i] = self.robot_positions[i]
                    new_positions[j] = self.robot_positions[j]
                    
        self.robot_positions = new_positions
        
        # Handle dirt
        for i, pos in enumerate(self.robot_positions):
            if pos in self.dirt_positions:
                self.dirt_positions.remove(pos)
                cleaned_this_step[i] = True
                
        done = len(self.dirt_positions) == 0 or self.steps >= self.max_steps
        
        info = {
            'collisions': collisions,
            'cleaned': cleaned_this_step,
            'steps': self.steps,
            'all_dirt_cleaned': len(self.dirt_positions) == 0
        }
        return self._get_state(), done, info
