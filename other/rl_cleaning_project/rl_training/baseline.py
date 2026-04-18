from collections import deque

class GreedyBFSBaseline:
    def __init__(self, env):
        self.env = env
        
    def _bfs_path(self, start, target_positions):
        if start in target_positions:
            return []
            
        queue = deque([(start, [])])
        visited = {start}
        
        # Directions: 0=Up, 1=Down, 2=Left, 3=Right
        moves = [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]
        
        while queue:
            (r, c), path = queue.popleft()
            
            for dr, dc, action in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.env.grid_size and 0 <= nc < self.env.grid_size:
                    if self.env.grid[nr, nc] == 0 and (nr, nc) not in visited:
                        if (nr, nc) in target_positions:
                            return path + [action]
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [action]))
        return []

    def act(self, state):
        actions = []
        dirt = set(state['dirt'])
        
        for i, pos in enumerate(state['robots']):
            if not dirt:
                actions.append(4) # stay
                continue
                
            path = self._bfs_path(pos, dirt)
            if path:
                actions.append(path[0])
            else:
                actions.append(4) # stay
                
        return actions
