"""BFS / A* pathfinding for grid world."""

from collections import deque
import heapq
import config as cfg


def bfs(grid, start, goal, blocked_cells=None):
    """BFS shortest path. Returns list of (row, col) from start to goal (inclusive).
    
    Args:
        grid: 2D list of cell types
        start: (row, col)
        goal: (row, col)
        blocked_cells: set of (row, col) to treat as walls (e.g. other robots)
    
    Returns:
        List of (row, col) or empty list if no path.
    """
    if start == goal:
        return [start]

    blocked = blocked_cells or set()
    rows, cols = len(grid), len(grid[0])
    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for action_id, (dr, dc) in cfg.ACTIONS.items():
            if action_id == 4:  # skip WAIT
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                if grid[nr][nc] != cfg.WALL and (nr, nc) not in blocked:
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    if (nr, nc) == goal:
                        # reconstruct path
                        path = []
                        node = goal
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        return path[::-1]
                    queue.append((nr, nc))
    return []  # no path found


def astar(grid, start, goal, blocked_cells=None):
    """A* pathfinding with Manhattan heuristic."""
    if start == goal:
        return [start]

    blocked = blocked_cells or set()
    rows, cols = len(grid), len(grid[0])

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}
    parent = {start: None}

    while open_set:
        f, g, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1]

        for action_id, (dr, dc) in cfg.ACTIONS.items():
            if action_id == 4:
                continue
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != cfg.WALL and (nr, nc) not in blocked:
                    new_g = g + 1
                    if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = new_g
                        f = new_g + heuristic((nr, nc), goal)
                        parent[(nr, nc)] = current
                        heapq.heappush(open_set, (f, new_g, (nr, nc)))
    return []


def manhattan_distance(a, b):
    """Manhattan distance between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
