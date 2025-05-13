import numpy as np
from collections import deque
import gym
from gym import spaces
from constants import *
import random

class MazeEnv(gym.Env):
    def __init__(self, mode='random'):
        """
        Args:
            mode (str): 'random' for auto-generated maze, 'manual' for user-drawn walls
        """
        super(MazeEnv, self).__init__()
        self.observation_space = spaces.Discrete(GRID_HEIGHT * GRID_WIDTH)
        self.action_space = spaces.Discrete(4)  
        self.start = (0, 0)
        self.goal = (GRID_HEIGHT - 1, GRID_WIDTH - 1)
        self.agent = self.start
        self.visited = set()
        
        # Initialize maze based on mode
        if mode == 'manual':
            self.maze = self._initialize_empty_maze()
        else:
            self.maze = self._generate_maze()

    def _initialize_empty_maze(self):
        """Create an empty maze with all paths (0s)"""
        return np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

    def _generate_maze(self):
        """Generate maze using randomized Prim's algorithm"""
        maze = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        walls = []
        x, y = self.start
        maze[y][x] = 0
        
        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                walls.append((nx, ny, x, y))
                
        while walls:
            random.shuffle(walls)
            wx, wy, cx, cy = walls.pop()
            if maze[wy][wx] == 1:
                passages = sum(
                    1 for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]
                    if 0 <= wx+dx < GRID_WIDTH and 0 <= wy+dy < GRID_HEIGHT and maze[wy+dy][wx+dx] == 0
                )
                if passages == 1:  
                    maze[wy][wx] = 0
                    for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                        nx, ny = wx+dx, wy+dy
                        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and maze[ny][nx] == 1:
                            walls.append((nx, ny, wx, wy))
        if not self._is_connected(maze):
            return self._generate_maze()
        return maze

    def _is_connected(self, maze):
        """Check if start and goal are connected using BFS"""
        visited = set()
        queue = deque([self.start])
        while queue:
            x, y = queue.popleft()
            if (y, x) == self.goal:
                return True
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and maze[ny][nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def toggle_wall(self, x, y):
        """Toggle wall/path at specified position, protecting start/goal"""
        if (y, x) == self.start or (y, x) == self.goal:
            return
        self.maze[y][x] = 1 if self.maze[y][x] == 0 else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.start
        self.visited = {self.start}
        return self._get_state(), {}

    def _get_state(self):
        return self.agent[0] * GRID_WIDTH + self.agent[1]

    def step(self, action):
        y, x = self.agent
        new_y, new_x = y, x
        
        if action == 0: new_y = max(0, y-1)
        elif action == 1: new_y = min(GRID_HEIGHT-1, y+1)
        elif action == 2: new_x = max(0, x-1)
        elif action == 3: new_x = min(GRID_WIDTH-1, x+1)

        if self.maze[new_y][new_x] == 0:
            self.agent = (new_y, new_x)
            self.visited.add(self.agent)

        if self.agent == self.goal:
            reward, terminated, truncated = 10, True, False
        elif (new_y, new_x) != (y, x):
            reward, terminated, truncated = -0.1, False, False
        else:
            reward, terminated, truncated = -0.5, False, False

        return self._get_state(), reward, terminated, truncated, {}