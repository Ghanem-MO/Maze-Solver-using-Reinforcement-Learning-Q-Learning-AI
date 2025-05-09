import numpy as np
import random
from collections import deque
import gym
from gym import spaces
from constants import *

class MazeEnv(gym.Env):
    """Custom Gym environment for maze navigation using reinforcement learning."""
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.observation_space = spaces.Discrete(GRID_HEIGHT * GRID_WIDTH)
        self.action_space = spaces.Discrete(4)  
        self.start = (0, 0)
        self.goal = (GRID_HEIGHT-1, GRID_WIDTH-1)
        self.agent = self.start
        self.visited = set()
        self.maze = self._initialize_empty_maze()

    def _initialize_empty_maze(self):
        maze = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        sy, sx = self.start
        gy, gx = self.goal
        maze[sy][sx] = 0
        maze[gy][gx] = 0
        return maze

    def toggle_wall(self, x, y):
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