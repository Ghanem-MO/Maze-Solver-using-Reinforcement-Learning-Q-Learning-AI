import numpy as np
import random
from collections import deque
import gym
from gym import spaces
from constants import *

class MazeEnv(gym.Env):
    """Custom Gym environment for maze navigation using reinforcement learning.

     Attributes:
         observation_space: Discrete space representing all possible cell positions
         action_space: Discrete space with 4 possible actions (up, down, left, right)
         start: Tuple (y,x) representing starting position
         goal: Tuple (y,x) representing target position
         agent: Current position of the agent in the maze
         visited: Set of visited cells for visualization
         maze: 2D numpy array representing the maze (1=wall, 0=path)
     """

    def __init__(self):
        """Initialize the maze environment."""
        super(MazeEnv, self).__init__()

        # Define observation space and action space
        self.observation_space = spaces.Discrete(GRID_HEIGHT * GRID_WIDTH)
        self.action_space = spaces.Discrete(4)  # 0=up, 1=down, 2=left, 3=right

        # Initialize start and goal positions
        self.start = (0, 0)
        self.goal = (GRID_HEIGHT - 1, GRID_WIDTH - 1)

        # Agent starts at the starting position
        self.agent = self.start
        self.visited = set()

        # Generate the maze
        self.maze = self._generate_maze()

    def _generate_maze(self):
        """Generate maze using randomized Prim's algorithm.

        Returns:
            numpy.ndarray: 2D array representing the maze (1=wall, 0=path)

        Algorithm Steps:
        1. Start with grid full of walls (1s)
        2. Pick start cell, mark as passage (0) and add its walls to list
        3. While there are walls in the list:
           a. Pick random wall from list
           b. If only one adjacent cell is passage, make wall a passage
           c. Add new walls to the list
        4. Ensure maze is solvable (start connected to goal)
        """
        # Initialize maze with all walls (1=wall, 0=path)
        maze = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)
        walls = []# List to track walls to process
        # Start from initial cell (convert to path)
        x, y = self.start
        maze[y][x] = 0

        # Add neighboring walls of starting cell
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]: # Right, Down, Left, Up
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                # Store wall position and adjacent cell
                walls.append((nx, ny, x, y))
        # Process walls until
        while walls:
            random.shuffle(walls) # Randomize wall selection
            wx, wy, cx, cy = walls.pop() # Current wall and adjacent cell

            if maze[wy][wx] == 1:
                # Count how many adjacent passages (0s)
                passages = sum(
                    1 for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    if (0 <= wx + dx < GRID_WIDTH and
                        0 <= wy + dy < GRID_HEIGHT and
                        maze[wy + dy][wx + dx] == 0)
                )
                # If only one adjacent passage, convert wall to path
                if passages == 1:
                    maze[wy][wx] = 0
                    # Add new neighboring walls
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = wx + dx, wy + dy
                        if (0 <= nx < GRID_WIDTH and
                                0 <= ny < GRID_HEIGHT and
                                maze[ny][nx] == 1):
                            walls.append((nx, ny, wx, wy))
        # Ensure maze is solvable (start connected to goal)
        if not self._is_connected(maze):
            return self._generate_maze()
        return maze

    def _is_connected(self, maze):
        """Check if start and goal are connected using BFS (Breadth-First Search).

        Args:
            maze: 2D numpy array representing the maze

        Returns:
            bool: True if path exists from start to goal, False otherwise
        """
        visited = set()
        queue = deque([self.start])

        while queue:
            x, y = queue.popleft()
            if (y, x) == self.goal:
                return True
            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GRID_WIDTH and
                        0 <= ny < GRID_HEIGHT and
                        maze[ny][nx] == 0 and
                        (nx, ny) not in visited):
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return False

    def step(self, action):
        """Execute one action in the environment.

         Args:
             action: Integer representing movement direction (0-3)

         Returns:
             tuple: (next_state, reward, done, info)
             - next_state: New state after action
             - reward: Reward for taking this action
             - done: Whether episode is complete
             - info: Additional info (empty dict)
         """
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = directions[action]

        new_x = self.agent[0] + dx
        new_y = self.agent[1] + dy

        if 0 <= new_x < len(self.maze) and 0 <= new_y < len(self.maze[0]) and self.maze[new_x][new_y] == 0:
            self.agent = (new_x, new_y)

        done = self.agent == self.goal
        reward = 1 if done else -0.1

        state = self._get_state()
        # Ensure the state is within the valid range
        assert 0 <= state < GRID_HEIGHT * GRID_WIDTH, f"Invalid state: {state}"
        return state, reward, done, {}
    def reset(self):
        """Reset environment to initial state.

       Returns:
           int: Initial state index
       """
        self.agent = self.start
        self.visited = {self.start}
        state = self._get_state()
        # Ensure the state is within the valid range
        assert 0 <= state < GRID_HEIGHT * GRID_WIDTH, f"Invalid state: {state}"
        return state
        # return self._get_state()

    def _get_state(self):
        """Convert the agent's position to a single state index."""
        return self.agent[0] * GRID_WIDTH + self.agent[1]

