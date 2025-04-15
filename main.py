import pygame
import numpy as np
import random

# Maze settings
GRID_WIDTH, GRID_HEIGHT = 20, 12  # Width and height of the maze in grid units
CELL_SIZE = 40  # Size of each cell (square) in pixels
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE  # Window width based on grid size
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE  # Window height based on grid size

# Color definitions for drawing
WHITE = (255, 255, 255)  # Color for empty spaces
BLACK = (0, 0, 0)  # Color for walls
RED = (200, 0, 0)  # Color for the agent (player)
GRAY = (180, 180, 180)  # Color for grid lines (light gray)
BLUE = (0, 120, 255)  # Color for the goal

# Maze Environment Class: Handles the maze generation and agent movement
class MazeEnv:
    def __init__(self):
        self.maze = self.generate_maze()  # Generate random maze
        self.start = (0, 0)  # Starting position of the agent (top-left corner)
        self.goal = (GRID_HEIGHT - 1, GRID_WIDTH - 1)  # Goal position (bottom-right corner)
        self.agent = self.start  # Initialize agent at the start position

    def generate_maze(self):
        # Generate a random maze with narrow paths
        maze = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=int)  # Create a grid filled with walls (1)

        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if random.random() < 0.75:
                    maze[y][x] = 0  # Create paths (0) with a 75% chance

        # Ensure start and goal positions are paths (clear)
        maze[0][0] = 0
        maze[GRID_HEIGHT - 1][GRID_WIDTH - 1] = 0
        return maze

    def reset(self):
        # Reset the agent to the start position
        self.agent = self.start
        return self._state()

    def _state(self):
        # Convert the (y, x) position to a single index for the Q-table
        return self.agent[0] * GRID_WIDTH + self.agent[1]

    def step(self, action):
        # Perform an action and move the agent (0=up, 1=down, 2=left, 3=right)
        y, x = self.agent
        if action == 0 and y > 0: y -= 1  # Move up
        elif action == 1 and y < GRID_HEIGHT - 1: y += 1  # Move down
        elif action == 2 and x > 0: x -= 1  # Move left
        elif action == 3 and x < GRID_WIDTH - 1: x += 1  # Move right

        # Only move if the next position is a path (0)
        if self.maze[y][x] == 0:
            self.agent = (y, x)

        # Reward: +1 for reaching the goal, -0.1 for each step
        reward = 1 if self.agent == self.goal else -0.1
        done = self.agent == self.goal  # Check if the agent has reached the goal
        return self._state(), reward, done  # Return next state, reward, and whether the goal is reached

# Q-learning training function: Trains the agent using Q-learning algorithm
def train_q_learning(env):
    q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, 4))  # Initialize Q-table with zeros (states × actions)
    alpha = 0.1  # Learning rate (how fast the agent learns)
    gamma = 0.9  # Discount factor (how much future rewards are valued)
    epsilon = 0.2  # Exploration rate (probability of choosing a random action instead of the best one)
    episodes = 1000  # Number of episodes to train the agent

    for _ in range(episodes):
        state = env.reset()  # Reset the environment and get the initial state
        done = False

        while not done:
            # Choose action: either explore randomly or exploit the best known action
            if random.random() < epsilon:
                action = random.randint(0, 3)  # Random action (exploration)
            else:
                action = np.argmax(q_table[state])  # Best known action (exploitation)

            # Take action and observe the next state, reward, and whether the goal is reached
            next_state, reward, done = env.step(action)

            # Update the Q-value for the current state and action using the Bellman equation
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state  # Move to the next state

    return q_table  # Return the trained Q-table

# Function to draw the maze grid and the agent's path
def draw_maze(screen, env, path=None):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)  # Define the rectangle for each cell

            if env.maze[y][x] == 1:  # If the cell is a wall (1)
                # Draw obstacles as black lines (stylized)
                if y > 0 and env.maze[y - 1][x] == 1:  # Draw vertical lines between adjacent walls
                    pygame.draw.line(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE), (x * CELL_SIZE, (y + 1) * CELL_SIZE), 1)
                if x > 0 and env.maze[y][x - 1] == 1:  # Draw horizontal lines between adjacent walls
                    pygame.draw.line(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, y * CELL_SIZE), 1)
            else:
                pygame.draw.rect(screen, WHITE, rect)  # Draw a white path for empty spaces

            # Draw grid lines (light gray for background grid layout)
            pygame.draw.line(screen, GRAY, (x * CELL_SIZE, 0), (x * CELL_SIZE, WINDOW_HEIGHT), 1)  # Vertical grid line
            pygame.draw.line(screen, GRAY, (0, y * CELL_SIZE), (WINDOW_WIDTH, y * CELL_SIZE), 1)  # Horizontal grid line

    # Draw goal (blue square) at the goal position
    gy, gx = env.goal
    pygame.draw.rect(screen, BLUE, (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw agent (red circle) at the agent's current position
    ay, ax = env.agent
    center = (ax * CELL_SIZE + CELL_SIZE // 2, ay * CELL_SIZE + CELL_SIZE // 2)  # Center of the agent's circle
    pygame.draw.circle(screen, RED, center, CELL_SIZE // 3)  # Draw the agent as a red circle

# Extract the best path from the Q-table
def extract_path(q_table, env):
    state = env.reset()  # Reset to the starting position
    done = False
    path = [env.agent]  # List to store the path taken by the agent
    visited = set()     # Set to store visited states to avoid loops

    while not done:
        visited.add(state)
        action = np.argmax(q_table[state])  # Pick the best action according to Q-table
        next_state, _, done = env.step(action)

        if next_state in visited:  # Avoid infinite loops if the agent revisits a state
            break
        path.append(env.agent)  # Add the agent's current position to the path
        state = next_state

    return path  # Return the path taken by the agent

# Run the Pygame visualization
def run_visualization(q_table, env):
    pygame.init()  # Initialize Pygame
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  # Create the window
    pygame.display.set_caption("Maze Solver using Q-learning")  # Set window title
    clock = pygame.time.Clock()  # Initialize the clock to control the frame rate

    path = extract_path(q_table, env)  # Extract the path from the trained Q-table
    running = True
    state = env.reset()  # Reset environment to the starting position
    step_idx = 0  # Index to track the current step in the path

    while running:
        clock.tick(5)  # Control the speed of the animation (5 frames per second)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Exit the loop if the window is closed

        if step_idx < len(path):
            env.agent = path[step_idx]  # Move the agent to the next position in the path
            step_idx += 1

        draw_maze(screen, env, path)  # Draw the maze with the current agent position
        pygame.display.flip()  # Update the screen to display the new frame

    pygame.quit()  # Quit Pygame when the loop ends

# Run everything
env = MazeEnv()  # Create maze environment
q_table = train_q_learning(env)  # Train the agent using Q-learning
run_visualization(q_table, env)  # Visualize the trained agent's path
