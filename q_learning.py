import numpy as np
import random
import pygame  # Import pygame for visualization
from environment import MazeEnv
from constants import *

# Initialize Pygame for training visualization
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Maze Training Visualization")
clock = pygame.time.Clock()

def draw_training_state(env):
    """Draw the environment state during training"""
    screen.fill(BLACK)
    
    # Draw cells (walls and paths)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if env.maze[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, rect)
            # Mark visited cells with a border
            if (y, x) in env.visited:
                pygame.draw.rect(screen, RED, rect, 1)
    
    # Draw grid lines
    for y in range(GRID_HEIGHT):
        pygame.draw.line(screen, GRAY, (0, y * CELL_SIZE), (WINDOW_WIDTH, y * CELL_SIZE), 1)
    for x in range(GRID_WIDTH):
        pygame.draw.line(screen, GRAY, (x * CELL_SIZE, 0), (x * CELL_SIZE, WINDOW_HEIGHT), 1)
    
    # Draw goal position
    gy, gx = env.goal
    pygame.draw.rect(screen, BLUE, (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    # Draw agent as a red circle
    ay, ax = env.agent
    center = (ax * CELL_SIZE + CELL_SIZE//2, ay * CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE//3)
    
    pygame.display.flip()

def train_q_learning(env, episodes=1000, alpha=0.1, gamma=0.95,
                    epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.01):
    """
    Train a Q-learning agent with real-time visualization
    """
    q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, env.action_space.n))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Check for quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return q_table
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            best_next = np.max(q_table[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            # Visualization update
            draw_training_state(env)
            clock.tick(Frame_RATE)  # Control visualization speed (30 FPS)
            
            state = next_state
            total_reward += reward

            # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 10 == 0:  # Print progress more frequently
            print(f"Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}")
    
        
    return q_table

def extract_path(q_table, env, max_steps=1000):
    """
    Extract the optimal path through the maze using the trained Q-table.
    
    Args:
        q_table (numpy.ndarray): Trained Q-table
        env (MazeEnv): The maze environment instance
        max_steps (int): Maximum steps to attempt before stopping (default: 1000)
        
    Returns:
        list: Sequence of (y,x) positions representing the optimal path
    """
    
    state = env.reset()  # Start from initial state
    path = [env.agent]  # Initialize path with starting position
    steps = 0
    
    # Follow the policy until reaching goal or max steps
    while steps < max_steps:
        # Select action with highest Q-value
        action = np.argmax(q_table[state])
        
        # Take action in environment
        state, _, done, _ = env.step(action)
        
        # Record new position
        path.append(env.agent)
        
        # Check if goal reached
        if done:
            break
            
        steps += 1
    
    return path