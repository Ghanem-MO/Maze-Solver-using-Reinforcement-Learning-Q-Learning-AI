import numpy as np
import random
from environment import MazeEnv
from constants import *
import pygame
from constants import *

# Initialize Pygame for training visualization
pygame.init()
training_screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Q-Learning Training Visualization")
training_clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

def draw_training_state(env, screen, font, episode, epsilon, total_reward):
    """Draw the current state during training with overlay stats"""
    screen.fill(BLACK)

    # Draw maze cells
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # Draw an rat trap instead of a wall
            if env.maze[y][x] == 1:
                trap_image = pygame.image.load("trap.png")
                trap_image = pygame.transform.scale(trap_image, (CELL_SIZE, CELL_SIZE))
                screen.blit(trap_image, (x * CELL_SIZE, y * CELL_SIZE))
            else:
                # Draw free cell
                pygame.draw.rect(screen, WHITE, rect)           

            if (y, x) in env.visited:
                pygame.draw.rect(screen, RED, rect, 1)

    # Draw grid lines
    for y in range(GRID_HEIGHT):
        pygame.draw.line(screen, GRAY, (0, y * CELL_SIZE), (WINDOW_WIDTH, y * CELL_SIZE), 1)
    for x in range(GRID_WIDTH):
        pygame.draw.line(screen, GRAY, (x * CELL_SIZE, 0), (x * CELL_SIZE, WINDOW_HEIGHT), 1)

    # Draw goal position
    gy, gx = env.goal
    # Draw goal position as a cheese image
    cheese_image = pygame.image.load("cheese.png")
    cheese_image = pygame.transform.scale(cheese_image, (CELL_SIZE, CELL_SIZE))
    screen.blit(cheese_image, (gx * CELL_SIZE, gy * CELL_SIZE))

    # Draw agent
    ay, ax = env.agent
    center = (ax * CELL_SIZE + CELL_SIZE // 2, ay * CELL_SIZE + CELL_SIZE // 2)
    # Draw agent as an image    
    agent_image = pygame.image.load("rat.png")
    agent_image = pygame.transform.scale(agent_image, (CELL_SIZE, CELL_SIZE))
    screen.blit(agent_image, (ax * CELL_SIZE, ay * CELL_SIZE))

    pygame.display.flip()

def train_q_learning(env, episodes=1000, alpha=0.1, gamma=0.95,
                    epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.01):
    q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, env.action_space.n))

    for episode in range(episodes):
        state, _ = env.reset()
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
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            best_next = np.max(q_table[next_state])
            td_target = reward + gamma * best_next
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            # Update state and reward
            state = next_state
            total_reward += reward

            # Draw training state
            draw_training_state(env, training_screen, font, episode, epsilon, total_reward)
            training_clock.tick(Frame_RATE)  # Limit to 30 FPS for visualization

        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        # Print episode stats
        if(episodes % 10 == 0 ):
            print(f"Episode: {episode}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}")
        
        # stop training if the last 10 episodes total reward does not improve by 10%
        if episode > 10 and total_reward < 0.1 * (episodes / 10):
            print(f"Stopping training at episode {episode} due to lack of improvement.")
            break

    return q_table

def extract_path(q_table, env, max_steps=1000):
    state, _ = env.reset()
    path = [env.agent]
    steps = 0
    
    while steps < max_steps:
        action = np.argmax(q_table[state])
        state, _, terminated, truncated, _ = env.step(action)
        path.append(env.agent)
        if terminated or truncated:
            break
        steps += 1
    return path