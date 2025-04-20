import numpy as np
import random
from environment import MazeEnv
from constants import *

def train_q_learning(env, episodes=1000, alpha=0.1, gamma=0.95,
                    epsilon=0.2, epsilon_decay=0.995, min_epsilon=0.01):
    """
    Train a Q-learning agent to solve the maze environment.
    
    Implements the standard Q-learning algorithm with epsilon-greedy exploration.
    
    Args:
        env (MazeEnv): The maze environment instance
        episodes (int): Number of training episodes (default: 1000)
        alpha (float): Learning rate (default: 0.1)
        gamma (float): Discount factor for future rewards (default: 0.95)
        epsilon (float): Initial exploration rate (default: 0.2)
        epsilon_decay (float): Rate at which epsilon decays each episode (default: 0.995)
        min_epsilon (float): Minimum exploration probability (default: 0.01)
        
    Returns:
        numpy.ndarray: The trained Q-table with shape (num_states, num_actions)
    """
    
    # Initialize Q-table with zeros
    # Dimensions: (GRID_HEIGHT * GRID_WIDTH) states × 4 actions
    q_table = np.zeros((GRID_HEIGHT * GRID_WIDTH, env.action_space.n))
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()  # Reset environment for new episode
        done = False
        total_reward = 0
        
        # Episode loop
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Exploration: random action
                action = env.action_space.sample()  
            else:
                # Exploitation: best known action from Q-table
                action = np.argmax(q_table[state])  
            
            # Execute action in environment
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update using Bellman equation:
            # Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
            best_next = np.max(q_table[next_state])  # Max Q-value for next state
            td_target = reward + gamma * best_next  # Target Q-value
            td_error = td_target - q_table[state][action]  # Temporal difference error
            q_table[state][action] += alpha * td_error  # Update Q-value
            
            # Transition to next state
            state = next_state
            total_reward += reward
        
        # Decay exploration rate (epsilon)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Print training progress every 100 episodes
        if episode % 100 == 0:
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