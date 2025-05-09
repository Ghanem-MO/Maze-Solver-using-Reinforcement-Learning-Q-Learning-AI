from environment import MazeEnv
from q_learning import train_q_learning
from visualization import run_visualization, setup_environment

def main():
    env = MazeEnv()
    print("Enter maze setup mode... Left-click to add/remove walls. Press ENTER to start training.")
    setup_environment(env)
    
    print("Training Q-learning agent...")
    q_table = train_q_learning(env, episodes=500)
    
    print("Running visualization...")
    run_visualization(env, q_table)

if __name__ == "__main__":
    main()