from environment import MazeEnv
from q_learning import train_q_learning
from visualization import run_visualization, setup_environment

def main():
    print("Choose maze setup mode:")
    print("1. Randomly generated maze")
    print("2. Manually draw walls")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '2':
        env = MazeEnv(mode='manual')
        print("\nManual maze setup mode activated.")
        print("Left-click to add walls, right-click to remove walls.")
        print("Press ENTER to start training once done.\n")
        setup_environment(env)
    else:
        env = MazeEnv(mode='random')
        print("\nRandom maze generated automatically.")
    
    print("Training Q-learning agent...")
    q_table = train_q_learning(env, episodes=200)
    
    print("Running visualization...")
    run_visualization(env, q_table)

if __name__ == "__main__":
    main()