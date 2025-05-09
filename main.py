from environment import MazeEnv  
from q_learning import train_q_learning  
from visualization import run_visualization  

def main():
  
    # Create an instance of our custom maze environment
    env = MazeEnv()
    print("Training Q-learning agent...")
    q_table = train_q_learning(env, episodes=1000,
 min_epsilon=0.09)
    print("Training complete. Running visualization...")
    
    run_visualization(env, q_table)
    
if __name__ == "__main__":
    main()