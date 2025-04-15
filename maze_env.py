from main import MazeEnv
import q_learning
# import render_maze


env = MazeEnv()
q_table = q_learning(env)
# render_maze(env, q_table)