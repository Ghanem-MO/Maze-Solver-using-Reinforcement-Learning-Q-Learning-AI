import pygame
from constants import *
from q_learning import extract_path

def draw_maze(screen, env, path=None):
    """Draw the maze, agent, goal, and path on the pygame screen"""
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
    
    # Draw path if provided
    if path:
        for i in range(1, len(path)):
             # Mark goal cell with a border
            pygame.draw.rect(screen, RED, rect, 1)
    
            py, px = path[i-1]
            ny, nx = path[i]
            pygame.draw.line(screen, GREEN, 
                            (px * CELL_SIZE + CELL_SIZE//2, py * CELL_SIZE + CELL_SIZE//2),
                            (nx * CELL_SIZE + CELL_SIZE//2, ny * CELL_SIZE + CELL_SIZE//2), 3)
            
    # Draw agent as a red circle
    ay, ax = env.agent
    center = (ax * CELL_SIZE + CELL_SIZE//2, ay * CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE//3)

def run_visualization(env, q_table):
    """Run pygame visualization of the maze and solution path"""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Maze Solver using Q-learning")
    clock = pygame.time.Clock()
    
    # Extract the optimal path from Q-table
    path = extract_path(q_table, env)
    running = True
    path_index = 0
    env.reset()
    
    # Main visualization loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Animate path step by step
        if path_index < len(path):
            env.agent = path[path_index]
            path_index += 1
        
        # Draw everything
        screen.fill(BLACK)
        draw_maze(screen, env, path[:path_index])
        pygame.display.flip()
        clock.tick(10)  # 10 FPS animation speed
    
    pygame.quit()