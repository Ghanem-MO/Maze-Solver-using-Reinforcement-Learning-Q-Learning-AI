import pygame
from constants import *
from q_learning import extract_path

def draw_maze(screen, env, path=None):
    screen.fill(BLACK)
    
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if env.maze[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, rect)
            if (y, x) in env.visited:
                pygame.draw.rect(screen, RED, rect, 1)
    
    for y in range(GRID_HEIGHT):
        pygame.draw.line(screen, GRAY, (0, y * CELL_SIZE), (WINDOW_WIDTH, y * CELL_SIZE), 1)
    for x in range(GRID_WIDTH):
        pygame.draw.line(screen, GRAY, (x * CELL_SIZE, 0), (x * CELL_SIZE, WINDOW_HEIGHT), 1)
    
    gy, gx = env.goal
    pygame.draw.rect(screen, BLUE, (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    if path:
        for i in range(1, len(path)):
            py, px = path[i-1]
            ny, nx = path[i]
            pygame.draw.line(screen, GREEN, 
                            (px * CELL_SIZE + CELL_SIZE//2, py * CELL_SIZE + CELL_SIZE//2),
                            (nx * CELL_SIZE + CELL_SIZE//2, ny * CELL_SIZE + CELL_SIZE//2), 3)
    
    ay, ax = env.agent
    center = (ax * CELL_SIZE + CELL_SIZE//2, ay * CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE//3)

def setup_environment(env):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Maze Setup - Click to Draw Walls")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                x = mx // CELL_SIZE
                y = my // CELL_SIZE
                if event.button == 1:  # Left-click
                    env.maze[y][x] = 1
                elif event.button == 3:  # Right-click
                    env.maze[y][x] = 0
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
        
        screen.fill(BLACK)
        draw_maze(screen, env)
        
        font = pygame.font.SysFont(None, 24)
        
        pygame.display.flip()
        clock.tick(60)

def run_visualization(env, q_table):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Maze Solver using Q-learning")
    clock = pygame.time.Clock()
    
    path = extract_path(q_table, env)
    running = True
    path_index = 0
    env.reset()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if path_index < len(path):
            env.agent = path[path_index]
            path_index += 1
        
        screen.fill(BLACK)
        draw_maze(screen, env, path[:path_index])
        pygame.display.flip()
        clock.tick(10)
    
    pygame.quit()