import pygame
from constants import *
from q_learning import extract_path
#from q_learning import extract_path
# Agent_Start=False;

def draw_maze(screen, env, path=None, possible_paths=None,current_frame=0,Agent_Start=False):
    """Draw the maze, agent, goal, and path on the pygame screen"""
    # Draw cells (walls and paths)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if env.maze[y][x] == 0 else BLACK
            pygame.draw.rect(screen, color, rect)

    
    # Draw grid lines
    for y in range(GRID_HEIGHT):
        pygame.draw.line(screen, GRAY, (0, y * CELL_SIZE), (WINDOW_WIDTH, y * CELL_SIZE), 1)
    for x in range(GRID_WIDTH):
        pygame.draw.line(screen, GRAY, (x * CELL_SIZE, 0), (x * CELL_SIZE, WINDOW_HEIGHT), 1)
    
    # Draw goal position
    gy, gx = env.goal
    pygame.draw.rect(screen, BLUE, (gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Draw Start position
    sy, sx = env.start
    pygame.draw.rect(screen, RED, (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE),1)

    if possible_paths:
        for (py, px), step in possible_paths.items():
            if (py, px) == env.goal:
                continue
            if (py, px) == env.start:
                continue
            if step <= current_frame:
                rect2 = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # Gradient animation color based on step
                intensity = min(255, 30 + step * 5)
                color = (200 - intensity // 2, 255 - intensity // 3, 100 + intensity // 2)
                pygame.draw.rect(screen, color, rect2)
                pygame.draw.rect(screen, GRAY, rect2, 1)

                # pygame.draw.line(screen, LIGHT_GREEN,
                #                  (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2),
                #                  (nx * CELL_SIZE + CELL_SIZE // 2, ny * CELL_SIZE + CELL_SIZE // 2), 2)

 # Draw the path from start to goal
    if path:
        for i in range(1, len(path)):
             # Mark goal cell with a border
            pygame.draw.rect(screen, RED,(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE),1)
             # Color start cell with RED
            pygame.draw.rect(screen, RED, (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            py, px = path[i-1]
            rect3 = pygame.Rect(px * CELL_SIZE, py * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen,GREEN , rect3)
            pygame.draw.rect(screen,GRAY, rect3, 1)
             # ny, nx = path[i]
            # pygame.draw.line(screen, GREEN,
            #                 (px * CELL_SIZE + CELL_SIZE//2, py * CELL_SIZE + CELL_SIZE//2),
            #                 (nx * CELL_SIZE + CELL_SIZE//2, ny * CELL_SIZE + CELL_SIZE//2), 3)
            #

    # Draw agent as a red circle
    ay, ax = env.agent
    center = (ax * CELL_SIZE + CELL_SIZE//2, ay * CELL_SIZE + CELL_SIZE//2)
    pygame.draw.circle(screen, RED, center, CELL_SIZE//3)

def generate_possible_paths(env, num_paths=1, max_steps=100):
    """Flood-fill with step tracking for animation."""
    from collections import deque
    visited = {}
    queue = deque([(env.start, 0)])  # (cell, step_index)
    visited[env.start] = 0 # start with agent starting

    while queue:
        (y, x), step = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                if env.maze[ny][nx] == 0 and (ny, nx) not in visited:
                    visited[(ny, nx)] = step + 1
                    queue.append(((ny, nx), step + 1))


    return visited  # Now returns dict of (y,x) -> step_index



def run_visualization(env, q_table):
    """Run pygame visualization of the maze and solution path."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Maze Solver using Q-learning")
    clock = pygame.time.Clock()

    # Extract the optimal path from Q-table
    path = extract_path(env,q_table)
    possible_paths = generate_possible_paths(env)
    frame_counter = 0 # for animation the pathes
    animation_duration_frames = 34 # Number of frames for animation

    running = True
    path_index = 0
    env.reset()

    # Main visualization loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Phase 1: animate possible paths only
        if frame_counter < animation_duration_frames:
            env.agent = env.start  # keep agent at start during animation
        # Phase 2: start moving the agent
        elif path_index < len(path):
            env.agent = path[path_index]
            path_index += 1
        # Draw everything
        screen.fill(BLACK)
        draw_maze(screen, env, path[:path_index], possible_paths=possible_paths, current_frame=frame_counter)
        pygame.display.flip()
        clock.tick(10)  # FPS
        frame_counter += 1
    pygame.quit()