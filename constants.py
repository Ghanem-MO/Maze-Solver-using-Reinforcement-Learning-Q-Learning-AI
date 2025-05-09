# Grid and display settings
# GRID_WIDTH, GRID_HEIGHT = 20, 12  
GRID_WIDTH, GRID_HEIGHT = 8, 8  
CELL_SIZE = 80  
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE  
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE  

# Control visualization speed (30 FPS)
Frame_RATE = 90  # FPS

# Color definitions (RGB)
WHITE = (255, 255, 255)  # Empty paths
BLACK = (0, 0, 0)        # Walls
RED = (250, 0, 0)        # Agent
GRAY = (180, 180, 180)   # Grid lines
BLUE = (0, 120, 255)     # Goal
GREEN = (0, 200, 0)      # Path visualization