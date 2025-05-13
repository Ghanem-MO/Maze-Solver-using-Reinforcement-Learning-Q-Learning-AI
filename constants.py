# ===============================
# Grid Configuration and Display Settings
# ===============================

# Grid dimensions (number of cells)
GRID_WIDTH, GRID_HEIGHT = 10, 10  

# Size of each cell in pixels
CELL_SIZE = 70  

# Window dimensions based on grid and cell size
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE  
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE  

# Frame rate to control the speed of visualization updates (in frames per second)
Frame_RATE = 140  


# ===============================
# RGB Color Definitions for Visualization
# ===============================

WHITE = (255, 255, 255)    # Represents empty/path cells in the maze
BLACK = (0, 0, 0)          # Represents walls or obstacles
RED = (250, 0, 0)          # Represents the agent's current position
YELLOW = (255, 255, 200)   # Represents visited cells during exploration (soft yellow for clarity)
GRAY = (180, 180, 180)     # Represents the grid lines (visual boundaries between cells)
BLUE = (0, 120, 255)       # Represents the goal cell (target destination)
GREEN = (0, 200, 0)        # Represents the solution path or optimal route
