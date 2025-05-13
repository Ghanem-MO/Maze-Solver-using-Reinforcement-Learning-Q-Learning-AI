import pygame
from constants import *
from environment import MazeEnv
from q_learning import train_q_learning
from visualization import run_visualization, setup_environment

# Global flag to toggle training visualization
show_training_visualization = True

def draw_button(screen, text, rect, color_normal, color_hover, mouse_pos):
    """
    Draws a clickable button with hover effect.

    Parameters:
        screen (pygame.Surface): The surface to draw the button on.
        text (str): Text label for the button.
        rect (pygame.Rect): Button position and size.
        color_normal (tuple): Button color when not hovered.
        color_hover (tuple): Button color when hovered.
        mouse_pos (tuple): Current mouse position to check for hover.
    """
    # Change button color on hover
    if rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, color_hover, rect)
    else:
        pygame.draw.rect(screen, color_normal, rect)
    
    # Render button text
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

def main_menu():
    """
    Displays the main menu GUI using Pygame and handles user interactions.

    Returns:
        str or None: Returns 'random', 'manual', or None based on user selection.
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Maze Q-Learning")

    global show_training_visualization
    running = True

    while running:
        screen.fill(YELLOW)  # Fill the screen background
        mouse_pos = pygame.mouse.get_pos()  # Track mouse position

        # Display current maze size info
        maze_size = f"Current Maze Size: {GRID_WIDTH} x {GRID_HEIGHT}" 
        font = pygame.font.Font(None, 36)
        text_surface = font.render(maze_size, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(400, 100))
        screen.blit(text_surface, text_rect)

        # Draw menu buttons
        draw_button(screen, "Random Maze", pygame.Rect(300, 200, 200, 50), (0, 128, 0), (0, 255, 0), mouse_pos)
        draw_button(screen, "Manual Maze", pygame.Rect(300, 300, 200, 50), (0, 128, 0), (0, 255, 0), mouse_pos)
        draw_button(screen, "Exit", pygame.Rect(300, 400, 200, 50), (128, 0, 0), (255, 0, 0), mouse_pos)

        # --- Checkbox for training visualization toggle ---
        checkbox_text = "Show Training" if show_training_visualization else "Hide Training"
        checkbox_font = pygame.font.Font(None, 24)
        checkbox_text_surface = checkbox_font.render(checkbox_text, True, (0, 0, 0))
        checkbox_text_rect = checkbox_text_surface.get_rect(center=(400, 500))
        screen.blit(checkbox_text_surface, checkbox_text_rect)

        # Draw checkbox box
        checkbox_rect = pygame.Rect(300, 490, 20, 20)
        pygame.draw.rect(screen, (255, 255, 255), checkbox_rect, 2)

        # Toggle checkbox state on click
        if pygame.mouse.get_pressed()[0]:  # Left mouse button is pressed
            if checkbox_text_rect.collidepoint(mouse_pos) or checkbox_rect.collidepoint(mouse_pos):
                show_training_visualization = not show_training_visualization
                # Wait until mouse is released to avoid rapid toggling
                while pygame.mouse.get_pressed()[0]:
                    for event in pygame.event.get():
                        if event.type == pygame.MOUSEBUTTONUP:
                            break

        # Fill checkbox with color based on state
        if show_training_visualization:
            pygame.draw.rect(screen, (0, 255, 0), checkbox_rect)  # Green = ON
        else:
            pygame.draw.rect(screen, (255, 0, 0), checkbox_rect)  # Red = OFF

        # --- Handle button click events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False  # Close the window
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if pygame.Rect(300, 200, 200, 50).collidepoint(mouse_pos):
                    return 'random'
                elif pygame.Rect(300, 300, 200, 50).collidepoint(mouse_pos):
                    return 'manual'
                elif pygame.Rect(300, 400, 200, 50).collidepoint(mouse_pos):
                    return None

        pygame.display.flip()  # Update the display

    pygame.quit()
    return None

def main():
    """
    Main application flow:
        - Displays the menu
        - Initializes the maze environment
        - Trains the agent using Q-learning
        - Displays the learned policy
    """
    choice = main_menu()
    print(f"User choice: {choice}")

    # Create maze environment based on user's menu choice
    if choice == 'random':
        env = MazeEnv(mode='random')
        print("\nRandom maze generated automatically.")
        setup_environment(env)
    elif choice == 'manual':
        env = MazeEnv(mode='manual')
        print("\nManual maze setup mode activated.")
        print("Left-click to add walls, right-click to remove walls.")
        print("Press ENTER to start training once done.\n")
        setup_environment(env)
    elif choice is None:
        print("Exiting the program.")
        return

    # Train agent using Q-learning
    print("Training Q-learning agent...")
    if show_training_visualization:
        print("Visualization enabled.")
        q_table = train_q_learning(env, episodes=200, epsilon_decay=0.99, min_epsilon=0.01, with_visualization=True)
    else:
        print("Visualization disabled.")
        q_table = train_q_learning(env, episodes=200, epsilon_decay=0.99, min_epsilon=0.01, with_visualization=False)
    print("Training completed.")

    # Display the agent running in the maze using the learned Q-table
    print("Running visualization...")
    run_visualization(env, q_table)

# Entry point of the script
if __name__ == "__main__":
    main()
