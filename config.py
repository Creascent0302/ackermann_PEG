import numpy as np

SIMULATION_CONFIG = {
    "time_step": 0.1,  # Time step for the simulation in seconds
    "max_steps": 500,  # Maximum number of steps in the simulation
}

# Configuration for the bicycle model
BICYCLE_MODEL_CONFIG = {
    "max_steering_angle": np.radians(75),  # Maximum steering angle in radians
    "max_speed": 0.4 ,                     # Maximum speed in m/s
    "wheelbase": 0.14,                     # Wheelbase of the bicycle in meters
    "min_turning_radius": 0.14 / np.tan(np.radians(60))  # Minimum turning radius in meters
}

ENV_CONFIG = {
    "gridnum_width": 30,  # Width of the environment in grid cells
    "gridnum_height": 15,  # Height of the environment in grid cells
    "obstacle_probability": 0.2,  # Probability of each grid cell being an obstacle
    "cell_size": 0.2,  # Size of each grid cell in pixel
    "agent_collision_radius": 0.07,  # Collision radius for agents
    "agent_capture_distance": 0.3  # Distance within which pursuers can capture the evader
}

# Pygame display configuration
PYGAME_CONFIG = {

    "render_size": 200,
    "fps": 20,        # Frames per second
    "font_size": 24,  # Font size for text
    "colors": {
        "WHITE": (255, 255, 255),
        "BLACK": (0, 0, 0),
        "GRAY": (200, 200, 200),
        "EVADER": (0, 0, 255),      # Blue
        "PURSUER": (255, 0, 0),     # Red
        "OBSTACLE": (128, 128, 128), # Gray
        "BACKGROUND": (255, 255, 255)  # White background
    }

}