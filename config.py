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
    "gridnum_width": 40,  # Width of the environment in grid cells
    "gridnum_height": 20,  # Height of the environment in grid cells
    "obstacle_probability": 0.2,  # Probability of each grid cell being an obstacle
    "cell_size": 0.2,  # Size of each grid cell in pixel
    "agent_collision_radius": 0.07,  # Collision radius for agents
    "agent_capture_distance": 0.3,  # Distance within which pursuers can capture the evader
    "node_generate_mode": "sun_ray",  # "sun_ray" or "sampling"
    "radius_boundary_of_repulsion_high": 3, # 引入排斥的半径边界(栅格距离)
    "radius_boundary_of_repulsion_low": 1,
    "angle_of_repulsion_high": np.pi / 12, # 排斥的角度范围，即以已有的需要避让的边界为中心，向两侧各延伸的角度，设置为 15 度
    "angle_of_repulsion_low": np.pi / 3,
    "angle_of_repulsion_default": np.pi / 20, # 默认的排斥角度范围
    "angle_iter_step": np.pi / 60 # 角度迭代步长，默认每 3 度发一条射线
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