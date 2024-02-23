import os
from .base_config import *

# CHANGE THIS TO NEW EXPERIMENT NAME
experiment_name = "2D_convergence"
deck_name = "2d_laser_rotation_original"

# PATHS
analysis_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "analysis",
    experiment_name,
    deck_name,
)
raw_data_folder = os.path.join(analysis_path, "raw")
processed_data_folder = os.path.join(analysis_path, "processed")
media_folder = os.path.join(analysis_path, "media")

# NAMES
leaving_particles_file_name = "leaving_particles.csv"
