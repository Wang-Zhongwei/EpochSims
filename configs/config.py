import os
from .base_config import *

# CHANGE THIS TO NEW EXPERIMENT NAME
experiment_name = "2023-08-31_convergence_test"
deck_name = "input"

# RUNNING TIME
hours = 2
minutes = 0
seconds = 0

# COMPUTE RESOURCES
nodes = 24
ntasks_per_node = 40

# PATHS
epoch_output_dir = os.path.join(EPOCH_OUTPUT_BASE, experiment_name)
experiment_folder = os.path.join(DATA_BASE, experiment_name)

raw_data_folder = os.path.join(experiment_folder, "raw")
processed_data_folder = os.path.join(experiment_folder, "processed")
media_folder = os.path.join(experiment_folder, "media")

# NAMES
leaving_particles_file_name = "leaving_particles.csv"
