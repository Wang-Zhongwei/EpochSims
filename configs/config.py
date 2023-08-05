import os
from .base_config import *

# CHANGE THIS TO NEW EXPERIMENT NAME
experiment_name = "2023-08-03_3D_8CB_800nm_1e21_22deg"
deck_name = 'input'

# RUNNING TIME
hours = 18
minutes = 0
seconds = 0

# COMPUTE RESOURCES
nodes = 24
ntasks_per_node = 40

# PATHS
epoch_output_folder = os.path.join(EPOCH_OUTPUT_BASE, experiment_name)
raw_data_folder = os.path.join(RAW_DATA_BASE, experiment_name)
processed_data_folder = os.path.join(PROCESSED_DATA_BASE, experiment_name)
media_folder = os.path.join(MEDIA_BASE, experiment_name)

# NAMES
leaving_particles_file_name = "leaving_particles.csv"
