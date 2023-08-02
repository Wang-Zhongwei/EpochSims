from .base_config import *
import os

experiment_name = "2023-08-01_3D_8CB_800nm_1e21_22deg"

epoch_output_path = os.path.join(EPOCH_OUTPUT_BASE, experiment_name)
raw_data_path = os.path.join(RAW_DATA_BASE, experiment_name)
processed_data_path = os.path.join(PROCESSED_DATA_BASE, experiment_name)
media_path = os.path.join(MEDIA_BASE, experiment_name)

leaving_particles_file_name = "leaving_particles.csv"
