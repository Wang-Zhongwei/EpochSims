from .base_config import *
import os

# CHANGE THIS TO NEW EXPERIMENT NAME
# naming convention: YYYY-MM-DD_<dim>_<material>_<wavelength>_<intensity in SI>_<incident_angle>
experiment_name = "2023-08-01_3D_8CB_800nm_1e21_22deg"

# TRY NOT TO CHANGE THESE
epoch_output_folder = os.path.join(EPOCH_OUTPUT_BASE, experiment_name)
raw_data_folder = os.path.join(RAW_DATA_BASE, experiment_name)
processed_data_folder = os.path.join(PROCESSED_DATA_BASE, experiment_name)
media_folder = os.path.join(MEDIA_BASE, experiment_name)

leaving_particles_file_name = "leaving_particles.csv"
