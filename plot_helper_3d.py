import logging
import os
import os

import numpy as np
import sdf_helper as sh

from utils import *

# Set up logging
logging.basicConfig(level=logging.INFO)

@timer
def save_frames(
    input_dir: str,
    file_prefix: str,
    var_name: str,
    subset: Plane,
    output_dir: str = None,
):

    logging.info(f'Starting to save frames for variable {var_name}')

    # Check if grid.npy exists in the input_dir
    grid_file_path = os.path.join(input_dir, "grid.npy")
    if os.path.isfile(grid_file_path):
        # Load the data from grid.npy into grid
        grid = np.load(grid_file_path, allow_pickle=True)
    else:
        grid = sh.getdata(
            os.path.join(input_dir, f"{file_prefix}_0000.sdf"),verbose=False
        ).Grid_Grid.data

        # Save the data to grid.npy
        np.save(grid_file_path, grid)

    x_array, y_array, z_array = grid
    slices = [slice(None), slice(None), slice(None)]
    if subset == Plane.XY:
        slices[2] = np.argmin(np.abs(z_array))
    elif subset == Plane.XZ:
        slices[1] = np.argmin(np.abs(y_array))
    elif subset == Plane.YZ:
        slices[0] = np.argmin(np.abs(x_array))
    slices = tuple(slices)

    num_frames = len([f for f in os.listdir(input_dir) if f.startswith(file_prefix)])
    
    out_file_name = f"{var_name}_{subset.value}.npy"

    # Load data
    var_data_list = []
    for frame in range(num_frames):
        logging.info(f'Loading frame {frame}')
        data = sh.getdata(os.path.join(input_dir, f"{file_prefix}_{frame:04d}.sdf"), verbose=False)
        var = data.__getattribute__(var_name).data
        del data
        if subset:
            var = var[slices]
        var_data_list.append(np.expand_dims(var, axis=0))
        

    # Concatenate data along time axis
    var_data = np.concatenate(var_data_list, axis=0)
    
    # save data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, out_file_name), var_data)

    logging.info(f'Finished saving frames for variable {var_name}')