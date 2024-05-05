import logging
import os
from typing import Union

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
    output_dir: str = None,
    subset: Union[None, Plane] = None,
    save_grid: bool = False,
):

    logging.info(f'Starting to save frames for variable {var_name}')

    if subset is not None or save_grid:
        grid = sh.getdata(
            os.path.join(input_dir, f"{file_prefix}_0000.sdf"), verbose=False
        ).Grid_Grid_mid.data

        if save_grid:
            np.save(os.path.join(output_dir, "grid.npy"), grid)

        if subset is not None:
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