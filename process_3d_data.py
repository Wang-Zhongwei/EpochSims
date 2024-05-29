import argparse
import logging
import os

import numpy as np
import sdf_helper as sh

from configs.metadata import get_plotting_parameters, get_simulation
from utils import Plane, Quantity, get_var_name, infer_prefix, timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@timer
def save_frames(
    input_dir: str,
    file_prefix: str,
    var_name: str,
    plane: Plane,
    output_dir: str,
):
    logging.info(f"Starting to save frames for variable {quantity.value}")

    # 1. Get the slice for the plane
    # Check if grid.npy exists in the input_dir
    grid_file_path = os.path.join(output_dir, "grid.npy")
    if os.path.isfile(grid_file_path):
        # Load the data from grid.npy into grid
        grid = np.load(grid_file_path, allow_pickle=True)
    else:
        grid_file_prefix = infer_prefix(input_dir, "Grid_Grid")
        if grid_file_prefix is None:
            raise ValueError(
                f"Could not infer file prefix for grid data in {input_dir}"
            )

        grid = sh.getdata(
            os.path.join(input_dir, f"{grid_file_prefix}_0000.sdf"), verbose=False
        ).Grid_Grid.data

        # Save the data to grid.npy
        np.save(grid_file_path, grid)

    x_array, y_array, z_array = grid
    slices = [slice(None), slice(None), slice(None)]
    if plane == Plane.XY:
        slices[2] = np.argmin(np.abs(z_array))
    elif plane == Plane.XZ:
        slices[1] = np.argmin(np.abs(y_array))
    elif plane == Plane.YZ:
        slices[0] = np.argmin(np.abs(x_array))
    slices = tuple(slices)

    # 2. Load the data
    num_frames = len([f for f in os.listdir(input_dir) if f.startswith(file_prefix)])
    out_file_name = f"{var_name}_{plane.value}.npy"
    var_data_list = []
    for frame in range(num_frames):
        logger.info(f"Loading frame {frame}")
        data = sh.getdata(
            os.path.join(input_dir, f"{file_prefix}_{frame:04d}.sdf"), verbose=False
        )
        
        if not hasattr(data, var_name):
            raise AttributeError(f"Variable {var_name} not found in the data")
        
        var = data.__getattribute__(var_name).data
        del data
        if plane:
            var = var[slices]
        var_data_list.append(np.expand_dims(var, axis=0))

    # Concatenate data along time axis
    var_data = np.concatenate(var_data_list, axis=0)

    # save data
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, out_file_name), var_data)

    logger.info(f"Finished saving frames for variable {quantity.value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate 2D data from a simulation.")
    parser.add_argument("simulation_ids", nargs="+", type=str, help="The simulation ids")
    args = parser.parse_args()
    simulation_ids = args.simulation_ids

    default_quantities = [
        Quantity.CHARGE_DENSITY,
        Quantity.NUMBER_DENSITY,
        Quantity.TEMPERATURE,
        Quantity.Ex
    ]

    default_planes = [
        Plane.XY,
        Plane.YZ,
    ]
    args = parser.parse_args()
    simulation_ids = args.simulation_ids

    for simulation_id in simulation_ids:
        sim = get_simulation(simulation_id)
        plotting_params = get_plotting_parameters(sim)
        # save frames
        for quantity in default_quantities:
            file_prefix = infer_prefix(sim.data_dir_path, quantity.value)
            if file_prefix is None:
                logger.error(f"Could not infer file prefix for {quantity.value}. Skipped. ")
                continue
            
            for species in plotting_params[quantity]["species"]:
                var_name = get_var_name(quantity, species)
                for plane in default_planes:
                    try:
                        save_frames(
                            sim.data_dir_path,
                            file_prefix,
                            var_name,
                            plane,
                            sim.analysis_dir_path,
                        )
                    except Exception as e:
                        logger.error(f"Error saving frames for {quantity} and {species}: {e}")
