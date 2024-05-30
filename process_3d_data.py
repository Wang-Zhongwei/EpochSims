import argparse
import logging
import os

import numpy as np
import sdf_helper as sh

from configs.metadata import get_plotting_parameters, get_simulation
from utils import Plane, Quantity, Simulation, get_quantity_name, infer_prefix, read_quantity_sdf_from_sdf, timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@timer
def save_frames_from_3d_data(
    simulation: Simulation,
    file_prefix: str,
    quantity_name: str,
    plane: Plane,
):
    var_data_list = []
    num_frames = sim.get_num_frames(file_prefix)
    for frame in range(num_frames):
        in_file_name= f"{file_prefix}_{frame:04d}.sdf"
        data = sh.getdata(
            os.path.join(simulation.data_dir_path, in_file_name), verbose=False
        )
        logger.info(f"Loaded {in_file_name}")
        
        quantity_sdf = read_quantity_sdf_from_sdf(data, quantity_name)
        
        planar_quantity_data = simulation.domain.get_planar_data(quantity_sdf.data, plane)
        var_data_list.append(np.expand_dims(planar_quantity_data, axis=0))

    # Concatenate data along time axis
    data_2_save = np.concatenate(var_data_list, axis=0)

    # save data
    out_file_name = f"{quantity_name}_{plane.value}.npy"
    np.save(os.path.join(simulation.analysis_dir_path, out_file_name), data_2_save)


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
                var_name = get_quantity_name(quantity, species)
                for plane in default_planes:
                    try:
                        save_frames_from_3d_data(
                            sim.data_dir_path,
                            file_prefix,
                            var_name,
                            plane,
                            sim.analysis_dir_path,
                        )
                    except Exception as e:
                        logger.error(f"Error saving frames for {quantity} and {species}: {e}")
