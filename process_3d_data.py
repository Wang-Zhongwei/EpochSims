import argparse
import logging
import os

import numpy as np
import sdf_helper as sh

from components import Plane, Quantity, Simulation
from utils import get_attribute_name, get_prefix, read_quantity_sdf_from_sdf, timer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_3d_data")

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
    # todo: implement get_npy_file_name class method
    npy_file_name = f"{quantity_name}_{plane.value}.npy"
    np.save(os.path.join(simulation.analysis_dir_path, npy_file_name), data_2_save)


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
        sim = Simulation.from_simulation_id(simulation_id)
        plotting_params = sim.get_plotting_parameters()
        # save frames
        for quantity in default_quantities:
            try: 
                file_prefix = get_prefix(sim.data_dir_path, quantity)
            except ValueError as e:
                logger.error(e)
                continue
            
            for species in plotting_params[quantity]["species"]:
                quantity_name = get_attribute_name(quantity, species)
                for plane in default_planes:
                    try:
                        save_frames_from_3d_data(
                            sim,
                            file_prefix,
                            quantity_name,
                            plane,
                        )
                    except Exception as e:
                        logger.error(f"Error saving frames for {quantity_name}: {e}")
