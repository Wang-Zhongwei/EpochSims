import argparse
import logging
import os

import numpy as np
import sdf_helper as sh

from components import Plane, Quantity, Simulation, Species
from utils import read_quantity_sdf_from_sdf, timer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("process_3d_data")

@timer
def save_frames_from_3d_data(
    simulation: Simulation,
    quantity: Quantity,
    species: Species,
    plane: Plane,
):
    data_list = []
    file_prefix = quantity.get_prefix(simulation.data_dir_path)
    num_frames = sim.get_num_frames(file_prefix)
    for frame in range(num_frames):
        in_file_name = f"{file_prefix}_{frame:04d}.sdf"
        data = sh.getdata(
            os.path.join(simulation.data_dir_path, in_file_name), verbose=False
        )
        logger.info(f"Loaded {in_file_name}")

        attr_name = quantity.get_attribute_name(species)
        quantity_sdf = read_quantity_sdf_from_sdf(data, attr_name)
        
        logger.debug(f"Shape of the quantity_sdf: {quantity_sdf.data.shape}")

        planar_quantity_data = simulation.domain.get_planar_data(
            quantity_sdf.data, plane
        )
        
        logger.debug(f"Shape of the planar_quantity_data: {planar_quantity_data.shape}")
        data_list.append(np.expand_dims(planar_quantity_data, axis=0))

    # Concatenate data along time axis
    data_2_save = np.concatenate(data_list, axis=0)

    # save data
    npy_file_name = quantity.get_npy_file_name(species, plane)
    np.save(os.path.join(simulation.analysis_dir_path, npy_file_name), data_2_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate 2D data from a simulation.")
    parser.add_argument(
        "simulation_ids", nargs="+", type=str, help="The simulation ids"
    )
    args = parser.parse_args()
    simulation_ids = args.simulation_ids

    default_quantities = [
        Quantity.CHARGE_DENSITY,
        Quantity.NUMBER_DENSITY,
        Quantity.TEMPERATURE,
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
        for quantity in default_quantities:
            for species in plotting_params[quantity]["species"]:
                for plane in default_planes:
                    try:
                        save_frames_from_3d_data(
                            sim,
                            quantity,
                            species,
                            plane,
                        )
                    except Exception as e:
                        logger.error(
                            f"Error saving frames for {quantity.value} {species.value} {plane.value}: {e}"
                        )
        
    # save Ex data
    # for simulation_id in simulation_ids:
    #     sim = Simulation.from_simulation_id(simulation_id)
    #     fmovie = sh.getdata(os.path.join(sim.data_dir_path, "fmovie_0000.sdf"), verbose=False)
    #     if hasattr(fmovie, Quantity.Ex_XY.value):
    #         quantity = Quantity.Ex_XY
    #     else:
    #         quantity = Quantity.Ex
    #     save_frames_from_3d_data(sim, quantity, None, Plane.XY)