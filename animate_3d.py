import logging
import os
import numpy as np
from configs.metadata import get_plotting_parameters, get_simulation
from utils import Plane, Quantity, Simulation, Species, get_plot_title, get_quantity_name
import argparse
from animate_2d import animate_data

logger = logging.getLogger("animate_3d")
logger.setLevel(logging.INFO)

def load_data_from_npy(simulation: Simulation, quantity: Quantity, species: Species, plane: Plane):
    quantity_name = get_quantity_name(quantity, species)
    # todo: implement get_npy_file_name class method
    npy_file_name = f"{quantity_name}_{plane.value}.npy"
    return np.load(os.path.join(simulation.analysis_dir_path, npy_file_name), allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("simulation_ids", nargs="+", type=str, help="The simulation ids")
    args = parser.parse_args()
    simulation_ids = args.simulation_ids

    default_quantities = [
        Quantity.Ex,
        Quantity.CHARGE_DENSITY,
        Quantity.NUMBER_DENSITY,
        Quantity.TEMPERATURE,
        Quantity.Px
    ]

    default_planes = [
        Plane.XY,
        Plane.YZ,
    ]
    
    for sim_id in simulation_ids:
        sim = get_simulation(sim_id)
        plotting_params = get_plotting_parameters(sim)
        for quantity in default_quantities:
            # timesteps = sim.get_output_timesteps()
            timesteps = np.linspace(0, 500e-15, 51)
            quantity_params = plotting_params.get(quantity)
            for species in quantity_params["species"]:
                for plane in default_planes:
                    try:
                        plot_title = get_plot_title(quantity, species, plane)  
                        data = load_data_from_npy(sim, quantity, species, plane)
                        extent = sim.domain.get_extent(plane)
                        ani, ax, cbar = animate_data(
                            data,
                            timesteps,
                            extent=extent,
                            norm=quantity_params["norm"],
                            cmap=quantity_params["cmap"],
                            normalization_factor=quantity_params["normalization_factor"],
                            smoothing_sigma=quantity_params["smoothing_sigma"]
                        )
                        
                        cbar.set_label(quantity_params["cbar_label"])
                        ax.set_title(plot_title)
                        ax.set_xlabel("x [m]")
                        ax.set_ylabel("y [m]")
                        
                        filename = plot_title.lower().replace(" ", "_") + "_movie.mp4"
                        out_path = os.path.join(sim.analysis_dir_path, filename)

                        ani.save(
                            out_path,
                            writer="ffmpeg",
                            fps=5,
                            dpi=300,
                        )
                        
                        logger.info(f"Saved animation {filename} to {out_path}")
                    except Exception as e:
                        logger.error(f"Failed to animate due to: {e}")
                        continue
