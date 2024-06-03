import argparse
import logging
import os

from components import Plane, Quantity, Simulation

logger = logging.getLogger("animate")
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("simulation_ids", nargs="+", type=str, help="The simulation ids")
args = parser.parse_args()
simulation_ids = args.simulation_ids

default_quantities = [
    # Quantity.Ex,
    Quantity.CHARGE_DENSITY,
    Quantity.NUMBER_DENSITY,
    Quantity.TEMPERATURE,
    Quantity.Px
]

for sim_id in simulation_ids:
    sim = Simulation.from_simulation_id(sim_id)
    plotting_params = sim.get_plotting_parameters()
    for quantity in default_quantities:
        quantity_params = plotting_params.get(quantity)
        for species in quantity_params["species"]:
            if sim.dimension == 2:
                default_planes = [None]
            else:
                default_planes = [Plane.XY, Plane.YZ]
                
            for plane in default_planes:
                try:
                    plot_title = quantity.get_plot_title(species, plane)
                    movie = sim.load_movie(quantity, species, plane)

                    ani, ax, cbar = movie.animate(
                        norm=quantity_params["norm"],
                        cmap=quantity_params["cmap"],
                        normalization_factor=quantity_params["normalization_factor"],
                        smoothing_sigma=quantity_params["smoothing_sigma"]
                    )
                    
                    cbar.set_label(quantity_params["cbar_label"])
                    ax.set_title(plot_title)
                    ax.set_xlabel("x [m]")
                    ax.set_ylabel("y [m]")
                    
                    filename = quantity.get_media_file_name(species, plane)
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
