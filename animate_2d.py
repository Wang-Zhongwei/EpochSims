import argparse
import logging
import os

from metadata import get_plotting_parameters, get_simulation

# if you are using vscode, add repo directory path to python.analysis.extraPaths in settings.json to enable syntax highlighting for dynamic imports
from plot_helper_2d import animate_quantity
from utils import Quantity

logger = logging.getLogger("animate_2d")
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="Process simulation ids.")
parser.add_argument("simulation_ids", nargs="+", type=str, help="The simulation ids")
args = parser.parse_args()
simulation_ids = args.simulation_ids

default_quantities = [
    Quantity.Ey,
    Quantity.CHARGE_DENSITY,
    Quantity.NUMBER_DENSITY,
    Quantity.TEMPERATURE,
]

for simulation_id in simulation_ids:
    simulation = get_simulation(simulation_id)
    plotting_params = get_plotting_parameters(simulation)
    for quantity in default_quantities:
        quantity_params = plotting_params.get(quantity)
        for species in quantity_params["species"]:
            plot_title = f"{quantity.value}".replace("_", " ")
            if species is not None:
                plot_title = f"{species.value} " + plot_title

            try:
                ani, ax, cbar = animate_quantity(
                    simulation.data_dir_path,
                    quantity_params["file_prefix"],
                    quantity,
                    species=species,
                    normalization_factor=quantity_params["normalization_factor"],
                    norm=quantity_params["norm"],
                    cmap=quantity_params["cmap"],
                    smoothing_sigma=quantity_params["smoothing_sigma"],
                )
                logger.debug(f"Animation created for {plot_title}")

                cbar.set_label(quantity_params["cbar_label"])
                ax.set_title(plot_title)
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")

                filename = plot_title.lower().replace(" ", "_") + "_movie.mp4"
                out_path = os.path.join(simulation.analysis_dir_path, filename)

                ani.save(
                    out_path,
                    writer="ffmpeg",
                    fps=10,
                    dpi=300,
                )
                logger.info(f"Saved animation {filename} to {out_path}")
            except Exception as e:
                logger.error(f"Error creating animation: {e}")
