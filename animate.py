import os
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np
from utils import Plane
import argparse
from plot_helper_2d import animate_data

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="Path to directory containing the input.npy files",
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default=None,
    help="Path to directory to save the media files. Default to input directory",
)

parser.add_argument(
    "-v",
    "--var_names",
    nargs="+",
    type=str,
    help="Names of the variables to animate. For example, 'Electric_Field_Ex'",
)

parser.add_argument(
    "-s",
    "--subset",
    type=str,
    help="Which plane to save. Options are 'XY', 'XZ', 'YZ', or None",
)

# parse inputs
args = parser.parse_args()
input_dir = args.input_dir
if args.output_dir is None:
    output_dir = args.input_dir
else:
    output_dir = args.output_dir
var_names = args.var_names
plane = Plane[args.subset]

# load env.py from analysis_data_dir
# todo: automatically create env.py according to indput.deck
env_path = os.path.join(input_dir, "env.py")
exec(open(env_path).read())

# mapping from var_name to plotting parameters
var_params = {
    "Electric_Field_Ex": {
        "title": "Electric Field Ex",
        "norm": SymLogNorm(linthresh=1e-2, linscale=1),
        "cmap": "bwr",
        "normalization_factor": E0,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$\frac{eE_x}{m_e c\omega}$",
    },
    "Electric_Field_Ey": {
        "title": "Electric Field Ey",
        "norm": SymLogNorm(linthresh=1e-2, linscale=1),
        "cmap": "bwr",
        "normalization_factor": E0,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$\frac{eE_y}{m_e c\omega}$",
    },
    "Electric_Field_Ez": {
        "title": "Electric Field Ez",
        "norm": SymLogNorm(linthresh=1e-2, linscale=1),
        "cmap": "bwr",
        "normalization_factor": E0,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$\frac{eE_z}{m_e c\omega}$",
    },
    "Derived_Charge_Density": {
        "title": "Charge Density",
        "norm": SymLogNorm(linthresh=1e-2, linscale=1),
        "cmap": "bwr",
        "normalization_factor": nc * elementary_charge,
        "smoothing_sigma": 3.0,
        "cbar_label": r"$\frac{\rho}{n_c e}$",
    },
    "Derived_Number_Density_Deuteron": {
        "title": "Deuteron Number Density",
        "norm": LogNorm(vmin=1e-5, vmax=2e1),
        "cmap": "viridis",
        "normalization_factor": nc,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$\frac{n_d}{n_c}$",
    },
    "Derived_Number_Density_Electron": {
        "title": " Electron Number Density",
        "norm": LogNorm(vmin=1e-5, vmax=2e1),
        "cmap": "viridis",
        "normalization_factor": nc,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$\frac{n_e}{n_c}$",
    },
    "Derived_Temperature_Deuteron": {
        "title": "Deuteron Temperature",
        "norm": LogNorm(vmin=1e-5, vmax=2e1),
        "cmap": "plasma",
        "normalization_factor": K_in_MeV,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$T_d$ [MeV]",
    },
    "Derived_Temperature_Electron": {
        "title": " Electron Number Density",
        "norm": LogNorm(vmin=1e-5, vmax=2e1),
        "cmap": "plasma",
        "normalization_factor": K_in_MeV,
        "smoothing_sigma": 0.0,
        "cbar_label": r"$T_e$ [MeV]",
    },
}

# load grid data
grid = np.load(os.path.join(input_dir, "grid.npy"), allow_pickle=True)
if plane == Plane.XY:
    i, j = 0, 1
elif plane == Plane.XZ:
    i, j = 0, 2
else:
    i, j = 1, 2
extent = [grid[i][0], grid[i][-1], grid[j][0], grid[j][-1]]

for var_name in var_names:
    # load data and normalize
    data = np.load(
        os.path.join(input_dir, f"{var_name}_{plane.value}.npy"), allow_pickle=True
    )

    # animate data
    ani, ax, cbar = animate_data(
        data,
        time_stamps,
        extent,
        cmap=var_params[var_name]["cmap"],
        norm=var_params[var_name]["norm"],
        normalization_factor=var_params[var_name]["normalization_factor"],
        smoothing_sigma=var_params[var_name]["smoothing_sigma"],
    )
    cbar.set_label(var_params[var_name]["cbar_label"])
    ax.set_title(var_params[var_name]["title"])
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")

    # save animation
    ani.save(
        os.path.join(output_dir, f"{var_name}_{plane.value}_Movie.mp4"),
        writer="ffmpeg",
        fps=10,
        dpi=300,
    )
