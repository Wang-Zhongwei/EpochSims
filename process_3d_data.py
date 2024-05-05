import argparse

from plot_helper_3d import save_frames
from utils import Plane

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="Path to directory containing the input.sdf files",
)

parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="Path to directory to save the output.npy files",
)

parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    help="Prefix of the input.sdf files. Such as 'smovie' is the prefix of 'smovie_0000.sdf'",
)

parser.add_argument(
    "-v",
    "--var_names",
    nargs='+',
    type=str,
    help="Names of the variables to save. For example, 'Derived_Charge_Density'",
)

parser.add_argument(
    "-s",
    "--subset",
    type=str,
    help="Which plane to save. Options are 'XY', 'XZ', 'YZ', or None",
)

parser.add_argument(
    "-g",
    "--save_grid",
    action="store_true",
    help="Whether to save the grid.npy file",
)

# parse arguments
args = parser.parse_args()
input_dir = args.input_dir
prefix = args.prefix
var_names = args.var_names
output_dir = args.output_dir
if args.subset:
    subset = Plane[args.subset]
else:
    subset = None
save_grid = args.save_grid

# save frames
for var_name in args.var_names:
    save_frames(
        input_dir,
        prefix,
        var_name,
        output_dir,
        subset,
        save_grid,
    )