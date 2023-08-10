import sdf_helper as sh
import numpy as np
import os
from utils import *


@timer
def save_scalar_frames(
    inp_folder: str,
    frame_slice: slice,
    scalar_field: Scalar = Scalar.NUMBER_DENSITY,
    species: Species = Species.PROTON,
    plane: Plane = Plane.XY,
    out_folder: str = None,
    save_grid: bool = False,
    prefix: str = "smovie"
):
    """Save scalar movie as numpy array

    Args:
        inp_folder (str): input folder
        frame_slice (slice): slice object for frame range
        scalar_field (Scalar, optional): scalar field to save. Defaults to Scalar.NUMBER_DENSITY.
        species (Species, optional): species. Defaults to Species.PROTON.
        plane (Plane, optional): scalar field on which plane. Defaults to Plane.XY.
        out_folder (str, optional): output directory if not specified then return data to memory. Defaults to None.
        save_grid (bool, optional): whether to save grid data. Defaults to False.

    Returns
        Optional[np.ndarray]: scalar movie data if out_folder is None
    """
    start = frame_slice.start if frame_slice.start is not None else 0
    step = frame_slice.step if frame_slice.step is not None else 1
    num_files = len([f for f in os.listdir(inp_folder) if f.startswith(prefix)])
    stop = frame_slice.stop if frame_slice.stop is not None else num_files
    frames = np.arange(start, stop, step)
    out_file_name = (
        f"{species.value}_{scalar_field.value}_{plane.value}.npy"
    )

    # Load grid
    grid = sh.getdata(
        os.path.join(inp_folder, f"{prefix}_{frames[0]:04d}.sdf"), verbose=False
    ).Grid_Grid_mid.data
    x_array, y_array, z_array = grid
    slices = [slice(None), slice(None), slice(None)]
    if plane == Plane.XY:
        slices[2] = np.argmin(np.abs(z_array))
    elif plane == Plane.XZ:
        slices[1] = np.argmin(np.abs(y_array))
    elif plane == Plane.YZ:
        slices[0] = np.argmin(np.abs(x_array))
    slices = tuple(slices)

    # Load data
    var_data_list = []
    var_name = f"{scalar_field.value}_{species.value}"
    for frame in frames:
        data = sh.getdata(os.path.join(inp_folder, f"{prefix}_{frame:04d}.sdf"), verbose=False)
        var = data.__getattribute__(var_name).data
        var_data_list.append(np.expand_dims(var[slices], axis=0))

    # Concatenate data along time axis
    var_data = np.concatenate(var_data_list, axis=0)
    if out_folder is None:
        return var_data
    # save data
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, out_file_name), var_data)
    np.save(os.path.join(out_folder, "grid.npy"), data.Grid_Grid_mid.data) if save_grid else None


if __name__ == "__main__":
    from configs.config import *

    os.makedirs(experiment_folder, exist_ok=True)

    for species in Species:
        for plane in (Plane.XY, Plane.YZ):
            for field in (Scalar.NUMBER_DENSITY, Scalar.TEMPERATURE):
                save_scalar_frames(
                    epoch_output_folder,
                    slice(None),
                    scalar_field=field,
                    species=species,
                    out_folder=raw_data_folder,
                    plane=plane,
                    save_grid=True,
                )
