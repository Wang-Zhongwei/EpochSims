import sdf_helper as sh
import numpy as np
import os
from utils import *


def save_vector_frames(
    inp_folder: str,
    frame_slice: slice,
    vector_field: Vector = Vector.Ey,
    plane: Plane = Plane.XY,
    out_folder: str = None,
):
    prefix = "fmovie"
    start = frame_slice.start if frame_slice.start is not None else 0
    step = frame_slice.step if frame_slice.step is not None else 1
    num_files = len([f for f in os.listdir(inp_folder) if f.startswith(prefix)])
    stop = frame_slice.stop if frame_slice.stop is not None else num_files
    frames = np.arange(start, stop, step)
    out_file_name = f"{vector_field.value}_{plane.value}_{start}_{stop}_{step}.npy"

    slices = [slice(None), slice(None), slice(None)]
    if plane == Plane.XY:
        slices[2] = 0
    elif plane == Plane.XZ:
        slices[1] = 0
    elif plane == Plane.YZ:
        slices[0] = 0
    slices = tuple(slices)

    # load data
    var_data_list = []
    var_name = f"{vector_field.value}_Core_{plane.value}_plane"
    for frame in frames:
        data = sh.getdata(os.path.join(inp_folder, f"{prefix}_{frame:04d}.sdf"), verbose=False)
        var = data.__getattribute__(var_name).data
        var_data_list.append(np.expand_dims(var[slices], axis=0))

    # concatenate data along time axis
    var_data = np.concatenate(var_data_list, axis=0)
    if out_folder is None:
        return var_data

    # save data
    os.makedirs(out_folder, exist_ok=True)
    np.save(os.path.join(out_folder, out_file_name), var_data)


if __name__ == "__main__":
    from configs.config import *

    # save vector movies
    for vector in (Vector.Ex, Vector.Ey, Vector.Ez, Vector.Bx, Vector.By, Vector.Bz):
        for plane in (Plane.XY, Plane.YZ):
            save_vector_frames(
                epoch_output_path, slice(None, None, 10), vector, plane, raw_data_path
            )
