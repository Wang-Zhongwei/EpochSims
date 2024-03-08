import multiprocessing as mp
import os
from typing import Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sdf
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize

from utils import Plane, Scalar, Species, Vector


def animate(
    field: np.ndarray,
    extent: Tuple[float, float, float, float] = [0, 1, 0, 1],
    norm: Normalize = None,
    cmap="viridis",
    **kwargs,
) -> Tuple[animation.FuncAnimation, Axes, Colorbar]:
    """field animation

    Args:
        field (np.ndarray): 3D array where the first dimension is assumed to be time
        extent (Tuple[float, float, float, float], optional): extent of the 2D plot. Defaults to None.
        norm (Normalize, optional): matplotlib.colors.Normalize. Defaults to [0, 1, 0, 1].
        cmap (str, optional): colormap passed to ax.imshow(). Defaults to "viridis".
        **kwargs: additional keyword arguments to pass to the save method of the FuncAnimation object

    Returns:
        Tuple[animation.FuncAnimation, Axes, Colorbar]: animation, axis, and colorbar
    """
    if norm is None:
        norm = Normalize(vmin=np.min(field), vmax=np.max(field))

    fig, ax = plt.subplots()
    img = ax.imshow(
        field[0].T,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        norm=norm,
        cmap=cmap,
    )
    cbar = fig.colorbar(img)

    # Update function for animation
    def update(i):
        img.set_array(field[i].T)

    ani = animation.FuncAnimation(fig, update, frames=range(len(field)), **kwargs)

    return ani, ax, cbar


def get_phase_space_data(
    inp_dir: str, data_frame: int, species: Species, dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The get_phase_space_data function loads phase space data from a single frame of a particle movie file.

    Args:
        inp_dir (str): input directory
        data_frame (int): frame number
        species (Species): particle species
        dim (int): dimension of the phase space data

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position, velocity and weight of the particles on that dimension.
    """
    assert dim in (1, 2, 3), "dim must be 1, 2, or 3"
    # use sdf.read to avoid thread-related error caused by global variables in sdf_helper.getdata()
    pmovie = sdf.read(os.path.join(inp_dir, f"pmovie_{data_frame:04d}.sdf"))
    x = pmovie.__getattribute__(
        f"Grid_Particles_subset_{species.value}PMovie_{species.value}"
    ).data[dim - 1]
    v = pmovie.__getattribute__(
        f"Particles_V{chr(ord('x') + dim - 1)}_subset_{species.value}PMovie_{species.value}"
    ).data
    weight = pmovie.__getattribute__(
        f"Particles_Weight_subset_{species.value}PMovie_{species.value}"
    ).data
    print(f"Frame {data_frame} loaded")
    return x, v, weight


def get_phase_space_distribution(
    data,
    inp_dir,
    animation_frame,
    interval,
    species,
    dim,
    range,
    bins=1000,
    normed=False,
):
    x, v, weight = get_phase_space_data(
        inp_dir, animation_frame * interval, species, dim
    )
    hist_data, _, _ = np.histogram2d(
        x, v, weights=weight, range=range, bins=bins, normed=normed
    )
    data[animation_frame] = hist_data


def animate_phase_space(inp_dir, out_dir, animation_name, interval, species, **kwargs):
    """The animate_phase_space function generates an animation of a phase space density plot from a series of input files.

    Parameters
        inp_dir (str): The path to the directory containing the input files.
        out_dir (str): The path to the directory where the output animation file will be saved.
        animation_name (str): name of the animation
        interval (int): The number of frames to skip between each loaded frame. This can be used to speed up the animation generation process by skipping frames.
        **kwargs: Additional keyword arguments to pass to the save method of the FuncAnimation object.

    Returns:
        None
    """
    dim = kwargs.pop("dim", 1)
    x_min = kwargs.pop("x_min", -1e-5)
    x_max = kwargs.pop("x_max", 1e-5)
    v_min = kwargs.pop("v_min", -5e7)
    v_max = kwargs.pop("v_max", 5e7)

    num_of_pmovies = sum(1 for f in os.listdir(inp_dir) if f.startswith("pmovie_"))

    # multi-processing is about twice as fast as multi-threading
    length = num_of_pmovies // interval
    dist = [None] * length
    with mp.Pool(processes=40) as pool:
        for animation_frame in range(length):
            pool.apply_async(
                get_phase_space_distribution,
                args=(
                    dist,
                    inp_dir,
                    animation_frame,
                    interval,
                    species,
                    dim,
                    [[x_min, x_max], [v_min, v_max]],
                ),
            )
        pool.close()
        pool.join()

    ani, ax = animate(dist, [x_min, x_max, v_min, v_max], **kwargs)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Phase Space Density Plot")
    ani.save(os.path.join(out_dir, animation_name), **kwargs)


if __name__ == "__main__":
    # load grid data: fix save grid
    from configs.config import *

    os.makedirs(media_folder, exist_ok=True)

    ext = "mp4"  # 'gif' or 'mp4'
    if ext == "gif":
        fps = 10
    elif ext == "mp4":  # need to have ffmpeg installed
        fps = 10

    # grid = np.load(os.path.join(raw_data_folder, "grid.npy"), allow_pickle=True)
    # x_array, y_array, z_array = grid

    for species in (Species.PROTON, Species.ELECTRON, Species.CARBON):
        animate_phase_space(
            lustre_data_path,
            media_folder,
            f"{species.value}_phase_space.mp4",
            5,
            species=species,
            fps=fps,
            dpi=300,
            blit=True,
        )
        # for plane in (Plane.XY, Plane.YZ):
        #     if plane == Plane.XY:
        #         extent = [x_array[0], x_array[-1], y_array[0], y_array[-1]]
        #         x_label = "x"
        #         y_label = "y"
        #     elif plane == Plane.YZ:
        #         extent = [y_array[0], y_array[-1], z_array[0], z_array[-1]]
        #         x_label = "y"
        #         y_label = "z"

        #     for field in (Scalar.NUMBER_DENSITY, Scalar.TEMPERATURE):
        #         quantity = np.load(
        #             os.path.join(
        #                 raw_data_folder, f"{species.value}_{field.value}_{plane.value}.npy"
        #             ),
        #             allow_pickle=True,
        #         )
        #         if field == Scalar.NUMBER_DENSITY:
        #             z_scale = 0.5
        #             cmap = "viridis"
        #         elif field == Scalar.TEMPERATURE:
        #             z_scale = 1
        #             cmap = "jet"

        #         for is_log_scale in (True, False):
        #             if is_log_scale:
        #                 quantity += 1  # to avoid log(0)
        #             ani, ax = animate_field(
        #                 quantity,
        #                 extent,
        #                 vmin=None,
        #                 vmax=None,
        #                 z_scale=z_scale,
        #                 log_scale=is_log_scale,
        #                 cmap=cmap,
        #             )
        #             ax.set_title(f"{species.value} {field.value.split('_')[-1]}")
        #             ax.set_xlabel(x_label)
        #             ax.set_ylabel(y_label)

        #             file_name = f"{species.value}_{field.value.split('_')[-1]}{'_log' if is_log_scale else ''}_{plane.value}.{ext}".lower()

        #             ani.save(os.path.join(media_folder, f"{file_name}"), fps=fps, dpi=300)
        #             print(f"Saved {file_name}")
