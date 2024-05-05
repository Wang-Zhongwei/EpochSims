import multiprocessing as mp
import os
from typing import List, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sdf
import sdf_helper as sh
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from scipy import ndimage

from utils import Scalar, Species, Vector, timer

@timer
def animate_data(
    data: np.ndarray,
    time_stamps: List[float] = None,
    extent: Tuple[float, float, float, float] = [0, 1, 0, 1],
    norm: Normalize = Normalize(),
    cmap="viridis",
    **kwargs,
) -> Tuple[animation.FuncAnimation, Axes, Colorbar]:
    """animate data over time on a 2D plot

    Args:
        data (np.ndarray): 3D array where the first dimension is assumed to be time
        time_stamps (List[float], optional): time stamps of each instant of data. Defaults to None.
        extent (Tuple[float, float, float, float], optional): extent of the 2D plot. Defaults to None.
        norm (Normalize, optional): matplotlib.colors.Normalize. Defaults to [0, 1, 0, 1].
        cmap (str, optional): colormap passed to ax.imshow(). Defaults to "viridis".
        **kwargs: additional keyword arguments to pass to the save method of the FuncAnimation object

    Returns:
        Tuple[animation.FuncAnimation, Axes, Colorbar]: animation, axis, and colorbar
    """
    fig, ax = plt.subplots()
    img = ax.imshow(
        data[0].T,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        norm=norm,
        cmap=cmap,
    )
    cbar = fig.colorbar(img)

    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=12,
        color="white",
        backgroundcolor="black",
    )

    # Update function for animation
    def update(i):
        img.set_array(data[i].T)

        if time_stamps is not None:
            time_text.set_text(f"t = {time_stamps[i]:.2e} s")
        else:
            time_text.set_text(f"Frame: {i}")

    ani = animation.FuncAnimation(fig, update, frames=range(len(data)), **kwargs)

    return ani, ax, cbar

def read_quantity_from_sdf(
    sdf: sdf.BlockList,
    field: Union[Scalar, Vector],
    species: Optional[Species] = None,
) -> sdf.BlockList:
    if species is not None:
        if species == Species.ALL:
            return sdf.__getattribute__(f"{field.value}")
        else:
            return sdf.__getattribute__(f"{field.value}_{species.value}")
    else:
        return sdf.__getattribute__(f"{field.value}")


def transform_func(
    data: np.ndarray, normalization_factor: float = 1.0, smoothing_sigma: float = 0.0
) -> np.ndarray:
    data /= normalization_factor
    if smoothing_sigma > 0:
        data = ndimage.gaussian_filter(data, sigma=smoothing_sigma)
    return data


def animate_field(
    input_dir: str,
    field: Union[Scalar, Vector],
    species: Optional[Species] = None,
    file_prefix: Optional[str] = None,
    normalization_factor: float = 1.0,
    smoothing_sigma: float = 0.0,
    **kwargs,
) -> Tuple[animation.FuncAnimation, Axes, Colorbar]:
    if file_prefix is None:
        file_prefix = "fmovie" if species is None else "smovie"

    num_frames = len([f for f in os.listdir(input_dir) if f.startswith(file_prefix)])

    # load data
    data, time_stamps = [], []
    for i in range(num_frames):
        sdf = sh.getdata(
            os.path.join(input_dir, f"{file_prefix}_{i:04d}.sdf"), verbose=False
        )
        quantity = read_quantity_from_sdf(sdf, field, species)
        data.append(quantity.data)
        time_stamps.append(sdf.Header["time"])

    # get domain extent
    grid = quantity.grid.data
    extent = []
    for i in range(len(grid)):
        extent.append(grid[i][0])
        extent.append(grid[i][-1])

    # normalize and smooth data
    with mp.Pool(mp.cpu_count()) as pool:
        data = pool.starmap(
            transform_func, [(d, normalization_factor, smoothing_sigma) for d in data]
        )

    return animate_data(
        data,
        time_stamps=time_stamps,
        extent=extent,
        **kwargs,
    )


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

    ani, ax = animate_data(dist, extent=[x_min, x_max, v_min, v_max], **kwargs)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Phase Space Density Plot")
    ani.save(os.path.join(out_dir, animation_name), **kwargs)
