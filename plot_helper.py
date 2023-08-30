import multiprocessing as mp
import concurrent.futures
from operator import truediv
from tkinter import E, Grid
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize
import sdf_helper as sh
import sdf

from utils import Plane, Scalar, Species


def animate_field(
    field,
    extent,
    vmin=None,
    vmax=None,
    scale_reduced=0.5,
    log_scale=False,
    cmap="viridis",
    interval=10,
):
    """Animate scalar field evolution

    Args:
        field (np.ndarray): Field of the species, including scalar field and component of vector field. First dimension if time, second and third are space.
        extent (list like): extent of the plot (x_min, x_max, y_min, y_max)
        vmin (float, optional): Minimum value of the colorbar. Defaults to None.
        vmax (float, optional): Maximum value of the colorbar. Defaults to None.
        reduced_factor (float, optional): Factor by which to reduce the maximum value of the colorbar. Defaults to 0.5.
        log_scale (bool, optional): Whether to use a log scale for the colorbar. Defaults to False.
        cmap (str, optional): Colormap to use. Defaults to 'viridis'.
        interval (int, optional): Interval between frames in fs. Defaults to 10.
    Returns:
        ani, ax: animation and axis object for downstream use
    """
    vmin = np.min(field) if vmin is None else vmin
    # Hardly any particles have density > 0.5 * max density. Adjust this as you prefer
    vmax = np.max(field) * scale_reduced if vmax is None else vmax

    if log_scale:
        norm = LogNorm(vmin=vmin if vmin > 0 else 1, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    img = ax.imshow(
        field[0, :, :].T,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        norm=norm,
        cmap=cmap,
    )
    fig.colorbar(img)

    # Update function for animation
    def update(i):
        img.set_array(field[i, :, :].T)
        # img.autoscale()
        if hasattr(ax, "frame_text"):
            ax.frame_text.remove()
        ax.frame_text = ax.text(
            0.95,
            0.95,
            f"{interval*i} fs",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
        )
        return (img, ax.frame_text)

    ani = animation.FuncAnimation(fig, update, frames=range(field.shape[0]), blit=True)

    return ani, ax


def get_data(inp_dir, frame, species, dim=0):
    assert dim in (0, 1, 2), "dim must be 0, 1, or 2"
    # use sdf.read to avoid thread-related error caused by global variables in sdf_helper.getdata()
    pmovie = sdf.read(os.path.join(inp_dir, f"pmovie_{frame:04d}.sdf"))
    q = pmovie.__getattribute__(
        f"Grid_Particles_subset_{species.value}PMovie_{species.value}"
    ).data[dim]
    v = pmovie.__getattribute__(
        f"Particles_V{chr(ord('x') + dim)}_subset_{species.value}PMovie_{species.value}"
    ).data
    weight = pmovie.__getattribute__(
        f"Particles_Weight_subset_{species.value}PMovie_{species.value}"
    ).data
    print(f"Frame {frame} loaded")
    return q, v, weight


def animate_phase_space(
    inp_dir, out_dir, file_name, interval, species=Species.PROTON, dim=0, **kwargs
):
    """The animate_phase_space function generates an animation of a phase space density plot from a series of input files.

    Parameters
        inp_dir (str): The path to the directory containing the input files.
        out_dir (str): The path to the directory where the output animation file will be saved.
        file_name (str): The name of the output animation file.
        interval (int): The number of frames to skip between each loaded frame. This can be used to speed up the animation generation process by skipping frames.
        **kwargs: Additional keyword arguments to pass to the save method of the FuncAnimation object.

    Returns:
        None
    """
    num_of_pmovies = sum(1 for f in os.listdir(inp_dir) if f.startswith("pmovie_"))

    # multi-processing is about twice as fast as multi-threading
    with mp.Pool(processes=40) as pool:
        futures = {
            frame: pool.apply_async(get_data, args=(inp_dir, frame, species, dim))
            for frame in range(0, num_of_pmovies, interval)
        }
        # with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor: # change max_workers
        #     futures = {
        #         frame: executor.submit(get_data, inp_dir, frame, species, dim)
        #         for frame in range(0, num_of_pmovies, interval)
        #     }

        q, v, weight = futures[0].get()
        hist_data, _, _ = np.histogram2d(
            q, v, weights=weight, range=[[-1e-5, 1e-5], [-5e7, 5e7]], bins=1000, normed=False
        )
        fig, ax = plt.subplots()
        im = ax.imshow(
            hist_data.T + 1,
            extent=[-1e-5, 1e-5, -5e7, 5e7],
            aspect="auto",
            origin="lower",
            norm=LogNorm(),
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title(f"Phase Space Density Plot t=0 fs")

        # Add a colorbar legend
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Density")

        def update(frame):
            try:
                q, v, weight = futures[frame].get()
                del futures[frame]
            except Exception as e:
                print(f"Frame {frame} failed")
                print(e)
                return (im,)

            hist_data, _, _ = np.histogram2d(
                q, v, weights=weight, range=[[-1e-5, 1e-5], [-5e7, 5e7]], bins=1000, normed=False
            )
            im.set_array(hist_data.T + 1)
            # im.autoscale()
            ax.set_title(f"Phase Space Density Plot t={frame} fs")

            print(f"Frame {frame} updated")
            return (im,)

        ani = animation.FuncAnimation(
            fig, update, frames=range(0, num_of_pmovies, interval), blit=True
        )
        ani.save(os.path.join(out_dir, file_name), **kwargs)


if __name__ == "__main__":
    # load grid data: fix save grid
    from configs.config import *

    os.makedirs(media_folder, exist_ok=True)

    ext = "mp4"  # 'gif' or 'mp4'
    if ext == "gif":
        fps = 10
    elif ext == "mp4":  # need to have ffmpeg installed
        fps = 10

    grid = np.load(os.path.join(raw_data_folder, "grid.npy"), allow_pickle=True)
    x_array, y_array, z_array = grid

    for species in (Species.PROTON, Species.ELECTRON, Species.CARBON):
        animate_phase_space(
            epoch_output_dir,
            media_folder,
            f"{species.value}_phase_space.mp4",
            5,
            species=species,
            fps=fps,
            dpi=300,
        )
        for plane in (Plane.XY, Plane.YZ):
            if plane == Plane.XY:
                extent = [x_array[0], x_array[-1], y_array[0], y_array[-1]]
                x_label = "x"
                y_label = "y"
            elif plane == Plane.YZ:
                extent = [y_array[0], y_array[-1], z_array[0], z_array[-1]]
                x_label = "y"
                y_label = "z"

            for field in (Scalar.NUMBER_DENSITY, Scalar.TEMPERATURE):
                quantity = np.load(
                    os.path.join(
                        raw_data_folder, f"{species.value}_{field.value}_{plane.value}.npy"
                    ),
                    allow_pickle=True,
                )
                if field == Scalar.NUMBER_DENSITY:
                    scale_reduced = 0.5
                    cmap = "viridis"
                elif field == Scalar.TEMPERATURE:
                    scale_reduced = 1
                    cmap = "jet"

                for is_log_scale in (True, False):
                    if is_log_scale:
                        quantity += 1  # to avoid log(0)
                    ani, ax = animate_field(
                        quantity,
                        extent,
                        vmin=None,
                        vmax=None,
                        scale_reduced=scale_reduced,
                        log_scale=is_log_scale,
                        cmap=cmap,
                    )
                    ax.set_title(f"{species.value} {field.value.split('_')[-1]}")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)

                    file_name = f"{species.value}_{field.value.split('_')[-1]}{'_log' if is_log_scale else ''}_{plane.value}.{ext}".lower()

                    ani.save(os.path.join(media_folder, f"{file_name}"), fps=fps, dpi=300)
                    print(f"Saved {file_name}")
