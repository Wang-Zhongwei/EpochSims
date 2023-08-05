from matplotlib import scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize

from utils import Plane, Scalar, Species


def animate_field(
    field, extent, vmin=None, vmax=None, scale_reduced=0.5, log_scale=False, cmap="viridis"
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

    # Create the initial figure and colorbar
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

    # Define an update function for the animation
    def update(i):
        # Update the data
        ax.imshow(
            field[i, :, :].T,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            norm=norm,
            cmap=cmap,
        )

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=range(field.shape[0]), blit=False)

    return ani, ax


if __name__ == "__main__":
    # load grid data: fix save grid
    from configs.config import *

    ext = "mp4"  # 'gif' or 'mp4'
    if ext == "gif":
        fps = 10
    elif ext == "mp4": # need to have ffmpeg installed
        fps = 10

    grid = np.load(os.path.join(raw_data_folder, "grid.npy"), allow_pickle=True)
    x_array, y_array, z_array = grid

    for species in (Species.PROTON, Species.ELECTRON, Species.CARBON):
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
                cmap = "viridis"
                if field == Scalar.NUMBER_DENSITY:
                    scale_reduced = 0.5
                elif field == Scalar.TEMPERATURE:
                    scale_reduced = 1
                    cmap = "jet"
                    quantity += 1

                for is_log_scale in (True, False):
                    ani, ax = animate_field(
                        quantity,
                        extent,
                        vmin=None,
                        vmax=None,
                        scale_reduced=scale_reduced,
                        log_scale=is_log_scale,
                        cmap=cmap,
                    )
                    ax.set_title(field.value)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)

                    file_name = f"{species.value}_{field.value.split('_')[-1]}{'_log' if is_log_scale else ''}_{plane.value}.{ext}".lower()

                    ani.save(os.path.join(media_folder, f"{file_name}"), fps=fps, dpi=300)
                    print(f"Saved {file_name}") 
