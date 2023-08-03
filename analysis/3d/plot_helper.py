import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm, Normalize


def animate_field(field, extent, vmin=None, vmax=None, scale_reduced=0.5, log_scale=False, cmap='viridis'):
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
    img = ax.imshow(field[0, :, :].T, origin='lower', 
                    extent=extent, 
                    interpolation='nearest', norm=norm, cmap=cmap)
    fig.colorbar(img)

    # Define an update function for the animation
    def update(i):
        # Update the data
        ax.imshow(field[i, :, :].T, origin='lower', 
                       extent=extent,
                       interpolation='nearest', norm=norm, cmap=cmap)

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=range(field.shape[0]), blit=False)

    return ani, ax