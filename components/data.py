from matplotlib import animation, pyplot as plt
from matplotlib.colors import Normalize
import multiprocessing as mp
import numpy as np
from scipy import ndimage


def gaussian_filter_func(data: np.ndarray, smoothing_sigma: float = 0.0) -> np.ndarray:
    data = ndimage.gaussian_filter(data, sigma=smoothing_sigma)
    return data


class Movie:
    def __init__(self, data: np.ndarray, extent: list, timesteps: np.ndarray):
        assert data.ndim == 3, "Movie data must be a 3D array"
        self.data = data
        self.extent = extent
        self.timesteps = timesteps

    def animate(
        self,
        norm=Normalize(),
        cmap="viridis",
        normalization_factor=1.0,
        smoothing_sigma=0.0,
        **kwargs,
    ):
        # todo: implement animate
        data = self.data
        if smoothing_sigma > 0:
            with mp.Pool(mp.cpu_count()) as pool:
                data = pool.starmap(
                    gaussian_filter_func,
                    [(d, smoothing_sigma) for d in data],
                )

        fig, ax = plt.subplots()
        img = ax.imshow(
            data[0].T / normalization_factor,
            extent=self.extent,
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
            img.set_array(data[i].T / normalization_factor)

            if self.timesteps is not None:
                time_text.set_text(f"t = {self.timesteps[i]:.2e} s")
            else:
                time_text.set_text(f"Frame: {i}")

        ani = animation.FuncAnimation(fig, update, frames=range(len(data)), **kwargs)

        return ani, ax, cbar
