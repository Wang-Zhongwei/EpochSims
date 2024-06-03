
from matplotlib.colors import Normalize
import numpy as np


class Movie:
    def __init__(self, data: np.ndarray, extent: list, timesteps: np.ndarray):
        assert data.ndim == 3, "Movie data must be a 3D array"
        self.data = data
        self.extent = extent
        self.timesteps = timesteps
    
    def animate(self, norm=Normalize(), cmap="viridis", normalization_factor=1.0, smoothing_sigma=0.0):
        # todo: implement animate
        pass