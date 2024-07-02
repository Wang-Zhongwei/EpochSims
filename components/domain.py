import numpy as np

from .enums import Plane


class Domain:
    def __init__(self, boundaries: list, grid_size: list, time_interval: list):
        self.boundaries = boundaries
        self.grid_size = grid_size
        self.time_interval = time_interval

    def __repr__(self):
        return f"Domain(boundaries={self.boundaries} m, grid_size={self.grid_size}, time_interval={self.time_interval} s)"

    def get_spacing_x(self):
        return (self.boundaries[0][1] - self.boundaries[0][0]) / self.grid_size[0]

    def get_spacing_y(self):
        assert len(self.grid_size) >= 2
        return (self.boundaries[1][1] - self.boundaries[1][0]) / self.grid_size[1]

    def get_spacing_z(self):
        assert len(self.grid_size) == 3
        return (self.boundaries[2][1] - self.boundaries[2][0]) / self.grid_size[2]

    def get_grid_coordinates(self):
        dim = len(self.grid_size)
        coordinates = []
        for i in range(dim):
            n = self.grid_size[i]
            coord = np.linspace(self.boundaries[i][0], self.boundaries[i][1], n + 1)
            coordinates.append(coord)
        return tuple(coordinates)

    def get_planar_data(self, original_data: np.ndarray, plane: Plane):
        # todo: move it under quantity class in the future
        assert original_data.ndim == 3 and len(self.grid_size) == 3
        tol = 1e-10
        xs, ys, zs = self.get_grid_coordinates()

        def get_indices(arr):
            sorted_indices = np.argsort(np.abs(arr))
            if np.abs(arr[sorted_indices[0]]) < tol:
                return sorted_indices[:3]
            else:
                return sorted_indices[:2]

        if plane == Plane.XY:
            indices = get_indices(zs)
            return np.mean(original_data[:, :, indices], axis=2)
        elif plane == Plane.XZ:
            indices = get_indices(ys)
            return np.mean(original_data[:, indices, :], axis=1)
        elif plane == Plane.YZ:
            indices = get_indices(xs)
            return np.mean(original_data[indices, :, :], axis=0)

    def get_extent(self, plane: Plane):
        if plane == Plane.XY or plane is None:
            return [
                self.boundaries[0][0],
                self.boundaries[0][1],
                self.boundaries[1][0],
                self.boundaries[1][1],
            ]
        elif plane == Plane.XZ:
            return [
                self.boundaries[0][0],
                self.boundaries[0][1],
                self.boundaries[2][0],
                self.boundaries[2][1],
            ]
        elif plane == Plane.YZ:
            return [
                self.boundaries[1][0],
                self.boundaries[1][1],
                self.boundaries[2][0],
                self.boundaries[2][1],
            ]
        else:
            raise ValueError(f"Invalid plane: {plane}")
