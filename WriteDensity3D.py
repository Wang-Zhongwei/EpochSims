import numpy as np
import os
import scipy.optimize as opt
from configs.config import * 

# Get the axis data.
output_dir = INPUT_BASE

x_array = np.arange(-9.992, 10.008, 0.016)
y_array = np.arange(-8.985, 18.015, 0.03)
z_array = np.arange(-8.985, 9.015, 0.03)
print(x_array.shape, y_array.shape, z_array.shape)

# ------------------------------------------------------------------------------
# Define useful constants

# No expansion number densitities.
n_elec = 3.274368e29  # m^-3
n_prot = 5.60094e28  # m^-3
n_carb = 4.52379e28  # m^-3
nc = 1.7343954e27

q_e = 1.6022e-19  # C
epsilon_0 = 8.854e-12  # F/m
m_e = 9.1094e-31  # kg
c = 299792458  # m/s

wp = np.sqrt(n_elec * q_e**2 / (epsilon_0 * m_e)) * 1e3
skin_depth = c / wp


from typing import List
# ------------------------------------------------------------------------------


def half_gaussian_density(x_array: np.ndarray, T: float, x0: float, sigma: float) -> np.ndarray:
    """Returns a density profile where x <= x0 is expanded as a Gaussian and
    x >= x0 is unexpanded.
    ------------------------------------------------------
    x_array: array of x-coordinates to compute the density, in um
    T: float, initial no expansion target thickness, in um. The target is
        assumed to be centered at x = 0.
    x0: float, loacation of the maximum of Gaussian, in um
    sigma: float, standard deviation of density profile"""

    # Compute the gaussian normalization.
    assert -T / 2 < x0 < T / 2
    d = T / 2 + x0
    A = d * np.sqrt(2 / (np.pi * sigma**2))

    output = np.empty(len(x_array))

    for i, x in enumerate(x_array):
        if x <= x0:
            output[i] = A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
        elif x0 < x <= T / 2:
            output[i] = 1
        else:
            output[i] = 0

    # Normalize the gaussian region again
    norm = np.trapz(output, x_array) / T

    return output / norm



def gaussian(x, A, x0, sigma):
    return A / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


T = 2.4
length = -0.03
L = 0.505
density = half_gaussian_density(x_array, T, (-T / 2 - length), L)

dx = x_array[1] - x_array[0]

areal_density = T * n_elec * 1e-6
expanded_areal_density = np.trapz(density, x_array) * n_elec * 1e-6
print(f"Unexpanded Areal Density: {areal_density} m^-2")
print(f"Expanded Areal Density: {expanded_areal_density} m^-2")

# Set to zero below 0.01*nc
ChangeMe = [value < 0.01 for value in density * n_elec / nc]
density[ChangeMe] = 0

print(np.amax(density * n_elec))

def write_density(
    save_path, x_data, y_array, z_array, y_min=None, y_max=None, z_min=None, z_max=None
):
    """Writes a density file with the given data.
    ------------------------------------------------------
    save_path: str, path to save the file
    x_data: array of density values
    y_array: array of y-coordinates
    z_array: array of z-coordinates
    y_min: float, minimum y-coordinate to include in the file
    y_max: float, maximum y-coordinate to include in the file
    z_min: float, minimum z-coordinate to include in the file
    z_max: float, maximum z-coordinate to include in the file"""

    if y_min is None:
        y_min = np.amin(y_array) + 1
    if y_max is None:
        y_max = np.amax(y_array) - 1

    if z_min is None:
        z_min = np.amin(z_array) + 1
    if z_max is None:
        z_max = np.amax(z_array) - 1

    SaveMe = np.zeros((len(x_array), len(y_array), len(z_array)), dtype=float)

    for k, z in enumerate(z_array):
        if z_min <= z <= z_max:
            for j, y in enumerate(y_array):
                if y_min <= y <= y_max:
                    SaveMe[:, j, k] = x_data

    # Write file
    with open(save_path, "wb") as f:
        # SaveMe = SaveMe.reshape(SaveMe.shape[::-1], order = 'F')
        SaveMe.T.tofile(f)


# ------------------------------------------------------------------------------
# Write data files
elec_file = os.path.join(output_dir, "ElecDens.dat")
prot_file = os.path.join(output_dir, "ProtDens.dat")
carb_file = os.path.join(output_dir, "CarbDens.dat")
write_density(elec_file, n_elec * density, y_array, z_array, y_max=13)
write_density(prot_file, density * n_prot, y_array, z_array, y_max=13)
write_density(carb_file, density * n_carb, y_array, z_array, y_max=13)
