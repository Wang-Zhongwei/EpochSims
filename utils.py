import logging
import os
import sys
import time
from enum import Enum
from math import cos, pi
from typing import Optional

import numpy as np
import sdf
import sdf_helper as sh
from scipy.constants import elementary_charge, epsilon_0, m_e, speed_of_light

logging.basicConfig(level=logging.INFO)


class Plane(Enum):
    XY = "XY"
    XZ = "XZ"
    YZ = "YZ"
    XYZ = "XYZ"


class Species(Enum):
    ELECTRON = "Electron"
    PROTON = "Proton"
    DEUTERON = "Deuteron"
    HYDROGEN = "Hydrogen"
    CARBON = "Carbon"


class Quantity(Enum):
    NUMBER_DENSITY = "Derived_Number_Density"
    TEMPERATURE = "Derived_Temperature"
    CHARGE_DENSITY = "Derived_Charge_Density"
    AVG_PARTICLE_ENERGY = "Derived_Average_Particle_Energy"
    Ex = "Electric_Field_Ex"
    Ey = "Electric_Field_Ey"
    Ez = "Electric_Field_Ez"
    Bx = "Magnetic_Field_Bx"
    By = "Magnetic_Field_By"
    Bz = "Magnetic_Field_Bz"
    Sx = "Derived_Poynting_Flux_x"
    Sy = "Derived_Poynting_Flux_y"
    Sz = "Derived_Poynting_Flux_z"
    Px = "Derived_Particles_Average_Px"


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function execution
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time after function execution

        logging.debug(
            f"Function {func.__name__} took {end_time - start_time:.3f} seconds to execute."
        )

        return result

    return wrapper


def get_tnsa_data(pmovie: sdf.BlockList) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(pmovie, "Particles_Ek_subset_TNSA_Hydrogen"):
        energy = pmovie.Particles_Ek_subset_TNSA_Hydrogen
    elif hasattr(pmovie, "Particles_Ek_subset_TNSA_Deuteron"):
        energy = pmovie.Particles_Ek_subset_TNSA_Deuteron
    elif hasattr(pmovie, "Particles_Ek_subset_TNSA_Deuteron_Deuteron"):
        energy = pmovie.Particles_Ek_subset_TNSA_Deuteron_Deuteron
    else:
        raise ValueError("No TNSA energy data found")

    if hasattr(pmovie, "Particles_Weight_subset_TNSA_Hydrogen"):
        weight = pmovie.Particles_Weight_subset_TNSA_Hydrogen
    elif hasattr(pmovie, "Particles_Weight_subset_TNSA_Deuteron"):
        weight = pmovie.Particles_Weight_subset_TNSA_Deuteron
    elif hasattr(pmovie, "Particles_Weight_subset_TNSA_Deuteron_Deuteron"):
        weight = pmovie.Particles_Weight_subset_TNSA_Deuteron_Deuteron
    else:
        raise ValueError("No TNSA weight data found")

    return energy.data, weight.data


def infer_prefix(dir_path, var_name):
    files = os.listdir(dir_path)
    prefixes = set(
        [
            f.split("_")[0]
            for f in files
            if f.endswith(".sdf") and not f.startswith("restart")
        ]
    )
    for prefix in prefixes:
        sdf = sh.getdata(os.path.join(dir_path, f"{prefix}_0000.sdf"), verbose=False)
        if hasattr(sdf, var_name):
            return prefix
    return None


def get_var_name(quantity: Quantity, species: Optional[Species]):
    quantity_name = f"{quantity.value}"
    if species is not None:
        quantity_name += f"_{species.value}"
    return quantity_name


class GaussianBeam:
    def __init__(
        self,
        I_0: float,
        lambda_0: float,
        w_0: float,
        tau: float,
        incidence_angle: float = 0,
        focus_x: float = 0,
        focus_y: float = 0,
        dim: float = 3,
    ):
        """

        Args:
            I_0 (float): Beam intensity in W/cm^2
            lambda_0 (float): Beam wavelength in m
            w_0 (float): Beam waist in m
            tau (float): Pulse duration. Defaults to None.
            incidence_angle (float, optional): Incidence angle of the beam in degrees. Defaults to 0.
            focus_x (float, optional): Focus x-coordinate. Defaults to 0.
            focus_y (float, optional): Focus y-coordinate. Defaults to 0.
            dim (int, optional): Dimension. Defaults to 3.
        """
        self.I_0 = I_0
        self.lambda_0 = lambda_0
        self.w_0 = w_0
        self.tau = tau
        self.incidence_angle = incidence_angle
        self.incidence_theta = incidence_angle * pi / 180
        self.focus_x = focus_x
        self.focus_y = focus_y
        self.dim = dim

    @property
    def beam_energy(self):
        """
        Returns:
            float: Beam energy in Joule
        """
        if self.tau is None:
            raise ValueError("Pulse duration tau is required to calculate beam energy")
        I_0 = self.I_0 * 1e4
        return pi * self.w_0**2 * I_0 * self.tau

    @property
    def peak_electric_field(self):
        """
        Returns:
            float: Peak electric field at the waist of the laser beam in V/m
        """
        I_0 = self.I_0 * 1e4
        return np.sqrt(2 * I_0 / epsilon_0 / speed_of_light)

    @property
    def angular_frequency(self):
        """
        Returns:
            float: Angular Frequency in rad/s
        """
        lambda_0 = self.lambda_0
        return 2 * pi * speed_of_light / lambda_0

    @property
    def critical_density(self):
        """

        Returns:
            float: Critical plasma density for the laser beam in 1/m^3. The critical density is the
            density at which the plasma frequency is equal to the frequency of the laser.
            Above this density, the plasma becomes opaque to the laser light.
        """
        return self.angular_frequency**2 * m_e * epsilon_0 / elementary_charge**2

    @property
    def normalized_amplitude(self):
        return m_e * speed_of_light * self.angular_frequency / elementary_charge

    def calc_beam_radius(self, z: float):
        """

        Args:
            z (float): Axial distance from beam waist in m

        Returns:
            float: Beam radius at the given axial distance in m
        """
        x_R = pi * self.w_0**2 / self.lambda_0
        return self.w_0 * (1 + (z / x_R) ** 2) ** 0.5

    def calc_z(self, w: float):
        """

        Args:
            w (float): Beam radius in m

        Returns:
            float: Axial distance from beam waist in m
        """
        x_R = pi * self.w_0**2 / self.lambda_0
        return x_R * ((w / self.w_0) ** 2 - 1) ** 0.5

    def calc_intensity(self, z: float, r: float = 0):
        """

        Args:
            z (float): Axial distance from beam waist in m
            r (float): Radial distance from beam axis in m

        Returns:
            float: Beam intartensity at the given axial and radial distance in W/cm^2
        """
        w = self.calc_beam_radius(z)
        return self.I_0 * (self.w_0 / w) ** (self.dim - 1) * np.exp(-((r / w) ** 2))

    def __repr__(self):
        return f"GaussianBeam(I_0={self.I_0} W/cm^2, lambda_0 = {self.lambda_0:.2e} m, w_0 = {self.w_0:.2e} m"


class Domain:
    def __init__(self, boundaries: list, grid_size: list, time_interval: list):
        self.boundaries = boundaries
        self.grid_size = grid_size
        self.time_interval = time_interval

    def __repr__(self):
        return f"Domain(boundaries={self.boundaries} m, grid_size={self.grid_size}, time_interval={self.time_interval} s)"


class Target:
    def __init__(
        self,
        material: str,
        electron_number_density: float,
        thickness: float,
        has_preplasma: bool = False,
    ):
        self.material = material
        self.electron_number_density = electron_number_density
        self.thickness = thickness
        self.has_preplasma = has_preplasma

    @property
    def plasma_frequency(self):
        return np.sqrt(
            self.electron_number_density * elementary_charge**2 / (m_e * epsilon_0)
        )

    def __repr__(self):
        return f"Target('{self.material}', {self.electron_number_density}, {self.thickness}, {self.has_preplasma})"


class Simulation:
    def __init__(
        self,
        simulation_id: str,
        data_dir_path: str,
        analysis_dir_path: str,
        laser: GaussianBeam,
        domain: Domain,
        target: Target,
    ) -> None:
        self.simulation_id = simulation_id
        self.data_dir_path = data_dir_path
        self.analysis_dir_path = analysis_dir_path
        self.laser = laser
        self.domain = domain
        self.target = target

        if not os.path.exists(self.data_dir_path):
            logging.error(f"Data directory {self.data_dir_path} does not exist")
            sys.exit(1)
        try:
            os.makedirs(self.analysis_dir_path, exist_ok=True)
        except PermissionError as e:
            logging.error(
                f"No permission for creating analysis directory {self.analysis_dir_path}"
            )
            sys.exit(1)
        except Exception as e:
            logging.error(
                f"Error creating analysis directory {self.analysis_dir_path}: {e}"
            )
            sys.exit(1)

    def __repr__(self):
        return f"Simulation(simulation_id='{self.simulation_id}', laser={repr(self.laser)}, domain={repr(self.domain)}, target={repr(self.target)})"

    def calc_beam_radius_on_target(self):
        axial_distance = -self.laser.focus_x / cos(self.laser.incidence_theta)
        return self.laser.calc_beam_radius(axial_distance)

    def calc_beam_area_on_target(self):
        return pi * self.calc_beam_radius_on_target() ** 2

    def calc_beam_intensity_on_target(self):
        axial_distance = -self.laser.focus_x / cos(self.laser.incidence_theta)
        return self.laser.calc_intensity(axial_distance)

    def get_output_timesteps(self, file_prefix: str):
        all_files = os.listdir(self.data_dir_path)
        matching_files = [
            f for f in all_files if f.startswith(file_prefix) and f.endswith(".sdf")
        ]
        num_frames = len(matching_files)
        return np.linspace(
            self.domain.time_interval[0], self.domain.time_interval[1], num_frames
        )
