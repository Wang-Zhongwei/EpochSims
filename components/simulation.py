import logging
import os
import sys

import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm
from numpy import cos, pi, sqrt
from scipy.constants import elementary_charge, m_e, speed_of_light, Boltzmann as k_B


from configs.base_config import ANALYSIS_BASE_PATH, OUTPUT_BASE_PATH, REPO_PATH
from configs.metadata import get_simulation_metadata, load_metadata
from utils import get_weight_and_energy

from .domain import Domain
from .enums import Plane, Quantity, Species
from .laser import GaussianBeam
from .target import Target
from .data import Movie
from scipy.integrate import odeint

import sdf_helper as sh

logger = logging.getLogger(__name__)


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
            logger.error(f"Data directory {self.data_dir_path} does not exist")
            sys.exit(1)
        try:
            os.makedirs(self.analysis_dir_path, exist_ok=True)
        except PermissionError as e:
            logger.error(
                f"No permission for creating analysis directory {self.analysis_dir_path}"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(
                f"Error creating analysis directory {self.analysis_dir_path}: {e}"
            )
            sys.exit(1)

    def __repr__(self):
        return f"Simulation(simulation_id='{self.simulation_id}', laser={repr(self.laser)}, domain={repr(self.domain)}, target={repr(self.target)})"

    def __str__(self):
        return f"Simulation with id {self.simulation_id}"

    @classmethod
    def from_simulation_id(cls, simulation_id: str) -> "Simulation":
        metadata = load_metadata(os.path.join(REPO_PATH, "configs", "metadata.json"))
        simulation_metadata = get_simulation_metadata(metadata, simulation_id)
        if simulation_metadata is None:
            raise ValueError(
                f"Simulation with id {simulation_id} not found in metadata.json"
            )

        domain_metadata = simulation_metadata["domain"]
        domain = Domain(
            domain_metadata["boundaries"],
            domain_metadata["grid_size"],
            domain_metadata["time_interval"],
        )

        target_metadata = simulation_metadata["target"]
        target = Target(
            target_metadata["material"],
            target_metadata["electron_number_density"],
            target_metadata["thickness"],
            target_metadata["has_preplasma"],
        )

        laser_metadata = simulation_metadata["laser"]
        laser = GaussianBeam(
            laser_metadata["I_0"],
            laser_metadata["lambda_0"],
            laser_metadata["w_0"],
            laser_metadata["tau"],
            incidence_angle=laser_metadata["incidence_angle"],
            focus_x=laser_metadata["focus"]["x"],
            focus_y=laser_metadata["focus"]["y"],
            dim=len(domain.boundaries),
        )

        data_dir_path = os.path.join(
            OUTPUT_BASE_PATH, simulation_metadata["data_dir_rel_path"]
        )
        analysis_dir_path = os.path.join(
            ANALYSIS_BASE_PATH, simulation_metadata["data_dir_rel_path"]
        )
        return cls(
            simulation_id,
            data_dir_path,
            analysis_dir_path,
            laser=laser,
            domain=domain,
            target=target,
        )

    @property
    def dimension(self):
        return len(self.domain.boundaries)

    def get_plotting_parameters(self) -> dict:
        K_in_MeV = 11604518122
        normalized_peak_electric_field = (
            self.laser.peak_electric_field / self.laser.normalized_amplitude
        )
        normalized_momentum = (
            self.laser.peak_electric_field
            * elementary_charge
            / self.laser.angular_frequency
        )
        pondermotive_temperature = (
            m_e
            * speed_of_light**2
            * (
                sqrt(
                    1
                    + self.calc_beam_intensity_on_target()
                    * (self.laser.lambda_0 * 1e6) ** 2
                    / 1.37e18
                )
                - 1
            )
            / k_B
        )
        return {
            Quantity.Ex: {
                "norm": SymLogNorm(
                    vmin=-normalized_peak_electric_field,
                    vmax=normalized_peak_electric_field,
                    linthresh=1e-2,
                    linscale=1,
                ),
                "cmap": "bwr",
                "species": [None],
                "normalization_factor": self.laser.normalized_amplitude,
                "smoothing_sigma": 0.0,
                "cbar_label": r"$\frac{eE_x}{m_e c\omega}$",
            },
            Quantity.Ey: {
                "norm": SymLogNorm(
                    vmin=-normalized_peak_electric_field,
                    vmax=normalized_peak_electric_field,
                    linthresh=1e-2,
                    linscale=1,
                ),
                "cmap": "bwr",
                "species": [None],
                "normalization_factor": self.laser.normalized_amplitude,
                "smoothing_sigma": 0.0,
                "cbar_label": r"$\frac{eE_y}{m_e c\omega}$",
            },
            Quantity.Ez: {
                "norm": SymLogNorm(
                    vmin=-normalized_peak_electric_field,
                    vmax=normalized_peak_electric_field,
                    linthresh=1e-2,
                    linscale=1,
                ),
                "cmap": "bwr",
                "species": [None],
                "normalization_factor": self.laser.normalized_amplitude,
                "smoothing_sigma": 0.0,
                "cbar_label": r"$\frac{eE_z}{m_e c\omega}$",
            },
            Quantity.CHARGE_DENSITY: {
                "norm": SymLogNorm(linthresh=1e-2, linscale=1),
                "cmap": "bwr",
                "species": [None],
                "normalization_factor": self.laser.critical_density * elementary_charge,
                "smoothing_sigma": 0.00075 * min(self.domain.grid_size),
                "cbar_label": r"$\frac{\rho}{n_c e}$",
            },
            Quantity.NUMBER_DENSITY: {
                "norm": LogNorm(vmin=1e-2, vmax=1e2),
                "cmap": "viridis",
                "species": [Species.ELECTRON, Species.DEUTERON],
                "normalization_factor": self.laser.critical_density,
                "smoothing_sigma": 0.0,
                # "smoothing_sigma": 0.001 * min(self.domain.grid_size),
                "cbar_label": r"$\frac{n}{n_c}$",
            },
            Quantity.TEMPERATURE: {
                "norm": LogNorm(vmin=1e-2, vmax=1e1),
                "cmap": "plasma",
                "species": [Species.ELECTRON, Species.DEUTERON],
                "normalization_factor": K_in_MeV,
                "cbar_label": r"$T$ [MeV]",
                # "normalization_factor": pondermotive_temperature,
                # "cbar_label": r"$T/T_p$",
                "smoothing_sigma": 0.0,
            },
            Quantity.Px: {
                "norm": SymLogNorm(linthresh=1e-2, linscale=1, vmin=-1000, vmax=1000),
                "cmap": "bwr",
                "species": [Species.DEUTERON],
                "normalization_factor": normalized_momentum,
                "smoothing_sigma": 0.0,
                "cbar_label": r"$P_x\omega_0/eE$",
            },
        }

    def calc_beam_radius_on_target(self):
        axial_distance = -self.laser.focus_x / cos(self.laser.incidence_theta)
        return self.laser.calc_beam_radius(axial_distance)

    def calc_beam_area_on_target(self):
        return pi * self.calc_beam_radius_on_target() ** 2

    def calc_beam_intensity_on_target(self):
        axial_distance = -self.laser.focus_x / cos(self.laser.incidence_theta)
        return self.laser.calc_intensity(axial_distance)

    def get_num_frames(self, file_prefix = "pmovie"):
        all_files = os.listdir(self.data_dir_path)
        matching_files = [
            f for f in all_files if f.startswith(file_prefix) and f.endswith(".sdf")
        ]
        return len(matching_files)

    def get_output_timesteps(self, file_prefix: str):
        num_frames = self.get_num_frames(file_prefix)
        return np.linspace(
            self.domain.time_interval[0], self.domain.time_interval[1], num_frames
        )

    def load_data_from_npy(self, quantity: Quantity, species: Species, plane: Plane):
        npy_file_name = quantity.get_npy_file_name(species, plane)
        return np.load(
            os.path.join(self.analysis_dir_path, npy_file_name), allow_pickle=True
        )

    def load_movie(
        self, quantity: Quantity, species: Species = None, plane: Plane = None
    ) -> Movie:
        file_prefix = quantity.get_prefix(self.data_dir_path)
        if self.dimension == 2:
            data = []
            quantity_name = quantity.get_attribute_name(species)
            num_frames = self.get_num_frames(file_prefix)
            for i in range(num_frames):
                sdf = sh.getdata(
                    os.path.join(self.data_dir_path, f"{file_prefix}_{i:04d}.sdf"),
                    verbose=False,
                )
                d = getattr(sdf, quantity_name).data
                data.append(d)
            data = np.array(data)
        else:
            data = self.load_data_from_npy(quantity, species, plane)

        logger.info(f"Loaded movie data for {quantity.value}")
        return Movie(
            data, self.domain.get_extent(plane), self.get_output_timesteps(file_prefix)
        )

    def load_frame_data(
        self,
        frame_number: int,
        quantity: Quantity,
        species: Species = None,
        plane: Plane = None,
    ) -> np.ndarray:
        if self.dimension == 2:
            attr_name = quantity.get_attribute_name(species)
            file_prefix = quantity.get_prefix(self.data_dir_path)
            sdf = sh.getdata(
                os.path.join(
                    self.data_dir_path, f"{file_prefix}_{frame_number:04d}.sdf"
                ),
                verbose=False,
            )
            return getattr(sdf, attr_name).data
        else:
            data = self.load_data_from_npy(quantity, species, plane)
            return data[frame_number]

    def get_spectrum(
        self,
        species: Species,
        frame_number: int,
        num_bins=50,
        log_bins=False,
        file_prefix="pmovie",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get species energy spectrum at `frame_number`

        Args:
            frame_number (int): frame at which to get the spectrum
            species: Species: Species for which to get the spectrum
            frame_number (int): Frame number at which to get the spectrum
            num_bins (int, optional): Number of bins. Defaults to 25.
            log_bins (bool, optional): If True, use log-spaced bins. Defaults to False.
            file_prefix (str, optional): prefix of the file that contains deuteron TNSA data. Defaults to "pmovie".

        Returns:
            tuple[np.ndarray, np.ndarray]: E [MeV], dN/dE
        """
        if not hasattr(self, "spectrum_data"):
            self.spectrum_data = {}
        
        if species.value not in self.spectrum_data:
            self.spectrum_data[species.value] = {}
        
        species_spectrum_data = self.spectrum_data[species.value]
        if species_spectrum_data.get(frame_number) is None:
            pmovie = sh.getdata(
                os.path.join(
                    self.data_dir_path, f"{file_prefix}_{frame_number:04d}.sdf"
                ),
                verbose=False,
            )
            species_spectrum_data[frame_number] = get_weight_and_energy(pmovie, species)

        frame_data = species_spectrum_data[frame_number]
        data, weight = frame_data
        data = data / 1e6 / elementary_charge

        min, max = 1, 200
        if log_bins:
            bin_edges = np.logspace(np.log10(min), np.log10(max), num_bins + 1)
        else:
            bin_edges = np.linspace(min, max, num_bins + 1)

        counts, _ = np.histogram(
            data,
            bins=bin_edges,
            weights=weight,
        )

        E = (bin_edges[:-1] + bin_edges[1:]) / 2
        dE = np.diff(bin_edges)
        dNdE = counts / dE

        return E, dNdE

    def calc_neutron_yield(
        self,
        d: float,
        cross_section_func: callable,
        multiplicity_func: callable,
        frame_idx=-1,
    ):
        """Calculate neutron yield for a given thickness `d` (m) and cross section function

        Args:
            d (float): thickness of the target in meters
            cross_section_func (callable): cross_section_func(E) returns the cross section [mbar] at energy E [MeV]
            multiplicity_func (callable): multiplicity_func(E) returns the average multiplicity of neutrons at energy E [MeV]
            frame_idx (int, optional): select which frame to calculate the neutron yield. Defaults to -1, means the last frame.

        Returns:
            int: Estimated neutron yield
        """
        # Constants
        m_p = 1.673e-27  # proton mass in kg
        m_e = 0.511  # electron mass in MeV
        m_d = 1875.6  # deuteron mass in MeV

        # Deuteron
        z = 1

        # Beryllium properties
        Z = 4  # atomic number of beryllium
        A = 9.0122  # atomic mass of beryllium
        rho = 1.85  # density of beryllium in g/cm^3
        I = 63.7e-6  # mean excitation energy of beryllium in MeV

        # Bethe formula for deuterons (Z_projectile = 1) in beryllium
        def dEdx(E, x):
            gamma = E / m_d + 1
            beta = np.sqrt(1 - 1 / gamma**2)
            T_max = (
                2
                * m_e
                * beta**2
                * gamma**2
                / (1 + 2 * gamma * m_e / m_d + (m_e / m_d) ** 2)
            )

            return (
                -0.307
                * z**2
                * (Z / A)
                * rho
                / beta**2
                * (
                    1 / 2 * np.log(2 * m_e * beta**2 * gamma**2 * T_max / I**2)
                    - beta**2
                )
            )

        def solve_bethe(E0, x_range):
            return odeint(dEdx, E0, x_range)

        Y = 0
        n_Be = rho * 1e3 / (A * m_p)
        d *= 1e2  # convert from m to cm

        if frame_idx == -1:
            num_frames = self.get_num_frames("pmovie")
            frame_idx = num_frames - 1

        E, dNdE = self.get_spectrum(Species.DEUTERON, frame_idx, num_bins=200, log_bins=False)
        delta_E = E[1] - E[0]

        x_range = np.linspace(0, d, 100)
        delta_x = (x_range[1] - x_range[0]) * 1e-2  # convert from cm to m
        for E0, dNdE0 in zip(E, dNdE):
            E_of_x = solve_bethe(E0, x_range)
            Y += (
                np.sum(cross_section_func(E_of_x) * 1e-31 * multiplicity_func(E_of_x))
                * dNdE0
            )

        Y *= n_Be * delta_E * delta_x
        return round(Y)
