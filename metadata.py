# metadata.py
import json
import os
from typing import Union

from matplotlib.colors import LogNorm, SymLogNorm
from scipy.constants import elementary_charge

from configs.base_config import ANALYSIS_BASE_PATH, OUTPUT_BASE_PATH, REPO_PATH
from utils import Domain, GaussianBeam, Quantity, Simulation, Species, Target


def load_metadata(file_path: str) -> dict:
    with open(file_path, "r") as file:
        metadata = json.load(file)
    return metadata


def get_simulation_metadata(metadata: dict, simulation_id: str) -> Union[None, dict]:
    for entry in metadata:
        if entry["id"] == simulation_id:
            return entry
    return None


def get_simulation(simulation_id: str) -> Simulation:
    metadata = load_metadata(os.path.join(REPO_PATH, "configs", "metadata.json"))
    simulation_metadata = get_simulation_metadata(metadata, simulation_id)
    if simulation_metadata is None:
        raise ValueError(f"Simulation with id {simulation_id} not found in metadata.json")

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
    simulation = Simulation(
        data_dir_path, analysis_dir_path, laser=laser, domain=domain, target=target
    )
    return simulation


def get_plotting_parameters(simulation: Simulation) -> dict:
    # todo: modify norm based on simulation configs
    K_in_MeV = 11604518122
    return {
        Quantity.Ez: {
            "norm": SymLogNorm(linthresh=1e-2, linscale=1),
            "cmap": "bwr",
            "species": [None],
            "normalization_factor": simulation.laser.normalized_amplitude,
            "smoothing_sigma": 0.0,
            "cbar_label": r"$\frac{eE_x}{m_e c\omega}$",
            "file_prefix": "fmovie",
        },
        Quantity.Ey: {
            "norm": SymLogNorm(linthresh=1e-2, linscale=1),
            "cmap": "bwr",
            "species": [None],
            "normalization_factor": simulation.laser.normalized_amplitude,
            "smoothing_sigma": 0.0,
            "cbar_label": r"$\frac{eE_y}{m_e c\omega}$",
            "file_prefix": "fmovie",
        },
        Quantity.Ez: {
            "norm": SymLogNorm(linthresh=1e-2, linscale=1),
            "cmap": "bwr",
            "species": [None],
            "normalization_factor": simulation.laser.normalized_amplitude,
            "smoothing_sigma": 0.0,
            "cbar_label": r"$\frac{eE_z}{m_e c\omega}$",
            "file_prefix": "fmovie",
        },
        Quantity.CHARGE_DENSITY: {
            "norm": SymLogNorm(linthresh=1e-2, linscale=1),
            "cmap": "bwr",
            "species": [None],
            "normalization_factor": simulation.laser.critical_density
            * elementary_charge,
            "smoothing_sigma": 0.00075 * min(simulation.domain.grid_size),
            "cbar_label": r"$\frac{\rho}{n_c e}$",
            "file_prefix": "smovie",
        },
        Quantity.NUMBER_DENSITY: {
            "norm": LogNorm(vmin=1e-5, vmax=2e1),
            "cmap": "viridis",
            "species": [Species.ELECTRON, Species.DEUTERON, Species.HYDROGEN],
            "normalization_factor": simulation.laser.critical_density,
            "smoothing_sigma": 0.0,
            "cbar_label": r"$\frac{n}{n_c}$",
            "file_prefix": "smovie",
        },
        Quantity.TEMPERATURE: {
            "norm": LogNorm(vmin=1e-5, vmax=2e1),
            "cmap": "plasma",
            "species": [Species.ELECTRON, Species.DEUTERON, Species.HYDROGEN],
            "normalization_factor": K_in_MeV,
            "smoothing_sigma": 0.0,
            "cbar_label": r"$T$ [MeV]",
            "file_prefix": "smovie",
        },
    }


# test
if __name__ == "__main__":
    simulation = get_simulation("0.2_area_20240227")
    print(simulation.calc_beam_radius_on_target())
    print(simulation.laser.critical_density)
    print(simulation.target.plasma_frequency)