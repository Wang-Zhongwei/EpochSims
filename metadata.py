# metadata.py
import json
import os

from configs.base_config import OUTPUT_BASE_PATH, ANALYSIS_BASE_PATH
from utils import Domain, GaussianBeam, Simulation, Target


def load_metadata(file_path="configs/metadata.json"):
    with open(file_path, "r") as file:
        metadata = json.load(file)
    return metadata


def get_simulation_metadata(metadata, simulation_id):
    for entry in metadata:
        if entry["id"] == simulation_id:
            return entry
    return None


def get_simulation(simulation_id):
    metadata = load_metadata()
    simulation_metadata = get_simulation_metadata(metadata, simulation_id)
    if simulation_metadata is None:
        raise ValueError(f"Simulation with id {simulation_id} not found in metadata")

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

    data_dir_path = os.path.join(OUTPUT_BASE_PATH, simulation_metadata["data_dir_rel_path"])
    analysis_dir_path = os.path.join(
        ANALYSIS_BASE_PATH, simulation_metadata["data_dir_rel_path"]
    )
    simulation = Simulation(
        data_dir_path, analysis_dir_path, laser=laser, domain=domain, target=target
    )
    return simulation


# test
if __name__ == "__main__":
    simulation = get_simulation("0.2_area_20240227")
    print(simulation.calc_beam_radius_on_target())
    print(simulation.laser.critical_density)
    print(simulation.target.plasma_frequency)
