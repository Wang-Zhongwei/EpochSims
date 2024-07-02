import logging
import os
import time

import numpy as np
import sdf
import sdf_helper as sh
from components import Quantity, Species, Plane

logging.basicConfig(level=logging.INFO)


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


def get_weight_and_energy(pmovie: sdf.BlockList, species: Species) -> tuple[np.ndarray, np.ndarray]:
    
    if species == Species.DEUTERON:
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
        
    elif species == Species.ELECTRON:
        energy = pmovie.Particles_Ek_subset_ElectronPMovie_Electron
        weight = pmovie.Particles_Weight_subset_ElectronPMovie_Electron
    else:
        raise ValueError(f"Species '{species.value}' not recognized")

    return energy.data, weight.data


def read_quantity_sdf_from_sdf(sdf: sdf.BlockList, quantity_name: str) -> sdf.BlockList:
    try:
        return sdf.__getattribute__(quantity_name)
    except AttributeError:
        raise ValueError(
            f"The specified quantity '{quantity_name}' does not exist in the {sdf}."
        )
