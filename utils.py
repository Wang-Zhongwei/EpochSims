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


def read_quantity_sdf_from_sdf(sdf: sdf.BlockList, quantity_name: str) -> sdf.BlockList:
    try:
        return sdf.__getattribute__(quantity_name)
    except AttributeError:
        raise ValueError(
            f"The specified quantity '{quantity_name}' does not exist in the {sdf}."
        )

def get_prefix(data_dir_path: str, quantity: Quantity):
    files = os.listdir(data_dir_path)
    prefixes = set(
        f.rsplit("_", maxsplit=1)[0]
        for f in files
        if f.endswith(".sdf") and not f.startswith("restart")
    )
    # return prefixes like temperature in Derived_Temperature
    for p in prefixes:
        if p in quantity.value.lower():
            data = sh.getdata(os.path.join(data_dir_path, f"{p}_0000.sdf"), verbose=False)
            if hasattr(data, quantity.value):
                return p
    
    # try default prefixes
    if quantity in (Quantity.Ex, Quantity.Ey, Quantity.Ez, Quantity.Bx, Quantity.By, Quantity.Bz):
        tentative_prefix = "fmovie"
    elif quantity in (Quantity.TEMPERATURE, Quantity.NUMBER_DENSITY, Quantity.CHARGE_DENSITY, Quantity.AVG_PARTICLE_ENERGY, Quantity.Px):
        tentative_prefix = "smovie"
    
    data = sh.getdata(os.path.join(data_dir_path, f"{tentative_prefix}_0000.sdf"), verbose=False)
    if hasattr(data, quantity.value):
        return tentative_prefix
    
    # brute force
    for p in prefixes:
        data = sh.getdata(os.path.join(data_dir_path, f"{p}_0000.sdf"), verbose=False)
        if hasattr(data, quantity.value):
            return p
    
    # raise exception if not found any prefix that has
    raise ValueError(f"Flie prefix not found for quantity {quantity.value} in {data_dir_path}")


def get_attribute_name(quantity: Quantity, species: Species = None):
    quantity_name = f"{quantity.value}"
    if species is not None:
        quantity_name += f"_{species.value}"
    return quantity_name

def get_plot_title(quantity: Quantity, species: Species = None, plane: Plane = None):
    plot_title = f"{quantity.value}".replace("_", " ")
    if species is not None:
        plot_title = f"{species.value} " + plot_title
    
    if plane is not None:
        plot_title += f" ({plane.value})"
        
    return plot_title