from enum import Enum
from math import cos, pi
import time

class Plane(Enum):
    XY = 'XY'
    XZ = 'XZ'
    YZ = 'YZ'
    XYZ = 'XYZ'
class Species(Enum):
    ELECTRON = 'Electron'
    PROTON = 'Proton'
    DEUTERON = 'Deuteron'
    CARBON = 'Carbon'
    ALL = ''

class Scalar(Enum):
    NUMBER_DENSITY = 'Derived_Number_Density'
    TEMPERATURE = 'Derived_Temperature'
    CHARGE_DENSITY = 'Derived_Charge_Density'
    AVG_PARTICLE_ENERGY = 'Derived_Average_Particle_Energy'

class Vector(Enum):
    Ex = 'Electric_Field_Ex'
    Ey = 'Electric_Field_Ey'
    Ez = 'Electric_Field_Ez'
    Bx = 'Magnetic_Field_Bx'
    By = 'Magnetic_Field_By'
    Bz = 'Magnetic_Field_Bz'
    Sx = 'Derived_Poynting_Flux_x'
    Sy = 'Derived_Poynting_Flux_y'
    Sz = 'Derived_Poynting_Flux_z'

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function execution
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time after function execution
        print(f"Function {func.__name__} took {end_time - start_time:.3f} seconds to execute.")
        return result

    return wrapper

def calc_beam_radius(w_0, z, lambda_0):
    """calculate beam radius at position z for a laser with wavelength lambda_0

    Args:
        w_0 (float): beam waist in um
        z (float): axial distance from beam waist in um
        lambda_0 (float): wavelength of laser in um
    Returns:
        float: beam radius w(z) at position z in um
    """

    x_R = pi * w_0**2 / lambda_0
    return w_0 * (1 + (z / x_R)**2)**0.5


def calc_beam_energy(I_0, w_0, tau):
    """calculate the energy of a Gaussian beam

    Args:
        I_0 (float): intensity of the beam in W/cm^2
        w_0 (float): spot size of the beam in um
        tau (float): duration of the pulse in fs

    Returns:
        float: energy of the beam in J
    """    
    w_0 = w_0 * 1e-6
    tau = tau * 1e-15
    I_0 = I_0 * 1e4
    
    return pi * w_0**2 * I_0 * tau


def calc_z(w_0, w, lambda_0):
    """Reverse function of calc_beam_waist

    Args:
        w_0 (float): beam waist in um
        w (float): beam radius w(z) at position z in um
        lambda_0 (float): wavelength of laser in um

    Returns:
        float: axial distance z from beam waist in um
    """    
    x_R = pi * w_0**2 / lambda_0
    return x_R * ((w / w_0)**2 - 1)**0.5