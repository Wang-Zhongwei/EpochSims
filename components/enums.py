
from enum import Enum


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