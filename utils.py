from enum import Enum
import time

class Species(Enum):
    PROTON = 'Proton'
    DEUTERON = 'Deuteron'
    HYDROGEN = 'Hydrogen'
    ELECTRON = 'Electron'
    CARBON = 'Carbon'

class Plane(Enum):
    XY = 'XY'
    XZ = 'XZ'
    YZ = 'YZ'
    XYZ = 'XYZ'

class Scalar(Enum):
    NUMBER_DENSITY = 'Derived_Number_Density'
    TEMPERATURE = 'Derived_Temperature'

class Vector(Enum):
    Ex = 'Electric_Field_Ex'
    Ey = 'Electric_Field_Ey'
    Ez = 'Electric_Field_Ez'
    Bx = 'Magnetic_Field_Bx'
    By = 'Magnetic_Field_By'
    Bz = 'Magnetic_Field_Bz'

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function execution
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time after function execution
        print(f"Function {func.__name__} took {end_time - start_time:.3f} seconds to execute.")
        return result

    return wrapper