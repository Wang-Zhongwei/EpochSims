from enum import Enum
import time

class Species(Enum):
    PROTON = 'Proton'
    ELECTRON = 'Electron'
    CARBON = 'Carbon'

class Plane(Enum):
    XY = 'xy'
    XZ = 'xz'
    YZ = 'yz'
    XYZ = 'xyz'

class Scalar(Enum):
    NUMBER_DENSITY = 'Derived_Number_Density'
    TEMPERATURE = 'Derived_Temperature'

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before function execution
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time after function execution
        print(f"Function {func.__name__} took {end_time - start_time:.3f} seconds to execute.")
        return result

    return wrapper