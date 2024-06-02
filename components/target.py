import numpy as np
from scipy.constants import elementary_charge, epsilon_0, m_e


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