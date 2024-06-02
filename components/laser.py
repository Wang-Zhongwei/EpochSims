
import numpy as np
from numpy import pi
from scipy.constants import elementary_charge, epsilon_0, m_e, speed_of_light


class GaussianBeam:
    def __init__(
        self,
        I_0: float,
        lambda_0: float,
        w_0: float,
        tau: float,
        incidence_angle: float = 0,
        focus_x: float = 0,
        focus_y: float = 0,
        dim: float = 3,
    ):
        """

        Args:
            I_0 (float): Beam intensity in W/cm^2
            lambda_0 (float): Beam wavelength in m
            w_0 (float): Beam waist in m
            tau (float): Pulse duration. Defaults to None.
            incidence_angle (float, optional): Incidence angle of the beam in degrees. Defaults to 0.
            focus_x (float, optional): Focus x-coordinate. Defaults to 0.
            focus_y (float, optional): Focus y-coordinate. Defaults to 0.
            dim (int, optional): Dimension. Defaults to 3.
        """
        self.I_0 = I_0
        self.lambda_0 = lambda_0
        self.w_0 = w_0
        self.tau = tau
        self.incidence_angle = incidence_angle
        self.incidence_theta = incidence_angle * pi / 180
        self.focus_x = focus_x
        self.focus_y = focus_y
        self.dim = dim

    @property
    def beam_energy(self):
        """
        Returns:
            float: Beam energy in Joule
        """
        if self.tau is None:
            raise ValueError("Pulse duration tau is required to calculate beam energy")
        I_0 = self.I_0 * 1e4
        return pi * self.w_0**2 * I_0 * self.tau

    @property
    def peak_electric_field(self):
        """
        Returns:
            float: Peak electric field at the waist of the laser beam in V/m
        """
        I_0 = self.I_0 * 1e4
        return np.sqrt(2 * I_0 / epsilon_0 / speed_of_light)

    @property
    def angular_frequency(self):
        """
        Returns:
            float: Angular Frequency in rad/s
        """
        lambda_0 = self.lambda_0
        return 2 * pi * speed_of_light / lambda_0

    @property
    def critical_density(self):
        """

        Returns:
            float: Critical plasma density for the laser beam in 1/m^3. The critical density is the
            density at which the plasma frequency is equal to the frequency of the laser.
            Above this density, the plasma becomes opaque to the laser light.
        """
        return self.angular_frequency**2 * m_e * epsilon_0 / elementary_charge**2

    @property
    def normalized_amplitude(self):
        return m_e * speed_of_light * self.angular_frequency / elementary_charge

    def calc_beam_radius(self, z: float):
        """

        Args:
            z (float): Axial distance from beam waist in m

        Returns:
            float: Beam radius at the given axial distance in m
        """
        x_R = pi * self.w_0**2 / self.lambda_0
        return self.w_0 * (1 + (z / x_R) ** 2) ** 0.5

    def calc_z(self, w: float):
        """

        Args:
            w (float): Beam radius in m

        Returns:
            float: Axial distance from beam waist in m
        """
        x_R = pi * self.w_0**2 / self.lambda_0
        return x_R * ((w / self.w_0) ** 2 - 1) ** 0.5

    def calc_intensity(self, z: float, r: float = 0):
        """

        Args:
            z (float): Axial distance from beam waist in m
            r (float): Radial distance from beam axis in m

        Returns:
            float: Beam intartensity at the given axial and radial distance in W/cm^2
        """
        w = self.calc_beam_radius(z)
        return self.I_0 * (self.w_0 / w) ** (self.dim - 1) * np.exp(-((r / w) ** 2))

    def __repr__(self):
        return f"GaussianBeam(I_0={self.I_0} W/cm^2, lambda_0 = {self.lambda_0:.2e} m, w_0 = {self.w_0:.2e} m"