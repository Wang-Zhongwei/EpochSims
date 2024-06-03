from enum import Enum
import os
import sdf_helper as sh


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

    def get_attribute_name(self, species: Species = None):
        quantity_name = f"{self.value}"
        if species is not None:
            quantity_name += f"_{species.value}"
        return quantity_name

    def get_plot_title(self, species: Species = None, plane: Plane = None):
        plot_title = f"{self.value}".replace("_", " ")
        if species is not None:
            plot_title = f"{species.value} " + plot_title

        if plane is not None:
            plot_title += f" {plane.value} plane"

        return plot_title

    def get_prefix(self, data_dir_path: str):
        files = os.listdir(data_dir_path)
        prefixes = set(
            f.rsplit("_", maxsplit=1)[0]
            for f in files
            if f.endswith(".sdf") and not f.startswith("restart")
        )
        # return prefixes like temperature in Derived_Temperature
        for p in prefixes:
            if p in self.value.lower().replace("_", " "):
                data = sh.getdata(
                    os.path.join(data_dir_path, f"{p}_0000.sdf"), verbose=False
                )
                if hasattr(data, self.value):
                    return p

        # try default prefixes
        if self in (
            Quantity.Ex,
            Quantity.Ey,
            Quantity.Ez,
            Quantity.Bx,
            Quantity.By,
            Quantity.Bz,
        ):
            tentative_prefix = "fmovie"
        elif self in (
            Quantity.TEMPERATURE,
            Quantity.NUMBER_DENSITY,
            Quantity.CHARGE_DENSITY,
            Quantity.AVG_PARTICLE_ENERGY,
            Quantity.Px,
        ):
            tentative_prefix = "smovie"

        data = sh.getdata(
            os.path.join(data_dir_path, f"{tentative_prefix}_0000.sdf"), verbose=False
        )
        if hasattr(data, self.value):
            return tentative_prefix

        # brute force
        for p in prefixes:
            data = sh.getdata(
                os.path.join(data_dir_path, f"{p}_0000.sdf"), verbose=False
            )
            if hasattr(data, self.value):
                return p

        # raise exception if not found any prefix that has
        raise ValueError(
            f"Flie prefix not found for quantity {self.value} in {data_dir_path}"
        )

    def get_npy_file_name(self, species: Species, plane: Plane):
        quantity_name = self.get_attribute_name(species)
        return f"{quantity_name}_{plane.value}.npy"

    def get_media_file_name(self, species: Species = None, plane: Plane = None):
        return (
            self.get_plot_title(species, plane).lower().replace(" ", "_") + "_movie.mp4"
        )
