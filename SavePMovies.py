import os
import numpy as np
import sdf_helper as sh
import pandas as pd
from utils import *

def compute_energy(vx, vy, vz, species):
     # Get particle mass.
    if species == Species.PROTON:
        m = 938.28
    elif species == Species.ELECTRON:
        m = 0.511
    elif species == Species.CARBON:
        m = 11177
    else:
        raise ValueError(f"Unknown species {species}")

    c = 299792458
    v_mag_squared = vx**2 + vy**2 + vz**2
    gamma = 1/np.sqrt(1 - v_mag_squared/c**2)
    energy = m*(gamma - 1)

    return energy

@timer
def save_species_to_csv(inp_path, species, out_path, E_cutoff_MeV=1, decimation_ratio=0.1):
    all_data = sh.getdata(inp_path, verbose=False)

    # Particle IDs might *not* be written
    try:
        id = all_data.__getattribute__(f'Particles_ID_subset_{species.value}PMovie_{species.value}').data
    except AttributeError:
        id = None

    # These are always written
    weights = all_data.__getattribute__(f'Particles_Weight_subset_{species.value}PMovie_{species.value}').data
    x, y, z = all_data.__getattribute__(f'Grid_Particles_subset_{species.value}PMovie_{species.value}').data
    vx = all_data.__getattribute__(f'Particles_Vx_subset_{species.value}PMovie_{species.value}').data
    vy = all_data.__getattribute__(f'Particles_Vy_subset_{species.value}PMovie_{species.value}').data
    vz = all_data.__getattribute__(f'Particles_Vz_subset_{species.value}PMovie_{species.value}').data

    # decimation
    survival_indices = np.random.rand(len(id)) < decimation_ratio
    id = id[survival_indices]
    weights = weights[survival_indices]
    x = x[survival_indices]
    y = y[survival_indices]
    z = z[survival_indices]
    vx = vx[survival_indices]
    vy = vy[survival_indices]
    vz = vz[survival_indices]
    

    # Compute energy
    energy = compute_energy(vx, vy, vz, species)

    # Filter particles
    indices = (energy >= E_cutoff_MeV) & (vx > 0)

    data = pd.DataFrame({
        'id': id[indices],
        'weights': weights[indices],
        'x': x[indices],
        'y': y[indices],
        'z': z[indices],
        'vx': vx[indices],
        'vy': vy[indices],
        'vz': vz[indices],
        'energy': energy[indices]
    })

    data.to_csv(out_path, index=False)
    

def combine_frames_to_leaving_particles(inp_folder, x_cutoff, out_path=None):
    """
    Combine all the frames in inp_folder into a single csv file, containing only the particles that leave the box.
    """
    if out_path is None:
        out_path = os.path.join(inp_folder, 'leaving_particles.csv')

    # list all files in the inp_folder with .csv extension
    csv_files = [f for f in os.listdir(inp_folder) if f.endswith('.csv')]
    csv_files.sort()

    data = pd.DataFrame(columns=['id', 'weights', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'energy', 'leaving_frame'])
    for file in csv_files:
        frame = pd.read_csv(os.path.join(inp_folder, file))
        frame = frame[frame['x'] > x_cutoff]
        # if id is absent add it to data 
        new_particles = frame[~frame['id'].isin(data['id'])]
        new_particles['leaving_frame'] = int(file.split('_')[1].split('.')[0])
        data = pd.concat([data, new_particles], ignore_index=True)

    data.to_csv(out_path, index=False)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    inp_folder = '/fs/scratch/PAS2137/wang15032/EpochOutput/20220511_3D_Exp800nm8CB_1e21_22deg'
    out_folder = 'data/3d'

    # for i in range(0, 321, 10):
    #     inp_path = os.path.join(inp_folder, f'pmovie_{i:04}.sdf')
    #     out_path = os.path.join(out_folder, f'TNSAProton_{i:04}.csv')
    #     save_species_to_csv(inp_path, Species.PROTON, out_path, E_cutoff_MeV=1, decimation_ratio=1)
    #     print(f'Saved {out_path}')

    combine_frames_to_leaving_particles(out_folder, 5e-6)
