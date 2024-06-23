#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --mem=700GB

#SBATCH --job-name=process_3d_data
#SBATCH --account=PAS0035
#SBATCH --output=/users/PAS2137/wang15032/EpochSims/logs/process_3d_data_%j.out
#PBS -m abe

simulation_id="${1:-2.6e21_preplasma_on_target_20240604}"
python /users/PAS2137/wang15032/EpochSims/process_3d_data.py $simulation_id