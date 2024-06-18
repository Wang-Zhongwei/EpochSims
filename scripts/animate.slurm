#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1 --ntasks-per-node=40

#SBATCH --job-name=animate
#SBATCH --account=PAS0035
#SBATCH --output=/users/PAS2137/wang15032/EpochSims/logs/animate_%j.out
#PBS -m abe

# parent_dir=/users/PAS2137/wang15032/EpochSims/analysis/2D_scaling_gaussian_preplasma

# find $parent_dir -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 -I {} basename {} | xargs python animate.py

simulation_id="${1:-2.6e21_preplasma_on_target_20240604}"
python animate.py $simulation_id