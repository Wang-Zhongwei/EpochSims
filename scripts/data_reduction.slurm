#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1 --ntasks-per-node=48


#SBATCH --job-name=data_reduction
#SBATCH --account=PAS0035
#SBATCH --output=/users/PAS2137/wang15032/EpochSims/logs/data_reduction_%j.out
#PBS -m abe

# simulation_id="${1:-2.6e21_preplasma_on_target_20240604}"

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "At least one simulation ID must be provided."
    exit 1
fi

# Run animate.py with all provided arguments
echo "Processing simulation IDs: $@"
python /users/PAS2137/wang15032/EpochSims/data_reduction.py "$@"