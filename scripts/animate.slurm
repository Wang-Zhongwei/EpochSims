#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1 --ntasks-per-node=48

#SBATCH --job-name=animate
#SBATCH --account=PAS0035
#SBATCH --output=/users/PAS2137/wang15032/EpochSims/logs/animate_%j.out
#PBS -m abe

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "At least one simulation ID must be provided."
    exit 1
fi

# Run animate.py with all provided arguments
echo "Processing simulation IDs: $@"
python /users/PAS2137/wang15032/EpochSims/animate.py "$@"
