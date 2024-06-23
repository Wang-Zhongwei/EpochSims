#!/bin/bash
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NUM_NODES} --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT_NAME}
#SBATCH --output=${OUTPUT_DIR_PATH}/${JOB_NAME}.out
#SBATCH --error=${OUTPUT_DIR_PATH}/${JOB_NAME}.err
#PBS -m abe

module load mvapich2/2.3.6

cd ${OUTPUT_DIR_PATH}
echo -e $(pwd) > $(pwd)/USE_DATA_DIRECTORY

# adjust MPI cache size
export MV2_NDREG_ENTRIES_MAX=32768
export MV2_NDREG_ENTRIES=20480
# execute the simulation
mpiexec -n ${NUM_PROCS} ${EXECUTABLE_PATH} > ${OUTPUT_DIR_PATH}/epoch_log.txt