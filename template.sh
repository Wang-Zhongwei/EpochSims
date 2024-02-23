#!/bin/bash
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NUM_NODES} --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --account=${ACCOUNT_NAME}
#SBATCH --output=${OUTPUT_DIR_PATH}/${JOB_NAME}.out
#SBATCH --error=${OUTPUT_DIR_PATH}/${JOB_NAME}.err
#PBS -m abe

echo -e ${OUTPUT_DIR_PATH} > ${OUTPUT_DIR_PATH}/USE_DATA_DIRECTORY
# save the input deck to the output directory
if [ ! -f "decks/${EXP_NAME}/${DECK_NAME}.deck" ]; then
    echo "${DECK_NAME}.deck file not found in folder decks/${EXP_NAME}"
    exit 1
fi
cp decks/${EXP_NAME}/${DECK_NAME}.deck ${OUTPUT_DIR_PATH}/input.deck

# execute the simulation
cd ${OUTPUT_DIR_PATH}
mpiexec -n ${NUM_PROCS} ${EXECUTABLE_PATH} > ${OUTPUT_DIR_PATH}/epoch_log.txt