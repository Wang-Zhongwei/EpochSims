#!/bin/bash
# Paths to your job scripts
simulation_script="/users/PAS2137/wang15032/EpochSims/scripts/2023-08-03_3D_8CB_800nm_1e21_22deg.job"
saveSMovies_script="/users/PAS2137/wang15032/EpochSims/scripts/SaveSMovies.job"
saveFMovies_script="/users/PAS2137/wang15032/EpochSims/scripts/SaveFMovies.job"
savePMovies_script="/users/PAS2137/wang15032/EpochSims/scripts/SavePMovies.job"

# Submit the simulation job and capture the job id
simulation_job_id=$(sbatch --parsable "$simulation_script")

# Submit the analysis jobs with dependency on the simulation job
sbatch --dependency=afterok:$simulation_job_id "$saveSMovies_script"
sbatch --dependency=afterok:$simulation_job_id "$saveFMovies_script"
sbatch --dependency=afterok:$simulation_job_id "$savePMovies_script"
    