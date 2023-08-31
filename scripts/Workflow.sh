#!/bin/bash
# Paths to your job scripts
simulation_script="/users/PAS2137/wang15032/EpochSims/scripts/2023-08-31_convergence_test.job"
saveSMovies_script="/users/PAS2137/wang15032/EpochSims/scripts/SaveSMovies.job"
saveFMovies_script="/users/PAS2137/wang15032/EpochSims/scripts/SaveFMovies.job"
savePMovies_script="/users/PAS2137/wang15032/EpochSims/scripts/SavePMovies.job"
plot_script="/users/PAS2137/wang15032/EpochSims/scripts/Plot.job"

# Submit the simulation job and capture the job id
simulation_job_id=$(sbatch --parsable "$simulation_script")

# Submit the analysis jobs with dependency on the simulation job
saveSMovies_job_id=$(sbatch --parsable --dependency=afterok:$simulation_job_id "$saveSMovies_script")
saveFMovies_job_id=$(sbatch --parsable --dependency=afterok:$simulation_job_id "$saveFMovies_script")
savePMovies_job_id=$(sbatch --parsable --dependency=afterok:$simulation_job_id "$savePMovies_script")

# Submit the plot job with dependency on the analysis jobs
plot_script_job_id=$(sbatch --parsable --dependency=afterok:$saveSMovies_job_id:$saveFMovies_job_id:$savePMovies_job_id "$plot_script")

echo "All jobs submitted"
echo "Simulation job ID: $simulation_job_id"
echo "SaveSMovies job ID: $saveSMovies_job_id"
echo "SaveFMovies job ID: $saveFMovies_job_id"
echo "SavePMovies job ID: $savePMovies_job_id"
echo "Plot job ID: $plot_script_job_id"
