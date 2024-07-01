if [ $# -lt 1 ]; then
    echo "At least 1 simulation id needs to be provided. Simulation ids are stored in configs/metadata.json."
    exit 1
fi

process_slurm="data_reduction.slurm"
animate_slurm="animate.slurm"

job_dir=$(dirname $0)

# Pass all arguments to the data_reduction.slurm script
process_job_id=$(sbatch --parsable "$job_dir/$process_slurm" "$@")
echo "Process job id: $process_job_id"

# Pass all arguments to the animate.slurm script, with dependency on the process job
animate_job_id=$(sbatch --parsable --dependency=afterok:$process_job_id "$job_dir/$animate_slurm" "$@")
echo "Animate job id: $animate_job_id"
