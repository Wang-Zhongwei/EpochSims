if [ $# -ne 1 ]; then
    echo "Exactly 1 simulation id needs to be provided. Simulation ids are stored in configs/metadata.json. "
    exit 1
fi

process_slurm="process_3d_data.slurm"
animate_slurm="animate.slurm"

job_dir=$(dirname $0)

process_job_id=$(sbatch --parsable "$job_dir/$process_slurm" $1)
echo "Process job id: $process_job_id"

animate_job_id=$(sbatch --parsable --dependency=afterok:$process_job_id "$job_dir/$animate_slurm" $1)
echo "Animate job id: $animate_job_id"



