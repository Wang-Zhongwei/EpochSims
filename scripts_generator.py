import os
import configs.config as config

experiment_name = config.experiment_name
try:
    deck_file = f"{config.deck_name}.deck"
except AttributeError:
    deck_file = f"{experiment_name}.deck"

try:
    job_file = f"{config.job_name}.job"
except AttributeError:
    job_file = f"{experiment_name}.job"


def generate_simulation_script():
    # Prepare deck_name and job_name using experiment_name

    # Prepare the contents of the .job script
    script_contents = f"""#!/bin/bash
#SBATCH --time={config.hours}:{config.minutes}:{config.seconds}
#SBATCH --nodes={config.nodes} --ntasks-per-node={config.ntasks_per_node}
#SBATCH --job-name={experiment_name}
#SBATCH --account={config.ACCOUNT_NUMBER}
#PBS -m abe

experiment_name={experiment_name}
deck_file={deck_file}
job_file={job_file}
num_procs={config.nodes * config.ntasks_per_node}

epoch_base_path={config.EPOCH_BASE}
epoch_bin_path=$epoch_base_path/bin/epoch3d
input_dir_path={config.INPUT_BASE}
script_dir_path={config.SCRIPT_BASE}
output_dir_path={config.EPOCH_OUTPUT_BASE}/$experiment_name

mkdir -p $output_dir_path
echo -e $output_dir_path >$output_dir_path/USE_DATA_DIRECTORY
cp $epoch_base_path/Makefile $output_dir_path/Makefile
cp $script_dir_path/$job_file $output_dir_path/$job_file
cp $input_dir_path/$deck_file $output_dir_path/input.deck
cp $input_dir_path/ElecDens.dat $output_dir_path/ElecDens.dat
cp $input_dir_path/ProtDens.dat $output_dir_path/ProtDens.dat
cp $input_dir_path/CarbDens.dat $output_dir_path/CarbDens.dat
cd $output_dir_path

mpiexec -n $num_procs $epoch_bin_path $output_dir_path/$deck_file >$output_dir_path/log.txt
    """

    # Write the contents to the .job file
    with open(os.path.join(config.SCRIPT_BASE, f"{job_file}"), "w") as f:
        f.write(script_contents)


def generate_analysis_script(movie_type):
    assert movie_type in [
        "S",
        "F",
        "P",
    ], "movie_type must be one of S(calar), F(ield), or P(article)"
    script_contents = f"""#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=32
#SBATCH --job-name=Save{movie_type}Movies
#SBATCH --account={config.ACCOUNT_NUMBER}
#PBS -m abe
conda activate general
python {config.PROJECT_BASE}/Save{movie_type}Movies.py > {config.LOG_BASE}/Save{movie_type}Movies.out
    """
    file_name = f"Save{movie_type}Movies.job"
    with open(os.path.join(config.SCRIPT_BASE, f"{file_name}"), "w") as f:
        f.write(script_contents)


def generate_write_density_script():
    script_contents = f"""#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=32
#SBATCH --job-name=WriteDensity
#SBATCH --account={config.ACCOUNT_NUMBER}
#PBS -m abe
conda activate general
python {config.PROJECT_BASE}/WriteDensity.py > {config.LOG_BASE}/WriteDensity.out
    """
    file_name = "WriteDensity3D.job"
    with open(os.path.join(config.SCRIPT_BASE, f"{file_name}"), "w") as f:
        f.write(script_contents)


def generate_workflow_script():
    script_contents = f"""#!/bin/bash
# Paths to your job scripts
simulation_script="{config.SCRIPT_BASE}/{job_file}"
saveSMovies_script="{config.SCRIPT_BASE}/SaveSMovies.job"
saveFMovies_script="{config.SCRIPT_BASE}/SaveFMovies.job"
savePMoview_script="{config.SCRIPT_BASE}/SavePMovies.job"

# Submit the simulation job and capture the job id
simulation_job_id=$(sbatch --parsable "$simulation_script")

# Submit the analysis jobs with dependency on the simulation job
sbatch --dependency=afterok:$simulation_job_id "$saveSMovies_script"
sbatch --dependency=afterok:$simulation_job_id "$saveFMovies_script"
sbatch --dependency=afterok:$simulation_job_id "$savePMovies_script"
    """
    file_name = "Workflow.sh"
    with open(os.path.join(config.SCRIPT_BASE, f"{file_name}"), "w") as f:
        f.write(script_contents)


if __name__ == "__main__":
    generate_simulation_script()
    generate_analysis_script("S")
    generate_analysis_script("F")
    generate_analysis_script("P")
    generate_write_density_script()
    generate_workflow_script()
