import os
import config

# Now we can access the variable from the config.py
experiment_name = config.experiment_name

# Prepare deck_name and job_name using experiment_name
try: 
    deck_file = f"{config.deck_name}.deck"
except AttributeError:
    deck_file = f"{experiment_name}.deck"

try: 
    job_file = f"{config.job_name}.job"
except AttributeError:
    job_file = f"{experiment_name}.job"


# Prepare the contents of the .job script
job_script_contents = f"""#!/bin/bash
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
output_dir_path={config.EPOCH_OUTPUT_BASE}/$experiment_name

mkdir -p $output_dir_path
echo -e $output_dir_path >$output_dir_path/USE_DATA_DIRECTORY
cp $epoch_base_path/Makefile $output_dir_path/Makefile
cp $input_dir_path/$job_file $output_dir_path/$job_file
cp $input_dir_path/$deck_file $output_dir_path/input.deck
cp $input_dir_path/ElecDens.dat $output_dir_path/ElecDens.dat
cp $input_dir_path/ProtDens.dat $output_dir_path/ProtDens.dat
cp $input_dir_path/CarbDens.dat $output_dir_path/CarbDens.dat
cd $output_dir_path

mpiexec -n $num_procs $epoch_bin_path $output_dir_path/$deck_file >$output_dir_path/log.txt
"""

# Write the contents to the .job file
with open(os.path.join(config.INPUT_BASE, f"{job_file}"), 'w') as f:
    f.write(job_script_contents)
