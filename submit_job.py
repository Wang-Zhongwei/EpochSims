import os
import argparse 
from datetime import datetime
import subprocess
from configs.base_config import *

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment-name', type=str)
parser.add_argument('-d', '--dimension', type=int, default=2)
parser.add_argument('-dn', '--deck-name', type=str)
parser.add_argument('-n', '--num-nodes', type=int, default=1)
parser.add_argument('-t', '--ntasks-per-node', type=int, default=1)
parser.add_argument('-l', '--time-limit', type=str, default="00:30:00")
args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d")
job_name = f"{args.deck_name}_{timestamp}"

ouput_dir_path = os.path.join(OUTPUT_BASE_PATH, args.experiment_name, job_name)
analysis_dir_path = os.path.join(ANALYSIS_BASE_PATH, args.experiment_name, job_name)

os.makedirs(ouput_dir_path, exist_ok=True)
os.makedirs(analysis_dir_path, exist_ok=True)

# load template.sh
with open("template.sh", "r") as f:
    template = f.read()
    # replace environment variables
    template = template.replace("${ACCOUNT_NAME}", ACCOUNT_NAME)
    template = template.replace("${NUM_NODES}", str(args.num_nodes))
    template = template.replace("${NTASKS_PER_NODE}", str(args.ntasks_per_node))
    template = template.replace("${NUM_PROCS}", str(args.num_nodes * args.ntasks_per_node))
    template = template.replace("${TIME_LIMIT}", args.time_limit)
    template = template.replace("${EXP_NAME}", args.experiment_name)
    template = template.replace("${DECK_NAME}", args.deck_name)
    template = template.replace("${JOB_NAME}", job_name)
    template = template.replace("${OUTPUT_DIR_PATH}", ouput_dir_path)
    if args.dimension == 3:
        template = template.replace("${EXECUTABLE_PATH}", EXECUTABLE_PATH_3D)
    else:
        template = template.replace("${EXECUTABLE_PATH}", EXECUTABLE_PATH_2D)

    # save template to output_dir
    with open(f"{ouput_dir_path}/{job_name}.sh", "w") as f:
        f.write(template)

# submit job
subprocess.run(f"sbatch {ouput_dir_path}/{job_name}.sh", shell=True)



