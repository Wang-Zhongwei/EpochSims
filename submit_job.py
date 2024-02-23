import os
import argparse 
from datetime import datetime
import subprocess
from configs.base_config import *

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str)
parser.add_argument('--deck-name', type=str)
parser.add_argument('--num-nodes', type=int, default=1)
parser.add_argument('--ntasks-per-node', type=int, default=1)
parser.add_argument('--time-limit', type=str, default="00:30:00")

args = parser.parse_args()

timestamp = datetime.now().strftime("%Y%m%d")
job_name = f"{timestamp}_{args.experiment_name}_{args.deck_name}"
ouput_dir_path = os.path.join(OUTPUT_BASE_PATH, job_name)
subprocess.run(f"conda activate {CONDA_ENV_NAME}", shell=True)
os.makedirs(ouput_dir_path, exist_ok=True)

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
    template = template.replace("${EXECUTABLE_PATH}", EXECUTABLE_PATH)

    # save template to output_dir
    with open(f"{ouput_dir_path}/{job_name}.sh", "w") as f:
        f.write(template)

# submit job
subprocess.run(f"sbatch {ouput_dir_path}/{job_name}.sh", shell=True)



