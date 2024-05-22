# EpochSims
EpochSims is a helper tool for job submission and data analysis on High Performance Computing (HPC) systems, specifically designed for the [epoch](https://github.com/Warwick-Plasma/epoch), particle-in-cell code plasma physics simulations.

# Folder Structure
- `configs`: used to define environment variables and experiment configurations.
- `decks`: contains input decks. Each subdirectory within `decks` corresponds to a different experiment. Within these subdirectories, you will find decks with slight variations.
- `analysis`: Contains animation and your jupyter notebook. It is not synced by git. 

# Usage 
## Initial configuration
Change environment variables in `configs/base_config.py`
## Submit running job
1. Add a new experiment configuration to `configs/metadata.json` 
2. Create a subdirectory `<experiment_name>` within `decks`
3. Write a deck file `<deck_name>.deck` under `decks/<experiment_name>` directory
4. Submit the job using `submit_job.py`
    ```bash
    python submit_job.py -e <experiment_name> -d <deck_name> -n <num_nodes> -t <num_tasks_per_node> -l <time_limit>
    ```
## Run post processing job
### 2D simulation
1. Default quantities and plotting parameters associated with them quantities are defined in `animate_2d.py` and `configs/metadata.py`. Change them if you want
2. Run animation jobs using `animate_2d.py`
    ```bash
    python animate_2d.py <sim_id_1> <sim_id_2> ... 
    ```
    It is recommended to encapsulate it with a slurm script and submit using sbatch
    ```bash
    #!/bin/bash
    #SBATCH --time=01:00:00
    #SBATCH --nodes=1 --ntasks-per-node=40
    #SBATCH --job-name=animate_2d
    #SBATCH --account=PAS0035
    #SBATCH --output=/users/PAS2137/wang15032/EpochSims/temp/animate_2d_%j.out
    #PBS -m abe


    python animate_2d.py 1e21_on_target_w_preplasma_20240521 1e21_off_target_w_preplasma_20240521 5e21_on_target_w_preplasma_20240521 5e21_off_target_w_preplasma_20240521
    ```
    Use sbatch to submit it 
    ```bash
    sbatch <path_to_your_job>
    ```