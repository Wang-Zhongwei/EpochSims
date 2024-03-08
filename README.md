# EpochSims
EpochSims is a helper tool for job submission and data analysis on High Performance Computing (HPC) systems, specifically designed for the [epoch](https://github.com/Warwick-Plasma/epoch), particle-in-cell code plasma physics simulations.

# Folder Structure
- `configs`: This directory is used to define environment variables and experiment parameters.
- `decks`: This directory contains input decks. Each subdirectory within `decks` corresponds to a different experiment. Within these subdirectories, you will find decks with slight variations.

# Usage 
1. Modify the configuration in `configs`
2. Create a subdirectory `<experiment_name>` within `decks`
3. Write a deck file `<deck_name>.deck` under `decks/<experiment_name>` directory
4. Submit the job using `submit_job.py`
```bash
python submit_job.py -e <experiment_name> -d <deck_name> -n <num_nodes> -t <num_tasks_per_node> -l <time_limit>
```
5. Raw outputs are in `OUTPUT_BASE_PATH/experiment_name/deck_name`
6. Analyze output data in `analysis` directory.