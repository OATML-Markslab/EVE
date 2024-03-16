#!/bin/bash
set -e # fail fully on first line failure (from Joost slurm_for_ml)

# Send python outputs (like print) directly to terminal/log without buffering
export PYTHONUNBUFFERED=1

export MSA_data_folder='./data/MSA'
export MSA_list='./data/mappings/example_mapping.csv'
export MSA_weights_location='./data/weights'
export protein_index=0
export num_cpus=1
export calc_method='evcouplings'

python3 calc_weights.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index "${protein_index}" \
    --MSA_weights_location "${MSA_weights_location}" \
    --num_cpus "$num_cpus" \
    --calc_method ${calc_method}
#    --skip_existing