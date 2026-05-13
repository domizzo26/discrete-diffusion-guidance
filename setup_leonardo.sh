#!/bin/bash
echo "Starting setup..."
module purge
module load cuda/12.2
module load gcc/12
module load python/3.11
echo "Modules loaded."

# Update this path to the REAL path you verified with 'ls'
source /leonardo_work/IscrC_UNMASKED/discrete-diffusion-guidance/discdiff/bin/activate
echo "Environment activated."

export HF_HOME="${PWD}/.hf_cache"
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"
export TMPDIR=$PWD/tmp_pip
echo "Setup complete."