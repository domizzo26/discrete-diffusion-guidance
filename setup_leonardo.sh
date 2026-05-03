#!/bin/bash

# 1. Load the necessary system modules (no Anaconda)
module purge
module load cuda/12.4
module load gcc/12

# 2. Activate your virtual environment (the path where you created it)
# Make sure to use the absolute path to your venv
source /leonardo_work/IscrC_UNMASKED/discrete-diffusion-guidance/discdiff/bin/activate

# Setup HF cache
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"

# Set the temp directory for pip/builds
export TMPDIR=$PWD/tmp_pip