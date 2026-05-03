#!/bin/bash

# 1. Load Anaconda module 
module load anaconda3

# 2. Initialize conda for the current shell session
eval "$(conda shell.bash hook)"

# 3. Activate the 'discdiff' conda environment
conda activate discdiff

# Setup HF cache
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"