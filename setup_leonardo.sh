#!/bin/bash

# 1. Carica il modulo conda (fondamentale su Leonardo)
module load anaconda3

# 2. Inizializza conda in modo sicuro
# Questo percorso è lo standard su Leonardo per inizializzare il comando 'conda'
source /leonardo/common/system/opt/anaconda/install/etc/profile.d/conda.sh

# 3. Attiva il tuo ambiente 'discdiff'
conda activate discdiff

# Setup HF cache
export HF_HOME="${PWD}/.hf_cache"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"