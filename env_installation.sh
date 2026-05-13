#!/bin/bash

# 1. Load necessary system modules
module purge
module load cuda/12.2 gcc/12 python/3.11

# 2. Setup paths
export VENV_PATH="$WORK/discrete-diffusion-guidance/discdiff"
export TMPDIR="$PWD/tmp_pip"
mkdir -p $TMPDIR

# 3. Create and activate venv
python3 -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# 4. Core dependencies
pip install --upgrade pip setuptools wheel
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install mkl==2023.2.0 causal-conv1d==1.4.0

# 5. Complex builds (Flash-Attn & Mamba)
# We do these separately because of the --no-build-isolation flag
pip install flash-attn==2.7.2.post1 --no-cache-dir --no-build-isolation
pip install mamba-ssm==1.2.0.post1 --no-build-isolation

# 6. Install the rest of the project requirements
if [ -f "requirements_leonardo.txt" ]; then
    pip install -r requirements_leonardo.txt
fi

echo "Environment setup complete!"
