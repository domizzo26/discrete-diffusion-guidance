#!/bin/bash
#SBATCH --job-name=mdlm_hybrid_cifar10
#SBATCH --account=IscrC_UNMASKED
#SBATCH --time=08:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=domitilla.izzo@studbocconi.it


# NOTE: Need to set the (local) dataset path for downloaded cifar-10 data
# For subset training, point to the preprocessed subset directory instead
PROJECT_ROOT="/leonardo_work/IscrC_UNMASKED/discrete-diffusion-guidance"
DATASET_PATH=${DATASET_PATH:-${PROJECT_ROOT}/data/cifar10}

<<comment
#  Usage:
cd scripts/
MODEL=<mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_cifar10_${MODEL} \
  train_cifar10_unet_guidance.sh
comment

# Setup environment
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit
#cd ../ || exit  # Go to the root directory of the repo
#REPO_ROOT=$(pwd)

# Convert DATASET_PATH to absolute path relative to repo root if not already absolute
if [[ "${DATASET_PATH}" != /* ]]; then
  DATASET_PATH="${REPO_ROOT}/${DATASET_PATH}"
fi

# Allow passing either dataset root or the cifar-10-batches-py subdirectory
if [[ "$(basename "${DATASET_PATH}")" == "cifar-10-batches-py" ]]; then
  DATASET_PATH="$(dirname "${DATASET_PATH}")"
fi

# Optional: reference images directory for f_mem
REFERENCE_DIR=${REFERENCE_DIR:-}
if [[ -n "${REFERENCE_DIR}" && "${REFERENCE_DIR}" != /* ]]; then
  REFERENCE_DIR="${REPO_ROOT}/${REFERENCE_DIR}"
fi

source $SLURM_SUBMIT_DIR/setup_leonardo.sh
mkdir -p /leonardo_work/IscrC_UNMASKED/.cache
export HF_HOME="/leonardo_work/IscrC_UNMASKED/.cache"
export TRANSFORMERS_CACHE="/leonardo_work/IscrC_UNMASKED/.cache/huggingface"
export NCCL_P2P_LEVEL=NVL
export HYDRA_FULL_ERROR=1
# Weights & Biases Config - Must be offline on compute nodes
export WANDB_MODE=offline

# Expecting:
#  - MODEL (mdlm, udlm)
if [ -z "${MODEL}" ]; then
  echo "MODEL is not set"
  exit 1
fi

T=0
if [ "${MODEL}" = "mdlm" ]; then
  PARAMETERIZATION=subs
  DIFFUSION="absorbing_state"
  ZERO_RECON_LOSS=False
  time_conditioning=False
  sampling_use_cache=True
elif [ "${MODEL}" = "udlm" ]; then
  PARAMETERIZATION=d3pm
  DIFFUSION="uniform"
  ZERO_RECON_LOSS=True
  time_conditioning=True
  sampling_use_cache=False
else
  echo "MODEL must be one of mdlm, udlm"
  exit 1
fi

# Optional: Set BATCH_SIZE for training (default: 250)
BATCH_SIZE=${BATCH_SIZE:-256}

# Optional: Set PROGRESSIVE_MASK_PROB for training (default: 0.0)
PROGRESSIVE_MASK_PROB=${PROGRESSIVE_MASK_PROB:-0.0}

# Optional: Set MAX_STEPS to train for more/fewer steps (default: 300000)
MAX_STEPS=${MAX_STEPS:-300000}

CHECKPOINT_EVERY_N_STEPS=${CHECKPOINT_EVERY_N_STEPS:-10000}

# Optional: Set VAL_CHECK_INTERVAL for validation frequency (default: 10000)
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-10000}

# Optional: f_mem computation during validation
COMPUTE_F_MEM=${COMPUTE_F_MEM:-false}
NUM_F_MEM_SAMPLES=${NUM_F_MEM_SAMPLES:-100}
MEM_THRESHOLD=${MEM_THRESHOLD:-0.333}

check_cifar10_dataset_path() {
  local dataset_root="$1"
  local batches_dir="${dataset_root}/cifar-10-batches-py"
  local required_files=(
    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    test_batch
    batches.meta
  )

  echo "Dataset root:     ${dataset_root}"
  echo "Expected batches: ${batches_dir}"

  if [[ ! -d "${batches_dir}" ]]; then
    echo "ERROR: Missing directory ${batches_dir}"
    echo "Hint: DATASET_PATH must point to the directory containing cifar-10-batches-py"
    return 1
  fi

  local missing_files=()
  for file_name in "${required_files[@]}"; do
    if [[ ! -f "${batches_dir}/${file_name}" ]]; then
      missing_files+=("${file_name}")
    fi
  done

  if (( ${#missing_files[@]} > 0 )); then
    echo "ERROR: Missing CIFAR-10 files in ${batches_dir}: ${missing_files[*]}"
    return 1
  fi

  return 0
}

echo "=============================================="
echo "Training Configuration"
echo "=============================================="
echo "MODEL:           ${MODEL}"
echo "Parameterization: ${PARAMETERIZATION}"
echo "Diffusion:       ${DIFFUSION}"
echo "Dataset path:    ${DATASET_PATH}"
echo "Batch size:      ${BATCH_SIZE}"
echo "Max steps:       ${MAX_STEPS}"
echo "Checkpoint every n steps: ${CHECKPOINT_EVERY_N_STEPS}"
echo "=============================================="

check_cifar10_dataset_path "${DATASET_PATH}" || exit 1

#cd /leonardo_work/IscrC_UNMASKED/discrete-diffusion-guidance/

# To enable preemption re-loading, set `hydra.run.dir`

cd "$PROJECT_ROOT" || { echo "Failed to change directory to $PROJECT_ROOT"; exit 1; }
echo "Current directory: $(pwd)"
echo "Using python from: $(which python)"
ls -F

srun python -u main.py --config-dir=configs \
  data=cifar10 \
  is_vision=True \
  diffusion=${DIFFUSION} \
  parameterization=${PARAMETERIZATION} \
  T=${T} \
  time_conditioning=${time_conditioning} \
  zero_recon_loss=${ZERO_RECON_LOSS} \
  data.train=${DATASET_PATH} \
  data.valid=${DATASET_PATH} \
  loader.global_batch_size=${BATCH_SIZE} \
  loader.eval_global_batch_size=64 \
  backbone=unet \
  model=unet \
  optim.lr=2e-4 \
  lr_scheduler=constant_warmup \
  lr_scheduler.num_warmup_steps=5000 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  +trainer.check_val_every_n_epoch=null \
  eval.compute_f_mem=${COMPUTE_F_MEM} \
  eval.reference_dir=${REFERENCE_DIR} \
  eval.num_f_mem_samples=${NUM_F_MEM_SAMPLES} \
  eval.mem_threshold=${MEM_THRESHOLD} \
  training.guidance.cond_dropout=0.1 \
  +training.progressive_mask_prob=${PROGRESSIVE_MASK_PROB} \
  eval.generate_samples=True \
  sampling.num_sample_batches=1 \
  sampling.batch_size=2 \
  sampling.use_cache=${sampling_use_cache} \
  sampling.steps=128 \
  wandb.name="cifar10_${RUN_NAME}" \
  hydra.run.dir="${PWD}/outputs/cifar10/${RUN_NAME}"