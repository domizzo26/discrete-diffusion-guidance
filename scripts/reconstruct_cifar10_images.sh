#!/bin/bash
#SBATCH --job-name=recon_cifar10
#SBATCH --account=IscrC_UNMASKED
#SBATCH --partition=boost_usr_prod
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=domitilla.izzo@studbocconi.it

# ============================================================================
# CIFAR-10 Image Reconstruction Script
# ============================================================================
# Reconstructs partially masked CIFAR-10 images using trained model(s).
#
# Usage:
# cd scripts/
# sbatch [sbatch_options] reconstruct_cifar10_images.sh ckpt1.ckpt [ckpt2.ckpt ...]
#
# Examples:
# # Reconstruct specific image with one checkpoint (positional)
# sbatch --export=ALL,INDEX=42 reconstruct_cifar10_images.sh outputs/cifar10/run1/checkpoints/last.ckpt
#
# # Multiple checkpoints
# sbatch --export=ALL,INDEX=100 reconstruct_cifar10_images.sh outputs/run1/checkpoints/last.ckpt outputs/run2/checkpoints/last.ckpt
#
# # Random image from category
# sbatch --export=ALL,CATEGORY=5 reconstruct_cifar10_images.sh outputs/cifar10/run1/checkpoints/last.ckpt
#
# # Custom masking
# sbatch --export=ALL,MASK_PERCENTAGE=30,MASK_FROM_TOP=true reconstruct_cifar10_images.sh outputs/run1/checkpoints/last.ckpt
#
# # Random blocks (non-overlapping scattered squares)
# sbatch --export=ALL,MASK_TYPE=random_blocks,MASK_PERCENTAGE=40 reconstruct_cifar10_images.sh outputs/run1/checkpoints/last.ckpt
#
# Optional environment variables:
# INDEX - Specific CIFAR-10 image index 0-49999 (required if CATEGORY/IMAGE_PATH not set)
# IMAGE_PATH - Path to custom image file (overrides index/category)
# IMAGE_LABEL - Class label for custom image (required for custom path with CFG)
# CATEGORY - CIFAR-10 category 0-9 (picks random image from class if INDEX not set)
# MASK_TYPE - Mask type: partial, random, random_blocks (default: random)
# MASK_PERCENTAGE - Percentage to mask, 0-100 (default: 50)
# MASK_FROM_TOP - Mask from top instead of bottom (default: false)
# OUTPUT_DIR - Output directory (default: auto-generated with timestamp)
# EPS - Noise schedule epsilon (default: 1e-5)
# SEED - Random seed (default: 42)
# SAMPLING_STEPS - Number of sampling steps (default: uses config)
# DATA_DIR - CIFAR-10 data directory (default: data/cifar10)
# ============================================================================

# Setup environment
PROJECT_ROOT="/leonardo_work/IscrC_UNMASKED/discrete-diffusion-guidance"
cd "${PROJECT_ROOT}" || { echo "ERROR: Could not cd to ${PROJECT_ROOT}"; exit 1; }
source "setup_leonardo.sh"

# Ensure project root is in PYTHONPATH and exit on error
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
set -e 

# Capture checkpoints from positional arguments
if [ $# -eq 0 ]; then
    echo "ERROR: No checkpoints provided as positional arguments."
    echo "Usage: sbatch [options] reconstruct_cifar10_images.sh ckpt1.ckpt [ckpt2.ckpt ...]"
    echo "Note: You can still use --export to set optional parameters like INDEX, MASK_TYPE, etc."
    exit 1
fi

CHECKPOINT_ARRAY=("$@")

# Set defaults
INDEX=${INDEX:-}
IMAGE_PATH=${IMAGE_PATH:-}
IMAGE_LABEL=${IMAGE_LABEL:-}
CATEGORY=${CATEGORY:-}
MASK_TYPE=${MASK_TYPE:-random}
MASK_PERCENTAGE=${MASK_PERCENTAGE:-50.0}
MASK_FROM_TOP=${MASK_FROM_TOP:-false}
OUTPUT_DIR=${OUTPUT_DIR:-}
SAMPLING_STEPS=${SAMPLING_STEPS:-}
DETERMINISTIC=${DETERMINISTIC:-false}
CFG_GAMMA=${CFG_GAMMA:-1.0}
DETERMINISTIC_THRESHOLD=${DETERMINISTIC_THRESHOLD:-}
NO_T_START_SCALING=${NO_T_START_SCALING:-false}
EPS=${EPS:-1e-5}
SEED=${SEED:-42}
DATA_DIR=${DATA_DIR:-/leonardo_work/IscrC_UNMASKED/discrete-diffusion-guidance/data/cifar10}

echo "=============================================="
echo "CIFAR-10 Image Reconstruction"
echo "=============================================="
echo "Checkpoints: ${#CHECKPOINT_ARRAY[@]} checkpoint(s)"
for ckpt in "${CHECKPOINT_ARRAY[@]}"; do
echo " - ${ckpt}"
done
echo "Index: ${INDEX:-auto (by category or random)}"
echo "Image Path: ${IMAGE_PATH:-none}"
echo "Image Label: ${IMAGE_LABEL:-none}"
echo "Category: ${CATEGORY:-auto (random)}"
echo "Mask type: ${MASK_TYPE}"
echo "Mask percentage: ${MASK_PERCENTAGE}%"
echo "Mask from top: ${MASK_FROM_TOP}"
echo "Output dir: ${OUTPUT_DIR:-auto (timestamped)}"
echo "Epsilon: ${EPS}"
echo "Sampling Steps: ${SAMPLING_STEPS:-from config}"
echo "Seed: ${SEED}"
echo "Data dir: ${DATA_DIR}"
echo "=============================================="

# Build command arguments
CMD_ARGS=(
    --checkpoints "${CHECKPOINT_ARRAY[@]}"
    --mask-type "${MASK_TYPE}"
    --mask-percentage "${MASK_PERCENTAGE}"
    --eps "${EPS}"
    --seed "${SEED}"
    --data-dir "${DATA_DIR}"
)

# Add optional arguments to the array
[ -n "${INDEX}" ] && CMD_ARGS+=(--index "${INDEX}")
[ -n "${CATEGORY}" ] && CMD_ARGS+=(--category "${CATEGORY}")
[ -n "${IMAGE_PATH}" ] && CMD_ARGS+=(--image-path "${IMAGE_PATH}")
[ -n "${IMAGE_LABEL}" ] && CMD_ARGS+=(--image-label "${IMAGE_LABEL}")
[ -n "${SAMPLING_STEPS}" ] && CMD_ARGS+=(--sampling-steps "${SAMPLING_STEPS}")
[ -n "${OUTPUT_DIR}" ] && CMD_ARGS+=(--output-dir "${OUTPUT_DIR}")
[ "${MASK_FROM_TOP}" = "true" ] && CMD_ARGS+=(--no-mask-from-bottom)
[ "${DETERMINISTIC}" = "true" ] && CMD_ARGS+=(--deterministic)
[ -n "${CFG_GAMMA}" ] && CMD_ARGS+=(--cfg-gamma "${CFG_GAMMA}")
[ -n "${DETERMINISTIC_THRESHOLD}" ] && CMD_ARGS+=(--deterministic-threshold "${DETERMINISTIC_THRESHOLD}")
[ "${NO_T_START_SCALING}" = "true" ] && CMD_ARGS+=(--no-t-start-scaling)

# Run reconstruction
echo ""
echo "Running reconstruction..."
srun python -u reconstruct_cifar10_images.py "${CMD_ARGS[@]}"

echo ""
echo "=============================================="
echo "Reconstruction complete!"
echo "Check the output directory outputs/cifar10/reconstructions/ for results:"
echo " - 00_original.png"
echo " - 01_masked.png"
echo " - 02_reconstructed_*.png (one per checkpoint)"
if [ -n "${OUTPUT_DIR}" ]; then
    echo "Results saved to: ${OUTPUT_DIR}"
else
    # Find the most recently created directory
    LATEST_DIR=$(ls -td outputs/cifar10/reconstructions/*/ 2>/dev/null | head -n 1)
    echo "Results saved to: ${LATEST_DIR:-outputs/cifar10/reconstructions/}"
    echo "Images: 00_original.png, 01_masked.png, 02_reconstructed_*.png"
fi
echo "=============================================="