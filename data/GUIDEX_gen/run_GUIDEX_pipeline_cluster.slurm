#!/bin/bash
#SBATCH --job-name=run_vllm_dataset
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00          # Adjust if needed
#SBATCH --mem=256GB                # Increase if you encounter memory issues
#SBATCH --gres=gpu:2
#SBATCH --output=GUIDEX_gen_out.%j.out
#SBATCH --error=GUIDEX_gen_err.%j.err


set -x  # Print commands before execution
set -e  # Exit on errors

echo "Starting SLURM job script"


# Print environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"


export TORCH_NCCL_BLOCKING_WAIT=1  # Makes NCCL operations blocking
export NCCL_TIMEOUT_SEC=300        # Increases the timeout (adjust as needed)
export NCCL_DEBUG=INFO

# NCCL settings for multi-GPU
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_NET_GDR_LEVEL=5

# Set HF cache directories relative to SLURM submit directory
export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/.cache/hub"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/.cache/datasets"

# Create cache directories if they don't exist
mkdir -p $HF_HUB_CACHE
mkdir -p $HF_DATASETS_CACHE

# Activate your conda environment
# Assumes conda is initialized (e.g. by running `conda init bash` once)
source ../../guidex_env/bin/activate
#conda activate guidex

echo "Environment activated"

# Verify CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"

# Run the pipeline –‑ paths are relative to this script's directory
srun python GUIDEX_pipeline.py \
      --input  fineweb-edu-1k.json \
      --output GuideX-1k.json \
      --subset 1000 \
      --batch-size 32
echo "SLURM job script completed"