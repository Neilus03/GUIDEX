#!/bin/bash
#SBATCH --job-name=run_vllm_dataset
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7-00:00:00          # Adjust if needed
#SBATCH --mem=256GB                # Increase if you encounter memory issues
#SBATCH --gres=gpu:2
#SBATCH --output=/sorgin1/users/neildlf/GoLLIE-dev/data/GUIDEX_gen/GUIDEX_gen_out.%j.out
#SBATCH --error=/sorgin1/users/neildlf/GoLLIE-dev/data/GUIDEX_gen/GUIDEX_gen_err.%j.err


set -x  # Print commands before execution
set -e  # Exit on errors

echo "Starting SLURM job script"

# Print environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Set HF cache directories to paths with sufficient space
export HF_HUB_CACHE=/sorgin1/users/neildlf/ncache/hub
export HF_DATASETS_CACHE=/sorgin1/users/neildlf/ncache/datasets

export TORCH_NCCL_BLOCKING_WAIT=1  # Makes NCCL operations blocking
export NCCL_TIMEOUT_SEC=300        # Increases the timeout (adjust as needed)
export NCCL_DEBUG=INFO

# NCCL settings for multi-GPU
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_NET_GDR_LEVEL=5

# Activate your virtual environment
source /sorgin1/users/neildlf/gollie/bin/activate

echo "Environment activated"

# Verify CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"

# Run the pipeline –‑ fill in the paths your script expects
srun python /sorgin1/users/neildlf/GoLLIE-dev/data/GUIDEX_gen/GUIDEX_pipeline.py \
      --input  /sorgin1/users/neildlf/GoLLIE-dev/data/GUIDEX_gen/fineweb-edu-100k.json \
      --output /sorgin1/users/neildlf/GoLLIE-dev/data/GUIDEX_gen/GUIDEX-100k.json \
      --subset 10000 \
      --batch-size 32
echo "SLURM job script completed"