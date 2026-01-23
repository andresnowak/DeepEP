#!/bin/bash
#SBATCH --job-name=deepep_tune_sms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -x
ulimit -c 0


# Set environment variables
export PYTHONUNBUFFERED=1
export NVSHMEM_DEBUG=TRACE
export FI_LOG_LEVEL=debug

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# Enable GDR (GPUDirect RDMA)
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128

export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_SYMMETRIC_SIZE=4G
export NVSHMEM_REMOTE_TRANSPORT=libfabric

export EP_SIZE=4
export NUM_TOKENS=16384 # 4096 * 4 # (seq_len * batch_size)


echo "SLURM Configuration:"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  SLURM_NNODES: $SLURM_NNODES"
echo "  SLURM_NTASKS: $SLURM_NTASKS"
echo "  SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "  SLURM_NODELIST: $SLURM_NODELIST"

# Launch with srun only if NOT already inside an srun task
if [ -n "$SLURM_STEP_ID" ]; then
    # Already inside srun (bash inside srun)
    cd benchmark
    torchrun --nproc_per_node=${EP_SIZE} \
        tune_sms.py \
        --num-tokens $NUM_TOKENS \
        --hidden 2048 \
        --num-experts 64 \
        --num-topk 8 \
        --output intranode_sms_tuning_results.json
else
    # Not inside srun yet, need to spawn with srun
srun --environment=deepep -u bash -lc '
set -x
cd benchmark

torchrun --nproc_per_node=${EP_SIZE} \
    tune_sms.py \
    --num-tokens $NUM_TOKENS \
    --hidden 2048 \
    --num-experts 64 \
    --num-topk 8 \
    --output intranode_sms_tuning_results.json
'
fi
