#!/bin/bash
#SBATCH --job-name=deepep_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -x
ulimit -c 0


# Set environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_LOCAL_SIZE=4
export WORLD_SIZE=$((SLURM_NNODES * WORLD_LOCAL_SIZE))  # Total GPUs across all
export RANK=$SLURM_PROCID
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID


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

# Libfabric HMEM (Heterogeneous Memory) settings - force GDRCopy usage
# export FI_HMEM_CUDA_USE_GDRCOPY=1
# export FI_MR_CACHE_MONITOR=userfaultfd

export NUM_TOKEN=16384 # 4096 * 4 # seq_len * batch_size

# Launch with srun only if NOT already inside an srun task
if [ -n "$SLURM_STEP_ID" ]; then
    # Already inside srun (bash inside srun)
    cd tests
    python test_intranode.py --num-processes 4 --hidden 2048 --num-experts 64 --num-topk 8
else
    # Not inside srun yet, need to spawn with srun
    srun --mpi=pmi2 --environment=deepep -u bash -lc '
set -x

# Set proper environment variables for multi-node PyTorch distributed
export RANK=$SLURM_PROCID
echo "Task $SLURM_PROCID (RANK=$RANK, WORLD_SIZE=$WORLD_SIZE) starting on $(hostname)"

cd tests

python test_intranode.py --num-processes 4 --hidden 2048 --num-experts 64 --num-topk 8 --num-tokens $NUM_TOKEN
'
fi
