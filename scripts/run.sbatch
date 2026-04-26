#!/bin/bash
#SBATCH --account=bgbz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --job-name=AdaptiveSpMM_phase1
#SBATCH --output=results/phase1_%j.out
#SBATCH --error=results/phase1_%j.err

set -euo pipefail

module purge
module load cuda/12.8

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

echo "=============================="
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=============================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Build GPU binary
make gpu

echo ""
echo "=== Synthetic powerlaw N=50000 F=64 ==="
./spmm_gpu --N 50000 --deg 20 --F 64 --runs 200 --warmup 10

echo ""
echo "=== Synthetic powerlaw N=50000 F=128 ==="
./spmm_gpu --N 50000 --deg 20 --F 128 --runs 200 --warmup 10

echo ""
echo "=== Synthetic uniform N=50000 F=64 ==="
./spmm_gpu --N 50000 --deg 20 --F 64 --uniform --runs 200 --warmup 10

# Uncomment once datasets are downloaded (see data/README.md):
# echo ""
# echo "=== Cora ==="
# ./spmm_gpu --graph data/cora.edgelist --F 64 --runs 200 --warmup 10

echo ""
echo "Done."
