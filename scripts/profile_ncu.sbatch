#!/bin/bash
#SBATCH --account=bgbz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --job-name=ncu_profile
#SBATCH --output=results/ncu_%j.out
#SBATCH --error=results/ncu_%j.err

set -euo pipefail
module purge
module load cuda/12.8
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

make gpu

# Metrics to collect:
#   sm__throughput                  — SM utilization
#   l1tex__t_sector_hit_rate        — L1 cache hit rate
#   lts__t_sector_hit_rate          — L2 cache hit rate
#   sm__warps_active                — warp occupancy
#   dram__bytes.sum.per_second      — memory bandwidth
#   smsp__sass_average_data_bytes_per_sector_mem_global_op_ld — global load efficiency
METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__bytes.sum.per_second,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"

# Profile adaptive kernel on ogbn-arxiv (medium graph, representative)
echo "=== Profiling adaptive kernel on ogbn-arxiv ==="
ncu --metrics "$METRICS" \
    --kernel-name spmm_adaptive_kernel \
    --launch-skip 10 --launch-count 5 \
    -o results/ncu_arxiv_adaptive \
    ./spmm_gpu --graph data/ogbn_arxiv.edgelist --F 64 --runs 20 --warmup 10

# Profile naive kernel on ogbn-arxiv (for comparison)
echo "=== Profiling naive kernel on ogbn-arxiv ==="
ncu --metrics "$METRICS" \
    --kernel-name spmm_naive_kernel \
    --launch-skip 10 --launch-count 5 \
    -o results/ncu_arxiv_naive \
    ./spmm_gpu --graph data/ogbn_arxiv.edgelist --F 64 --runs 20 --warmup 10

# Profile adaptive kernel on ogbn-products (large graph, stresses high-degree path)
echo "=== Profiling adaptive kernel on ogbn-products ==="
ncu --metrics "$METRICS" \
    --kernel-name spmm_adaptive_kernel \
    --launch-skip 10 --launch-count 5 \
    -o results/ncu_products_adaptive \
    ./spmm_gpu --graph data/ogbn_products.edgelist --F 64 --runs 20 --warmup 10 --skip-cpu

echo ""
echo "NCU report files written to results/ncu_*.ncu-rep"
echo "Open with: ncu-ui results/ncu_arxiv_adaptive.ncu-rep"
echo ""
echo "--- Summary of collected metrics (from stdout above) ---"
