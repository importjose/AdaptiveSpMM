#!/bin/bash
#SBATCH --account=bgbz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --job-name=AdaptiveSpMM_phase4
#SBATCH --output=results/phase4_%j.out
#SBATCH --error=results/phase4_%j.err

set -euo pipefail
module purge
module load cuda/12.8
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

make gpu
chmod +x spmm_gpu

RUNS=200
WARMUP=10
CSV="results/bench_results.csv"

# CSV header
echo "graph,N,nnz,F,thresh_low,thresh_med,cpu_ms,naive_ms,cusparse_ms,warp_ms,multiwarp_ms,adapt3_ms,adapt2_ms,preprocess_ms" > "$CSV"

echo "=== Full benchmark: real datasets ==="

# Cora (small, CPU baseline included)
for F in 32 64 128; do
    echo "--- cora.edgelist F=$F ---"
    ./spmm_gpu --graph data/cora.edgelist --F $F --runs $RUNS --warmup $WARMUP --csv >> "$CSV"
done

# ogbn-arxiv (medium, CPU baseline included)
for F in 32 64 128; do
    echo "--- ogbn_arxiv.edgelist F=$F ---"
    ./spmm_gpu --graph data/ogbn_arxiv.edgelist --F $F --runs $RUNS --warmup $WARMUP --csv >> "$CSV"
done

# ogbn-products (large, skip CPU baseline — too slow)
for F in 32 64 128; do
    echo "--- ogbn_products.edgelist F=$F ---"
    ./spmm_gpu --graph data/ogbn_products.edgelist --F $F --runs $RUNS --warmup $WARMUP --skip-cpu --csv >> "$CSV"
done

echo ""
echo "=== Full benchmark: synthetic power-law graphs ==="

for N in 10000 50000 200000; do
    for F in 64 128; do
        echo "--- synthetic N=$N F=$F ---"
        ./spmm_gpu --N $N --deg 20 --F $F --runs $RUNS --warmup $WARMUP --csv >> "$CSV"
    done
done

echo ""
echo "Results saved to $CSV"
cat "$CSV"
