#!/bin/bash
#SBATCH --account=bgbz-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --job-name=thresh_sweep
#SBATCH --output=results/sweep_%j.out
#SBATCH --error=results/sweep_%j.err

set -euo pipefail
module purge
module load cuda/12.8
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

make gpu
chmod +x spmm_gpu

F=64
RUNS=50
WARMUP=5
CSV="results/thresh_sweep.csv"

echo "graph,N,nnz,F,thresh_low,thresh_med,cpu_ms,naive_ms,cusparse_ms,adapt3_ms,adapt2_ms,preprocess_ms" > "$CSV"

for GRAPH in data/cora.edgelist data/ogbn_arxiv.edgelist data/ogbn_products.edgelist; do
    echo "=== Threshold sweep: $GRAPH ==="
    for TL in 4 8 16 32; do
        for TM in 32 64 128 256 512; do
            # thresh_low must be strictly less than thresh_med
            if [ "$TL" -ge "$TM" ]; then continue; fi
            echo "  thresh_low=$TL  thresh_med=$TM"
            ./spmm_gpu --graph "$GRAPH" --F $F --runs $RUNS --warmup $WARMUP \
                --thresh-low $TL --thresh-med $TM --skip-cpu --skip-uniform --csv >> "$CSV"
        done
    done
done

echo ""
echo "Threshold sweep saved to $CSV"
