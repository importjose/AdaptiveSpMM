#!/usr/bin/env bash
# Download benchmark datasets to data/
# Run this on Delta (login node is fine — files are small except Reddit/ogbn-products).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# ── Helper: convert PyG/OGB edge_index CSVs to our edge-list format ───────────
# Usage: ogb_to_edgelist <raw_dir> <out_file>
ogb_to_edgelist() {
    python3 - "$1" "$2" <<'PYEOF'
import sys, numpy as np, os, pathlib

raw_dir, out_file = sys.argv[1], sys.argv[2]
# OGB stores edges as numpy arrays
edge_index = np.load(os.path.join(raw_dir, "edge.csv.gz"))  # fallback
# Try standard OGB format
try:
    import ogb.nodeproppred as ogbn
    pass
except ImportError:
    print("ogb not installed — install with: pip install ogb", file=sys.stderr)
    sys.exit(1)

PYEOF
}

echo "=== Downloading Cora ==="
# Cora comes bundled with many GNN frameworks. Use a simple mirror.
if [ ! -f cora.edgelist ]; then
    python3 - <<'PYEOF'
# Minimal Cora downloader using torch_geometric (if available) or direct download
try:
    from torch_geometric.datasets import Planetoid
    import torch
    dataset = Planetoid(root='/tmp/cora_pyg', name='Cora')
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    N = data.num_nodes
    srcs, dsts = edge_index[0], edge_index[1]
    M = len(srcs)
    with open('cora.edgelist', 'w') as f:
        f.write(f"# Cora citation network\n")
        f.write(f"{N} {M}\n")
        for s, d in zip(srcs, dsts):
            f.write(f"{s} {d} 1.0\n")
    print(f"Saved cora.edgelist: {N} nodes, {M} edges")
except ImportError:
    print("torch_geometric not available — download Cora manually.")
    print("Alternatively, run: pip install torch torch_geometric")
PYEOF
else
    echo "  cora.edgelist already exists, skipping."
fi

echo ""
echo "=== Downloading ogbn-arxiv ==="
if [ ! -f ogbn_arxiv.edgelist ]; then
    python3 - <<'PYEOF'
try:
    import ogb.utils.url as _u; _u.decide_download = lambda url: True
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-arxiv', root='/tmp/ogb_data')
    graph, _ = dataset[0]
    srcs = graph['edge_index'][0]
    dsts = graph['edge_index'][1]
    N = graph['num_nodes']
    M = len(srcs)
    with open('ogbn_arxiv.edgelist', 'w') as f:
        f.write(f"# ogbn-arxiv\n")
        f.write(f"{N} {M}\n")
        for s, d in zip(srcs, dsts):
            f.write(f"{s} {d} 1.0\n")
    print(f"Saved ogbn_arxiv.edgelist: {N} nodes, {M} edges")
except ImportError:
    print("ogb not installed — run: pip install ogb")
PYEOF
else
    echo "  ogbn_arxiv.edgelist already exists, skipping."
fi

echo ""
echo "=== Downloading ogbn-products (large ~2M nodes) ==="
if [ ! -f ogbn_products.edgelist ]; then
    python3 - <<'PYEOF'
try:
    import ogb.utils.url as _u; _u.decide_download = lambda url: True
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name='ogbn-products', root='/tmp/ogb_data')
    graph, _ = dataset[0]
    srcs = graph['edge_index'][0]
    dsts = graph['edge_index'][1]
    N = graph['num_nodes']
    M = len(srcs)
    with open('ogbn_products.edgelist', 'w') as f:
        f.write(f"# ogbn-products\n")
        f.write(f"{N} {M}\n")
        for s, d in zip(srcs, dsts):
            f.write(f"{s} {d} 1.0\n")
    print(f"Saved ogbn_products.edgelist: {N} nodes, {M} edges")
except ImportError:
    print("ogb not installed — run: pip install ogb")
PYEOF
else
    echo "  ogbn_products.edgelist already exists, skipping."
fi

echo ""
echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.edgelist 2>/dev/null || echo "  (none downloaded yet)"
