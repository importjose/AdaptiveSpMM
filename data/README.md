# Data

Graph datasets are NOT committed to source control. Download them on Delta.

## Quick setup on Delta

```bash
module load python/3.10   # or whatever is available
pip install --user ogb torch torch_geometric
bash scripts/download_data.sh
```

## Expected files

| File                    | Nodes      | Edges         | Notes            |
|-------------------------|------------|---------------|------------------|
| `cora.edgelist`         | 2,708      | 10,556        | Citation network |
| `ogbn_arxiv.edgelist`   | 169,343    | 1,166,243     | Citation network |
| `ogbn_products.edgelist`| 2,449,029  | 61,859,140    | E-commerce graph |

Reddit (~114M edges) is optional; add it if storage/time allow.

## Edge-list format

```
# comment lines are ignored
N M              ← node count and edge count (optional)
src dst weight   ← one directed edge per line, 0-indexed, weight optional
```
