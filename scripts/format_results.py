#!/usr/bin/env python3
"""
Format AdaptiveSpMM benchmark results into tables.

Usage:
    python3 scripts/format_results.py results/bench_results.csv
    python3 scripts/format_results.py results/thresh_sweep.csv --sweep
"""

import csv
import sys
import argparse


def format_bench(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No data found.")
        return

    print("## Benchmark Results\n")
    header = ("| Graph | N | nnz | F | CPU (ms) | Naive (ms) | cuSPARSE (ms) "
              "| 3-Bucket (ms) | 2-Bucket (ms) | Speedup vs Naive | Speedup vs cuSPARSE |")
    sep    = ("|-------|---|-----|---|----------|------------|---------------"
              "|---------------|---------------|------------------|---------------------|")
    print(header)
    print(sep)

    for r in rows:
        naive_ms    = float(r['naive_ms'])
        cusparse_ms = float(r['cusparse_ms'])
        adapt3_ms   = float(r['adapt3_ms'])

        sp_naive    = naive_ms    / adapt3_ms if adapt3_ms > 0 else 0.0
        sp_cusparse = cusparse_ms / adapt3_ms if adapt3_ms > 0 else 0.0

        cpu_str = f"{float(r['cpu_ms']):.2f}" if float(r['cpu_ms']) > 0 else "—"

        print(f"| {r['graph']} | {r['N']} | {r['nnz']} | {r['F']} "
              f"| {cpu_str} | {naive_ms:.2f} | {cusparse_ms:.2f} "
              f"| {adapt3_ms:.2f} | {float(r['adapt2_ms']):.2f} "
              f"| {sp_naive:.2f}x | {sp_cusparse:.2f}x |")

    print()
    print("## Preprocessing Overhead\n")
    print("| Graph | F | Preprocess (ms) | 3-Bucket (ms) | Overhead % |")
    print("|-------|---|-----------------|---------------|------------|")
    for r in rows:
        pre_ms    = float(r['preprocess_ms'])
        adapt3_ms = float(r['adapt3_ms'])
        pct       = 100.0 * pre_ms / adapt3_ms if adapt3_ms > 0 else 0.0
        print(f"| {r['graph']} | {r['F']} | {pre_ms:.3f} | {adapt3_ms:.2f} | {pct:.1f}% |")


def format_sweep(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No data found.")
        return

    # Group by graph, find best thresh config per graph by adapt3_ms
    from collections import defaultdict
    by_graph = defaultdict(list)
    for r in rows:
        by_graph[r['graph']].append(r)

    print("## Threshold Sensitivity — Best Configuration per Graph\n")
    print("| Graph | thresh_low | thresh_med | 3-Bucket (ms) | 2-Bucket (ms) | vs default (8/64) |")
    print("|-------|------------|------------|---------------|---------------|-------------------|")

    for graph, graph_rows in sorted(by_graph.items()):
        best = min(graph_rows, key=lambda r: float(r['adapt3_ms']))
        default_rows = [r for r in graph_rows if r['thresh_low'] == '8' and r['thresh_med'] == '64']
        default_ms = float(default_rows[0]['adapt3_ms']) if default_rows else float(best['adapt3_ms'])
        speedup = default_ms / float(best['adapt3_ms'])
        print(f"| {graph} | {best['thresh_low']} | {best['thresh_med']} "
              f"| {float(best['adapt3_ms']):.2f} | {float(best['adapt2_ms']):.2f} "
              f"| {speedup:.2f}x |")

    print()
    print("## Full Sweep Table\n")
    print("| Graph | thresh_low | thresh_med | 3-Bucket (ms) | 2-Bucket (ms) |")
    print("|-------|------------|------------|---------------|---------------|")
    for r in sorted(rows, key=lambda r: (r['graph'], int(r['thresh_low']), int(r['thresh_med']))):
        print(f"| {r['graph']} | {r['thresh_low']} | {r['thresh_med']} "
              f"| {float(r['adapt3_ms']):.2f} | {float(r['adapt2_ms']):.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--sweep', action='store_true',
                        help='Format as threshold sensitivity sweep results')
    args = parser.parse_args()

    if args.sweep:
        format_sweep(args.csv_file)
    else:
        format_bench(args.csv_file)


if __name__ == '__main__':
    main()
