#!/usr/bin/env python3
"""Build a dense adjacency matrix CSV from graph_edges.csv.

Uses node ordering from data/public/node_types.csv.
Outputs data/public/adjacency_matrix.csv by default.
"""

import argparse
import csv
from pathlib import Path


def read_nodes(node_types_path: Path):
    with node_types_path.open(newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None or "node_id" not in header[0]:
            raise ValueError("node_types.csv must have a header with 'node_id'.")
        nodes = [row[0] for row in r]
    return nodes


def read_edges(edge_path: Path):
    edges = []
    with edge_path.open(newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None or header[:2] != ["src", "dst"]:
            raise ValueError("graph_edges.csv must have header: src,dst,edge_type")
        for row in r:
            if len(row) < 2:
                continue
            edges.append((row[0], row[1]))
    return edges


def build_adjacency(nodes, edges):
    n = len(nodes)
    index = {node_id: i for i, node_id in enumerate(nodes)}
    adj = [[0] * n for _ in range(n)]
    missing = 0
    for src, dst in edges:
        i = index.get(src)
        j = index.get(dst)
        if i is None or j is None:
            missing += 1
            continue
        adj[i][j] = 1
    return adj, missing


def write_csv(output_path: Path, nodes, adj):
    with output_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node_id"] + nodes)
        for node_id, row in zip(nodes, adj):
            w.writerow([node_id] + row)


def main():
    parser = argparse.ArgumentParser(description="Build adjacency matrix CSV from edge list.")
    parser.add_argument(
        "--node-types",
        default="data/public/node_types.csv",
        help="Path to node_types.csv (defines node order).",
    )
    parser.add_argument(
        "--edges",
        default="data/public/graph_edges.csv",
        help="Path to graph_edges.csv (edge list).",
    )
    parser.add_argument(
        "--output",
        default="data/public/adjacency_matrix.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    node_path = Path(args.node_types)
    edge_path = Path(args.edges)
    out_path = Path(args.output)

    nodes = read_nodes(node_path)
    edges = read_edges(edge_path)
    adj, missing = build_adjacency(nodes, edges)
    write_csv(out_path, nodes, adj)

    print(f"Wrote adjacency matrix to {out_path}")
    if missing:
        print(f"Warning: {missing} edges referenced missing node IDs")


if __name__ == "__main__":
    main()
