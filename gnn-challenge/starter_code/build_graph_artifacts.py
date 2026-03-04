import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData


def build_graph(edges_df, node_df, allowed_edge_types):
    node_ids = node_df["node_id"].tolist()
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    data = HeteroData()
    data["node"].num_nodes = len(node_ids)

    for etype in allowed_edge_types:
        df = edges_df[edges_df.edge_type == etype]
        src = torch.tensor([node_map[i] for i in df.src], dtype=torch.long)
        dst = torch.tensor([node_map[i] for i in df.dst], dtype=torch.long)
        data["node", etype, "node"].edge_index = torch.stack([src, dst])

    return data, node_map


def main():
    parser = argparse.ArgumentParser(description="Build and save PyG graph artifacts.")
    parser.add_argument(
        "--data-dir",
        default="data/public",
        help="Directory containing train.csv, test.csv, graph_edges.csv, node_types.csv",
    )
    parser.add_argument(
        "--out",
        default="data/public/graph_artifacts.pt",
        help="Output path for torch.save(...) artifact",
    )
    parser.add_argument(
        "--use-ancestry-in-test",
        action="store_true",
        help="Include ancestry edges in the test graph",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    edges_df = pd.read_csv(os.path.join(data_dir, "graph_edges.csv"))
    node_df = pd.read_csv(os.path.join(data_dir, "node_types.csv"))

    # Identify target column (train uses 'disease_labels' or 'target')
    target_col_train = "disease_labels" if "disease_labels" in train_df.columns else "target"

    # Build graphs
    train_graph, node_map = build_graph(edges_df, node_df, ["similarity"])
    test_edge_types = ["similarity", "ancestry"] if args.use_ancestry_in_test else ["similarity"]
    test_graph, _ = build_graph(edges_df, node_df, test_edge_types)

    # Node features: only shared columns between train and test
    shared_cols = set(train_df.columns).intersection(set(test_df.columns))
    feat_cols = sorted([c for c in shared_cols if c not in ["node_id", target_col_train, "sample_id"]])

    x = torch.zeros((len(node_map), len(feat_cols)))
    train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
    test_idx = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)
    x[train_idx] = torch.tensor(train_df[feat_cols].values, dtype=torch.float)
    x[test_idx] = torch.tensor(test_df[feat_cols].values, dtype=torch.float)

    train_graph["node"].x = x
    test_graph["node"].x = x

    # Labels (train only; others are -1)
    y = -1 * np.ones(len(node_map), dtype=int)
    y[train_idx] = train_df[target_col_train].values.astype(int)
    y = torch.tensor(y, dtype=torch.long)
    train_graph["node"].y = y
    test_graph["node"].y = y

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    artifact = {
        "train_graph": train_graph,
        "test_graph": test_graph,
        "node_map": node_map,
        "feat_cols": feat_cols,
        "target_col_train": target_col_train,
        "use_ancestry_in_test": args.use_ancestry_in_test,
    }
    torch.save(artifact, args.out)
    print(f"Saved graph artifacts to {args.out}")


if __name__ == "__main__":
    main()
