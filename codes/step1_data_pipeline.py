"""
Step 1 — Data Pipeline & Exploratory Analysis
Self-Supervised GNNs for Anomaly Detection in Financial Networks
Elliptic Bitcoin Dataset

WHAT THIS SCRIPT DOES (mirrors the 5 sub-steps in the execution guide):
  1. Load the three raw Elliptic CSVs and merge into one node table (indexed by txId).
  2. Build the directed edge index Ã and the 166-dim feature matrix X
     (94 local + 72 aggregated).
  3. Recover the time-step column (1..49) and verify the known label distribution
     (~2% illicit, ~21% licit, ~77% unknown).
  4. Plot: (a) label counts per time step, (b) node/edge counts per time step,
     (c) illicit-fraction per time step.
  5. Freeze the train/val/test split (steps 1-34 train, 35-49 test, with
     35-42 = pre-shutdown and 43-49 = post-shutdown flagged separately) and
     save everything to one versioned .npz artifact.

HOW TO RUN
  1. Put the three raw files in the same folder as this script (or edit RAW_DIR
     below), keeping the original Kaggle filenames:
       elliptic_txs_features.csv
       elliptic_txs_edgelist.csv
       elliptic_txs_classes.csv
  2. python step1_data_pipeline.py
  3. Outputs land in ./outputs/:
       elliptic_graph.npz          <- the frozen, versioned data object
       eda_label_counts_per_ts.png
       eda_node_edge_counts_per_ts.png
       eda_illicit_fraction_per_ts.png
       eda_summary.txt             <- printed stats, saved to disk too

DEPENDENCIES: pandas, numpy, scipy, matplotlib  (no torch/PyG required —
see the bottom of this file for a one-line snippet to wrap the .npz into a
PyG `Data` object once you're in an environment that has torch installed.)
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RAW_DIR = os.environ.get("ELLIPTIC_RAW_DIR", ".")
OUT_DIR = "outputs"
FEATURES_FILE = os.path.join(RAW_DIR, "elliptic_txs_features.csv")
EDGES_FILE = os.path.join(RAW_DIR, "elliptic_txs_edgelist.csv")
CLASSES_FILE = os.path.join(RAW_DIR, "elliptic_txs_classes.csv")

N_LOCAL_FEATS = 94   # includes the time-step feature itself
N_AGG_FEATS = 72
N_TOTAL_FEATS = N_LOCAL_FEATS + N_AGG_FEATS  # 166

TRAIN_TS_MAX = 34          # steps 1..34  -> train
TEST_TS_MIN = 35           # steps 35..49 -> test
SHUTDOWN_TS = 43           # dark-market shutdown; test splits into pre/post here
MAX_TS = 49

RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Step 1.1 — Load & merge
# --------------------------------------------------------------------------- #
def load_raw(features_path, edges_path, classes_path):
    for p in (features_path, edges_path, classes_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing raw file: {p}\n"
                f"Download the Elliptic Bitcoin dataset and place the three "
                f"CSVs (elliptic_txs_features.csv, elliptic_txs_edgelist.csv, "
                f"elliptic_txs_classes.csv) in '{RAW_DIR}', or set the "
                f"ELLIPTIC_RAW_DIR environment variable to point at them."
            )

    # Features file has NO header. Columns: txId, then 166 numeric features.
    # Kaggle layout: col0=txId, col1=time step, col1..col94 = local (94),
    # col95..col166 = aggregated (72).
    feat_cols = ["txId"] + [f"feat_{i}" for i in range(1, N_TOTAL_FEATS + 1)]
    features = pd.read_csv(features_path, header=None, names=feat_cols)

    classes = pd.read_csv(classes_path)  # columns: txId, class  (values: '1','2','unknown')
    classes.columns = [c.strip().lower() for c in classes.columns]
    classes = classes.rename(columns={"txid": "txId"})

    edges = pd.read_csv(edges_path)  # columns: txId1, txId2
    edges.columns = [c.strip().lower() for c in edges.columns]

    return features, edges, classes


def merge_nodes(features, classes):
    df = features.merge(classes, on="txId", how="left").copy()
    # Map class -> integer label: illicit=1, licit=0, unknown=-1
    label_map = {"1": 1, "2": 0, "unknown": -1}
    df["label"] = df["class"].map(label_map)
    if df["label"].isna().any():
        n_bad = df["label"].isna().sum()
        raise ValueError(
            f"{n_bad} rows had an unrecognized class value; check the raw "
            f"classes CSV for unexpected labels."
        )
    df["label"] = df["label"].astype(int)
    df = df.drop(columns=["class"])

    # time step lives in feat_1 by construction; expose it as its own column too
    df["time_step"] = df["feat_1"].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Step 1.2 — Build X and directed edge index Ã
# --------------------------------------------------------------------------- #
def build_feature_matrix(node_df):
    feat_cols = [f"feat_{i}" for i in range(1, N_TOTAL_FEATS + 1)]
    X = node_df[feat_cols].to_numpy(dtype=np.float32)
    assert X.shape[1] == N_TOTAL_FEATS == 166
    return X


def build_edge_index(node_df, edges_df):
    txid_to_idx = {tx: i for i, tx in enumerate(node_df["txId"].to_numpy())}

    src, dst = [], []
    dropped = 0
    for a, b in zip(edges_df["txid1"], edges_df["txid2"]):
        if a in txid_to_idx and b in txid_to_idx:
            src.append(txid_to_idx[a])
            dst.append(txid_to_idx[b])
        else:
            dropped += 1  # edge referencing a txId not present in features table

    edge_index = np.vstack([np.array(src, dtype=np.int64),
                             np.array(dst, dtype=np.int64)])

    n = len(node_df)
    adj = sp.coo_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)),
        shape=(n, n),
    ).tocsr()

    if dropped:
        print(f"  [warn] dropped {dropped} edges referencing unknown txIds")

    return edge_index, adj


# --------------------------------------------------------------------------- #
# Step 1.3 — Verify label distribution
# --------------------------------------------------------------------------- #
def label_distribution(node_df):
    counts = node_df["label"].value_counts().sort_index()
    total = len(node_df)
    dist = {
        "illicit": (counts.get(1, 0), counts.get(1, 0) / total),
        "licit": (counts.get(0, 0), counts.get(0, 0) / total),
        "unknown": (counts.get(-1, 0), counts.get(-1, 0) / total),
    }
    return dist


# --------------------------------------------------------------------------- #
# Step 1.4 — EDA plots
# --------------------------------------------------------------------------- #
def plot_label_counts_per_ts(node_df, out_path):
    pivot = (
        node_df.groupby(["time_step", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[-1, 0, 1], fill_value=0)
    )
    pivot.columns = ["unknown", "licit", "illicit"]

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot[["licit", "illicit", "unknown"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#2a9d8f", "#e63946", "#cfd8dc"], width=0.85,
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Node count")
    ax.set_title("Label counts per time step")
    ax.axvline(x=SHUTDOWN_TS - 1, color="black", linestyle="--", linewidth=1,
               label=f"shutdown (t={SHUTDOWN_TS})")
    ax.legend(title=None)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pivot


def plot_node_edge_counts_per_ts(node_df, edges_df, out_path):
    txid_to_ts = dict(zip(node_df["txId"], node_df["time_step"]))
    node_counts = node_df.groupby("time_step").size()

    # attribute an edge to the source node's time step
    edge_ts = edges_df["txid1"].map(txid_to_ts)
    edge_counts = edge_ts.value_counts().sort_index()

    ts_range = np.arange(1, MAX_TS + 1)
    node_counts = node_counts.reindex(ts_range, fill_value=0)
    edge_counts = edge_counts.reindex(ts_range, fill_value=0)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(ts_range, node_counts.values, color="#264653", marker="o",
              markersize=3, label="nodes")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Node count", color="#264653")
    ax1.tick_params(axis="y", labelcolor="#264653")

    ax2 = ax1.twinx()
    ax2.plot(ts_range, edge_counts.values, color="#e76f51", marker="s",
              markersize=3, label="edges")
    ax2.set_ylabel("Edge count", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    ax1.axvline(x=SHUTDOWN_TS, color="black", linestyle="--", linewidth=1)
    ax1.set_title("Node & edge counts per time step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return node_counts, edge_counts


def plot_illicit_fraction_per_ts(node_df, out_path):
    labeled = node_df[node_df["label"] >= 0]
    frac = (
        labeled.groupby("time_step")["label"]
        .mean()  # mean of {0,1} = illicit fraction among labeled nodes
        .reindex(np.arange(1, MAX_TS + 1))
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frac.index, frac.values, color="#e63946", marker="o", markersize=4)
    ax.axvline(x=SHUTDOWN_TS, color="black", linestyle="--", linewidth=1,
               label=f"shutdown (t={SHUTDOWN_TS})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Illicit fraction (among labeled nodes)")
    ax.set_title("Illicit fraction per time step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return frac


# --------------------------------------------------------------------------- #
# Step 1.5 — Freeze train/val/test split
# --------------------------------------------------------------------------- #
def build_split_masks(node_df):
    ts = node_df["time_step"].to_numpy()
    train_mask = ts <= TRAIN_TS_MAX
    test_mask = ts >= TEST_TS_MIN
    pre_shutdown_mask = (ts >= TEST_TS_MIN) & (ts < SHUTDOWN_TS)
    post_shutdown_mask = ts >= SHUTDOWN_TS

    assert (train_mask | test_mask).all(), "every node must fall in train or test"
    assert not (train_mask & test_mask).any(), "train/test must not overlap"
    assert (pre_shutdown_mask | post_shutdown_mask | train_mask).all()

    return {
        "train_mask": train_mask,
        "test_mask": test_mask,
        "pre_shutdown_mask": pre_shutdown_mask,
        "post_shutdown_mask": post_shutdown_mask,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("Loading raw CSVs...")
    features, edges, classes = load_raw(FEATURES_FILE, EDGES_FILE, CLASSES_FILE)
    log(f"  features: {features.shape}, edges: {edges.shape}, classes: {classes.shape}")

    log("Merging into single node table...")
    node_df = merge_nodes(features, classes)
    log(f"  merged node table: {node_df.shape}")

    log("Building feature matrix X and edge index...")
    X = build_feature_matrix(node_df)
    edge_index, adj = build_edge_index(node_df, edges)
    log(f"  X: {X.shape}  (expected (N, 166))")
    log(f"  edge_index: {edge_index.shape}  (expected (2, E))")

    log("Verifying label distribution...")
    dist = label_distribution(node_df)
    for k, (count, pct) in dist.items():
        log(f"  {k:8s}: {count:7d}  ({pct*100:.2f}%)")

    log("Generating EDA plots...")
    plot_label_counts_per_ts(node_df, os.path.join(OUT_DIR, "eda_label_counts_per_ts.png"))
    plot_node_edge_counts_per_ts(node_df, edges, os.path.join(OUT_DIR, "eda_node_edge_counts_per_ts.png"))
    illicit_frac = plot_illicit_fraction_per_ts(node_df, os.path.join(OUT_DIR, "eda_illicit_fraction_per_ts.png"))
    peak_ts = illicit_frac.idxmax()
    log(f"  peak illicit fraction at time step {peak_ts} ({illicit_frac.max()*100:.2f}%)")

    log("Freezing train/val/test split...")
    masks = build_split_masks(node_df)
    for name, m in masks.items():
        log(f"  {name}: {m.sum()} nodes")

    log("Saving versioned data object -> outputs/elliptic_graph.npz")
    np.savez_compressed(
        os.path.join(OUT_DIR, "elliptic_graph.npz"),
        X=X,
        edge_index=edge_index,
        labels=node_df["label"].to_numpy(dtype=np.int64),
        time_step=node_df["time_step"].to_numpy(dtype=np.int64),
        txid=node_df["txId"].to_numpy(),
        train_mask=masks["train_mask"],
        test_mask=masks["test_mask"],
        pre_shutdown_mask=masks["pre_shutdown_mask"],
        post_shutdown_mask=masks["post_shutdown_mask"],
    )
    sp.save_npz(os.path.join(OUT_DIR, "elliptic_adj.npz"), adj)

    with open(os.path.join(OUT_DIR, "eda_summary.txt"), "w") as f:
        f.write("\n".join(log_lines))

    log("Done.")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# OPTIONAL: convert the saved .npz into a PyTorch Geometric Data object once
# you're in an environment with torch + torch_geometric installed (e.g. Colab).
# --------------------------------------------------------------------------- #
"""
import numpy as np
import torch
from torch_geometric.data import Data

d = np.load("outputs/elliptic_graph.npz", allow_pickle=True)
data = Data(
    x=torch.tensor(d["X"], dtype=torch.float),
    edge_index=torch.tensor(d["edge_index"], dtype=torch.long),
    y=torch.tensor(d["labels"], dtype=torch.long),
)
data.time_step = torch.tensor(d["time_step"], dtype=torch.long)
data.train_mask = torch.tensor(d["train_mask"])
data.test_mask = torch.tensor(d["test_mask"])
data.pre_shutdown_mask = torch.tensor(d["pre_shutdown_mask"])
data.post_shutdown_mask = torch.tensor(d["post_shutdown_mask"])
"""
