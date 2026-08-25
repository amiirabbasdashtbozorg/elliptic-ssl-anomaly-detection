"""
Step 2 — Supervised Baseline (R2)
Self-Supervised GNNs for Anomaly Detection in Financial Networks
Elliptic Bitcoin Dataset

WHAT THIS SCRIPT DOES
  1. Loads the frozen graph object from Step 1 (outputs/elliptic_graph.npz).
  2. Builds a symmetrically-normalized adjacency (Kipf & Welling GCN
     normalization: D^-1/2 (A + I) D^-1/2), implemented directly with
     torch.sparse — no torch_geometric dependency required.
  3. Trains a 2-layer GCN end-to-end on LABELED nodes only, restricted to
     the train split (time steps 1-34), following Weber et al. (2019).
  4. Optionally (--model gcn_concat) implements the Alarab et al. (2020)
     variant: concatenate the GCN's final-layer embedding with a linearly
     transformed copy of the raw 166-dim features before the classifier head.
  5. Evaluates on the frozen test split (time steps 35-49) and reports
     precision / recall / F1 for the illicit class, micro-F1, and AUC-PR.
  6. Saves a metrics table (CSV + JSON) to outputs/ for reuse in Step 6.

HOW TO RUN
  python step2_supervised_baseline.py \
      --graph outputs/elliptic_graph.npz \
      --model gcn            # or: gcn_concat  (Alarab et al. variant)

DEPENDENCIES: torch, numpy, scikit-learn
  pip install torch --break-system-packages   (CPU build is fine)
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, f1_score, average_precision_score

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------------------------------- #
# Graph loading & normalization
# --------------------------------------------------------------------------- #
def load_graph(path):
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"],
        "edge_index": d["edge_index"],
        "labels": d["labels"],
        "time_step": d["time_step"],
        "train_mask": d["train_mask"],
        "test_mask": d["test_mask"],
        "pre_shutdown_mask": d["pre_shutdown_mask"],
        "post_shutdown_mask": d["post_shutdown_mask"],
    }


def build_normalized_adj(edge_index, num_nodes):
    """Symmetric GCN normalization: D^-1/2 (A + A^T + I) D^-1/2, as a
    torch sparse COO tensor. Symmetrizing is standard practice for GCN
    message passing on Elliptic (Weber et al. treat the transaction graph
    as undirected for propagation purposes)."""
    src, dst = edge_index[0], edge_index[1]

    # symmetrize: add reverse edges
    row = np.concatenate([src, dst, np.arange(num_nodes)])
    col = np.concatenate([dst, src, np.arange(num_nodes)])
    vals = np.ones(len(row), dtype=np.float32)

    # collapse duplicate (row, col) pairs (e.g. edges that already existed
    # in both directions, or multi-edges) so the degree count is correct
    idx, inv = np.unique(np.stack([row, col]), axis=1, return_inverse=True)
    row, col = idx[0], idx[1]
    vals = np.ones(len(row), dtype=np.float32)

    deg = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(deg, row, 1.0)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)

    norm_vals = deg_inv_sqrt[row] * vals * deg_inv_sqrt[col]

    indices = torch.tensor(np.stack([row, col]), dtype=torch.long)
    values = torch.tensor(norm_vals, dtype=torch.float32)
    with torch.sparse.check_sparse_tensor_invariants(False):
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    return adj


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, adj, x):
        return torch.sparse.mm(adj, self.lin(x))


class GCNBaseline(nn.Module):
    """Plain 2-layer GCN classifier (Weber et al. 2019 style)."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, adj, x):
        h = F.relu(self.gc1(adj, x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gc2(adj, h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        return logits, h


class GCNConcatBaseline(nn.Module):
    """Alarab et al. (2020) variant: concatenate the GCN embedding with a
    linearly-transformed copy of the raw features before the classifier."""

    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.raw_proj = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = dropout

    def forward(self, adj, x):
        h = F.relu(self.gc1(adj, x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gc2(adj, h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        raw = F.relu(self.raw_proj(x))
        combined = torch.cat([h, raw], dim=1)
        logits = self.classifier(combined)
        return logits, combined


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def get_labeled_mask(labels, split_mask):
    return split_mask & (labels >= 0)


def compute_metrics(y_true, y_pred, y_score_illicit):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    try:
        auc_pr = average_precision_score(y_true, y_score_illicit)
    except ValueError:
        auc_pr = float("nan")

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    return {
        "illicit_precision": float(precision[0]),
        "illicit_recall": float(recall[0]),
        "illicit_f1": float(f1[0]),
        "micro_f1": float(micro_f1),
        "auc_pr": float(auc_pr),
        "n_illicit_true": int(support[0]),
        "n_total": int(len(y_true)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def train(model, adj, X, labels, train_idx, device, epochs=200, lr=0.01,
          weight_decay=5e-4, class_weighted=True, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    y_train = labels[train_idx]
    if class_weighted:
        counts = torch.bincount(y_train, minlength=2).float()
        weights = (counts.sum() / (2.0 * counts.clamp(min=1)))
        weights = weights.to(device)
    else:
        weights = None

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(adj, X)
        loss = F.cross_entropy(logits[train_idx], y_train, weight=weights)
        loss.backward()
        optimizer.step()

        if verbose and (epoch % 20 == 0 or epoch == 1):
            with torch.no_grad():
                pred = logits[train_idx].argmax(dim=1)
                acc = (pred == y_train).float().mean().item()
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}  train_acc={acc:.4f}")
        history.append(loss.item())

    return history


@torch.no_grad()
def evaluate(model, adj, X, labels, eval_idx):
    model.eval()
    logits, _ = model(adj, X)
    probs = F.softmax(logits[eval_idx], dim=1)
    y_true = labels[eval_idx].cpu().numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()
    y_score = probs[:, 1].cpu().numpy()  # P(illicit)
    return compute_metrics(y_true, y_pred, y_score)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/elliptic_graph.npz")
    parser.add_argument("--model", choices=["gcn", "gcn_concat"], default="gcn")
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_class_weight", action="store_true",
                         help="disable inverse-frequency class weighting in the loss")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--eval_only", default=None,
                         help="path to a saved .pt checkpoint; skip training and just "
                              "re-evaluate (with raw TP/FP/FN counts) using this model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading graph from {args.graph} ...")
    g = load_graph(args.graph)
    num_nodes = g["X"].shape[0]
    print(f"  nodes={num_nodes}  features={g['X'].shape[1]}  edges={g['edge_index'].shape[1]}")

    X = torch.tensor(g["X"], dtype=torch.float32, device=device)
    labels = torch.tensor(g["labels"], dtype=torch.long, device=device)  # -1/0/1

    print("Building normalized adjacency ...")
    adj = build_normalized_adj(g["edge_index"], num_nodes).to(device)

    train_mask = torch.tensor(g["train_mask"])
    test_mask = torch.tensor(g["test_mask"])
    pre_mask = torch.tensor(g["pre_shutdown_mask"])
    post_mask = torch.tensor(g["post_shutdown_mask"])

    labels_cpu = torch.tensor(g["labels"], dtype=torch.long)
    train_labeled = get_labeled_mask(labels_cpu, train_mask).nonzero(as_tuple=True)[0].to(device)
    test_labeled = get_labeled_mask(labels_cpu, test_mask).nonzero(as_tuple=True)[0].to(device)
    pre_labeled = get_labeled_mask(labels_cpu, pre_mask).nonzero(as_tuple=True)[0].to(device)
    post_labeled = get_labeled_mask(labels_cpu, post_mask).nonzero(as_tuple=True)[0].to(device)

    print(f"  labeled train nodes: {len(train_labeled)}   labeled test nodes: {len(test_labeled)}")
    print(f"  (pre-shutdown labeled: {len(pre_labeled)}, post-shutdown labeled: {len(post_labeled)})")

    # labels tensor for training loss must be 0/1 only (illicit=1, licit=0);
    # -1 (unknown) rows are excluded via the *_labeled index tensors above,
    # so it's safe to clamp here purely to keep cross_entropy's class range valid.
    labels_for_loss = labels.clamp(min=0)

    model_cls = GCNBaseline if args.model == "gcn" else GCNConcatBaseline
    model = model_cls(
        in_dim=X.shape[1], hidden_dim=args.hidden_dim, num_classes=2, dropout=args.dropout
    ).to(device)
    print(f"Model: {args.model}  ({sum(p.numel() for p in model.parameters())} params)")

    if args.eval_only:
        print(f"Loading checkpoint {args.eval_only} (skipping training) ...")
        model.load_state_dict(torch.load(args.eval_only, map_location=device, weights_only=True))
    else:
        print("Training ...")
        train(
            model, adj, X, labels_for_loss, train_labeled, device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            class_weighted=not args.no_class_weight,
        )

    print("Evaluating ...")
    test_metrics = evaluate(model, adj, X, labels_for_loss, test_labeled)
    pre_metrics = evaluate(model, adj, X, labels_for_loss, pre_labeled)
    post_metrics = evaluate(model, adj, X, labels_for_loss, post_labeled)

    results = {
        "model": args.model,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "class_weighted": not args.no_class_weight,
        "test_35_49": test_metrics,
        "test_pre_shutdown_35_42": pre_metrics,
        "test_post_shutdown_43_49": post_metrics,
    }

    print("\n=== Test metrics (time steps 35-49) ===")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v}")
    print("\n=== Pre-shutdown (35-42) vs Post-shutdown (43-49) illicit-F1 ===")
    print(f"  pre : F1={pre_metrics['illicit_f1']:.4f}  "
          f"(n_illicit_true={pre_metrics['n_illicit_true']}, n_total={pre_metrics['n_total']}, "
          f"TP={pre_metrics['tp']}, FP={pre_metrics['fp']}, FN={pre_metrics['fn']})")
    print(f"  post: F1={post_metrics['illicit_f1']:.4f}  "
          f"(n_illicit_true={post_metrics['n_illicit_true']}, n_total={post_metrics['n_total']}, "
          f"TP={post_metrics['tp']}, FP={post_metrics['fp']}, FN={post_metrics['fn']})")

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"step2_baseline_{args.model}_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_csv = os.path.join(args.out_dir, f"step2_baseline_{args.model}_metrics.csv")
    with open(out_csv, "w") as f:
        f.write("split,illicit_precision,illicit_recall,illicit_f1,micro_f1,auc_pr\n")
        for split_name, m in [("test_35_49", test_metrics),
                               ("pre_shutdown_35_42", pre_metrics),
                               ("post_shutdown_43_49", post_metrics)]:
            f.write(f"{split_name},{m['illicit_precision']:.4f},{m['illicit_recall']:.4f},"
                    f"{m['illicit_f1']:.4f},{m['micro_f1']:.4f},{m['auc_pr']:.4f}\n")

    # save the trained model + embeddings-producing state for reuse/inspection
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"step2_baseline_{args.model}.pt"))

    print(f"\nSaved metrics -> {out_json}, {out_csv}")
    print(f"Saved model weights -> outputs/step2_baseline_{args.model}.pt")


if __name__ == "__main__":
    main()
