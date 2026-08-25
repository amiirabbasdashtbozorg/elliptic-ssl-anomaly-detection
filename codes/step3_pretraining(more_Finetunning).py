"""
Step 3 — Self-Supervised Pretraining (R3)
Self-Supervised GNNs for Anomaly Detection in Financial Networks
Elliptic Bitcoin Dataset

WHAT THIS SCRIPT DOES
  Primary pretext task: node-level graph contrastive learning, GRACE-style.
    - Two augmented views of the graph are generated per epoch via
      edge dropout + feature-dimension masking.
    - A shared GCN encoder embeds both views.
    - A small projection head maps embeddings into a contrastive space.
    - NT-Xent / InfoNCE loss pulls the two views of the SAME node together
      and pushes apart all other nodes in a random mini-batch sampled each
      step (full N x N similarity is infeasible at 200K+ nodes, so the loss
      is computed on a subsample of node embeddings each step, not the
      graph structure itself — every node still gets full-neighborhood
      message passing before subsampling).
    - Labels are NEVER touched anywhere in this file.

  Leakage ablation (R3.3): every run trains under one of two SCOPES:
    - "trainonly": pretraining sees only the induced subgraph over nodes
      with time_step <= 34 (same nodes the downstream task is allowed to
      use). Edges to future (35-49) nodes are removed, not just the nodes.
    - "full": pretraining sees the entire graph (all 49 time steps),
      exploiting all unlabeled nodes at the cost of a weaker leakage
      guarantee.
    Pass --scope both (default here, per project decision) to train BOTH
    and get two checkpoints + two curves for direct comparison.

  Optional second pretext task (masked feature reconstruction, R3.4):
    pass --second_task to also train a BERT-style masked-feature-reconstruction
    encoder (mask a random subset of each node's 166 features, reconstruct
    them from the unmasked features + neighborhood) under the same scope(s),
    producing a second family of checkpoint(s) for the Step 3/7 ablation
    table (contrastive vs. generative).

HOW TO RUN
  # primary task only, both scopes (what the project decided on):
  python step3_pretraining.py --graph outputs/elliptic_graph.npz

  # add the second pretext task too:
  python step3_pretraining.py --graph outputs/elliptic_graph.npz --second_task

  # just one scope, e.g. for a quick iteration:
  python step3_pretraining.py --graph outputs/elliptic_graph.npz --scope trainonly

OUTPUTS (all under --out_dir, default outputs/)
  step3_contrastive_<scope>.pt         encoder+projection-head checkpoint
  step3_contrastive_<scope>_curve.csv  per-epoch loss log
  step3_contrastive_<scope>_curve.png  loss curve plot
  step3_masked_<scope>.pt / _curve.*   (only if --second_task)

DEPENDENCIES: torch, numpy, matplotlib
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------------------------------- #
# Graph loading, subgraph construction, normalization
# (kept self-contained / duplicated from step2 on purpose — step2 is already
#  finalized for the report and shouldn't be touched retroactively)
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


def build_subgraph(g, node_mask):
    """Induce a subgraph on nodes where node_mask is True: keeps only those
    nodes' features, and only edges where BOTH endpoints survive (dropping
    edges into the future, not just dropping future nodes from the loss)."""
    keep_idx = np.nonzero(node_mask)[0]
    old_to_new = -np.ones(len(node_mask), dtype=np.int64)
    old_to_new[keep_idx] = np.arange(len(keep_idx))

    X_sub = g["X"][keep_idx]
    time_step_sub = g["time_step"][keep_idx]

    src, dst = g["edge_index"][0], g["edge_index"][1]
    both_kept = node_mask[src] & node_mask[dst]
    new_src = old_to_new[src[both_kept]]
    new_dst = old_to_new[dst[both_kept]]
    edge_index_sub = np.stack([new_src, new_dst])

    return {
        "X": X_sub,
        "edge_index": edge_index_sub,
        "time_step": time_step_sub,
        "num_nodes": len(keep_idx),
    }


def build_normalized_adj(edge_index, num_nodes):
    """Symmetric GCN normalization D^-1/2 (A + A^T + I) D^-1/2, as a torch
    sparse COO tensor. Recomputed fresh each time it's called since edge
    dropout augmentation changes the edge set every step."""
    src, dst = edge_index[0], edge_index[1]
    row = np.concatenate([src, dst, np.arange(num_nodes)])
    col = np.concatenate([dst, src, np.arange(num_nodes)])

    idx = np.unique(np.stack([row, col]), axis=1)
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
# GRACE-style augmentations
# --------------------------------------------------------------------------- #
def drop_edges(edge_index, drop_prob):
    """Remove each edge independently with probability drop_prob."""
    num_edges = edge_index.shape[1]
    keep = np.random.rand(num_edges) > drop_prob
    return edge_index[:, keep]


def mask_feature_dims(X, mask_prob):
    """GRACE-style feature masking: zero out entire feature DIMENSIONS
    (columns), the same dimensions for every node, chosen independently
    each call with probability mask_prob. Zeroing whole dimensions (rather
    than random per-node-per-feature entries) is the augmentation used in
    the original GRACE paper and forces the encoder not to rely on any
    single feature column being reliably present."""
    num_feats = X.shape[1]
    mask = (torch.rand(num_feats, device=X.device) > mask_prob).float()
    return X * mask.unsqueeze(0)


def augment_view(X, edge_index, num_nodes, edge_drop_prob, feat_mask_prob, device):
    edge_index_aug = drop_edges(edge_index, edge_drop_prob)
    adj_aug = build_normalized_adj(edge_index_aug, num_nodes).to(device)
    X_aug = mask_feature_dims(X, feat_mask_prob)
    return adj_aug, X_aug


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, adj, x):
        return torch.sparse.mm(adj, self.lin(x))


class GCNEncoder(nn.Module):
    """Shared 2-layer GCN encoder, reused (frozen or fine-tuned) by Step 4."""

    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, adj, x):
        h = F.relu(self.gc1(adj, x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gc2(adj, h))
        return h


class ProjectionHead(nn.Module):
    """Small MLP mapping encoder output into the space where the
    contrastive loss is computed — standard SimCLR/GRACE practice, so the
    encoder embedding itself isn't forced to be directly optimal for the
    (somewhat arbitrary) contrastive temperature/geometry."""

    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, h):
        return self.net(h)


class MaskedFeatureDecoder(nn.Module):
    """Small decoder head reconstructing the original 166-dim feature
    vector from the encoder's embedding, used only for the optional second
    pretext task."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, h):
        return self.net(h)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def nt_xent_loss(z1, z2, temperature=0.5):
    """Symmetric InfoNCE / NT-Xent over a batch of paired embeddings.
    z1[i] and z2[i] are the two augmented views of the SAME node -> positive
    pair. Every other embedding in the batch (from either view) is a
    negative. z1, z2: (B, proj_dim)."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)  # (2B, proj_dim)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # mask out self-similarity on the diagonal
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float("-inf"))

    # positive of row i (i<B -> from view1) is row i+B (view2), and vice versa
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)

    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(2 * B, device=z.device), pos_idx]
    return loss.mean()


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def train_contrastive(X, edge_index, num_nodes, device, epochs, lr, hidden_dim,
                       proj_dim, edge_drop_prob, feat_mask_prob, temperature,
                       batch_size, verbose=True):
    encoder = GCNEncoder(in_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    proj_head = ProjectionHead(hidden_dim, proj_dim).to(device)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    # cosine decay so the last stretch of (many more) epochs actually settles
    # into a minimum instead of the loss oscillating at a fixed high lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        optimizer.zero_grad()

        adj1, X1 = augment_view(X, edge_index, num_nodes, edge_drop_prob, feat_mask_prob, device)
        adj2, X2 = augment_view(X, edge_index, num_nodes, edge_drop_prob, feat_mask_prob, device)

        h1 = encoder(adj1, X1)
        h2 = encoder(adj2, X2)

        batch_idx = torch.randint(0, num_nodes, (min(batch_size, num_nodes),), device=device)
        z1 = proj_head(h1[batch_idx])
        z2 = proj_head(h2[batch_idx])

        loss = nt_xent_loss(z1, z2, temperature=temperature)
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(loss.item())
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}  contrastive_loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

    return encoder, proj_head, history


def train_masked_feature(X, edge_index, num_nodes, device, epochs, lr, hidden_dim,
                          mask_prob, verbose=True):
    encoder = GCNEncoder(in_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    decoder = MaskedFeatureDecoder(hidden_dim, X.shape[1]).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)

    adj = build_normalized_adj(edge_index, num_nodes).to(device)

    history = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        # fresh per-node-per-feature mask each epoch (unlike the contrastive
        # task's dimension-wide mask — here we want the model to reconstruct
        # arbitrary held-out entries, closer to the BERT masked-LM analogy)
        keep_prob = 1.0 - mask_prob
        mask = (torch.rand_like(X) < keep_prob).float()  # 1 = visible, 0 = masked
        X_masked = X * mask

        h = encoder(adj, X_masked)
        X_recon = decoder(h)

        # loss only on the masked (held-out) entries
        hole = 1.0 - mask
        loss = ((X_recon - X) ** 2 * hole).sum() / hole.sum().clamp(min=1.0)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}  masked_feat_mse={loss.item():.4f}")

    return encoder, decoder, history


# --------------------------------------------------------------------------- #
# Plotting / saving
# --------------------------------------------------------------------------- #
def save_curve(history, title, csv_path, png_path):
    with open(csv_path, "w") as f:
        f.write("epoch,loss\n")
        for i, v in enumerate(history, start=1):
            f.write(f"{i},{v:.6f}\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(history) + 1), history, color="#264653")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/elliptic_graph.npz")
    parser.add_argument("--scope", choices=["trainonly", "full", "both"], default="both")
    parser.add_argument("--second_task", action="store_true",
                         help="also run masked-feature-reconstruction pretraining")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400,
                         help="raised from the original 150 — the loss curve hadn't "
                              "converged by epoch 150 in the initial run")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--edge_drop_prob", type=float, default=0.2)
    parser.add_argument("--feat_mask_prob", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8192,
                         help="raised from 4096 — more negatives per InfoNCE step "
                              "gives a stronger, less noisy contrastive signal")
    parser.add_argument("--masked_epochs", type=int, default=100)
    parser.add_argument("--masked_mask_prob", type=float, default=0.3)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading graph from {args.graph} ...")
    g = load_graph(args.graph)
    print(f"  full graph: nodes={g['X'].shape[0]}  edges={g['edge_index'].shape[1]}")

    scopes = ["trainonly", "full"] if args.scope == "both" else [args.scope]

    for scope in scopes:
        print(f"\n{'='*70}\nSCOPE = {scope}\n{'='*70}")
        if scope == "trainonly":
            node_mask = g["time_step"] <= 34
            sub = build_subgraph(g, node_mask)
        else:
            sub = {
                "X": g["X"],
                "edge_index": g["edge_index"],
                "time_step": g["time_step"],
                "num_nodes": g["X"].shape[0],
            }
        print(f"  scope graph: nodes={sub['num_nodes']}  edges={sub['edge_index'].shape[1]}")

        X = torch.tensor(sub["X"], dtype=torch.float32, device=device)
        edge_index = sub["edge_index"]
        num_nodes = sub["num_nodes"]

        # --- primary task: contrastive ---
        print(f"\n[{scope}] Training GRACE-style contrastive encoder ...")
        encoder, proj_head, history = train_contrastive(
            X, edge_index, num_nodes, device,
            epochs=args.epochs, lr=args.lr, hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim, edge_drop_prob=args.edge_drop_prob,
            feat_mask_prob=args.feat_mask_prob, temperature=args.temperature,
            batch_size=args.batch_size,
        )
        ckpt_path = os.path.join(args.out_dir, f"step3_contrastive_{scope}.pt")
        torch.save({
            "encoder_state_dict": encoder.state_dict(),
            "proj_head_state_dict": proj_head.state_dict(),
            "hidden_dim": args.hidden_dim,
            "proj_dim": args.proj_dim,
            "in_dim": X.shape[1],
            "scope": scope,
            "task": "contrastive",
        }, ckpt_path)
        save_curve(
            history, f"Contrastive pretraining loss ({scope})",
            os.path.join(args.out_dir, f"step3_contrastive_{scope}_curve.csv"),
            os.path.join(args.out_dir, f"step3_contrastive_{scope}_curve.png"),
        )
        print(f"  saved checkpoint -> {ckpt_path}")

        # --- optional second task: masked feature reconstruction ---
        if args.second_task:
            print(f"\n[{scope}] Training masked-feature-reconstruction encoder ...")
            m_encoder, m_decoder, m_history = train_masked_feature(
                X, edge_index, num_nodes, device,
                epochs=args.masked_epochs, lr=args.lr, hidden_dim=args.hidden_dim,
                mask_prob=args.masked_mask_prob,
            )
            m_ckpt_path = os.path.join(args.out_dir, f"step3_masked_{scope}.pt")
            torch.save({
                "encoder_state_dict": m_encoder.state_dict(),
                "decoder_state_dict": m_decoder.state_dict(),
                "hidden_dim": args.hidden_dim,
                "in_dim": X.shape[1],
                "scope": scope,
                "task": "masked_feature",
            }, m_ckpt_path)
            save_curve(
                m_history, f"Masked-feature reconstruction loss ({scope})",
                os.path.join(args.out_dir, f"step3_masked_{scope}_curve.csv"),
                os.path.join(args.out_dir, f"step3_masked_{scope}_curve.png"),
            )
            print(f"  saved checkpoint -> {m_ckpt_path}")

    print("\nDone. All checkpoints + training curves saved under", args.out_dir)


if __name__ == "__main__":
    main()
