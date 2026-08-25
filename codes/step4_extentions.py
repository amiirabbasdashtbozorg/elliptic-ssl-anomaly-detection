%%writefile step4_probe_finetune.py

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, f1_score, average_precision_score

torch.manual_seed(42)

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
# Models & Extensions
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, adj, x):
        return torch.sparse.mm(adj, self.lin(x))

class GCNEncoder(nn.Module):
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

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, num_classes),
        )

    def forward(self, h):
        return self.net(h)

class ProbeFineTuneModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, adj, x):
        h = self.encoder(adj, x)
        return self.head(h), h

# --- Extension A2: Domain-Adversarial Training ---
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, alpha):
        x = GradientReversalLayer.apply(x, alpha)
        return self.net(x)

# --- Extension B1: Teacher Model (Step 2 Baseline) ---
class GCNBaseline(nn.Module):
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

# --------------------------------------------------------------------------- #
# Checkpoint loading & Utils
# --------------------------------------------------------------------------- #
def build_encoder_from_checkpoint(checkpoint_path, in_dim, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_dim = ckpt["hidden_dim"]
    encoder = GCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    return encoder, hidden_dim

def subsample_labeled_indices(labeled_idx_sorted, frac, seed):
    if frac >= 1.0: return labeled_idx_sorted
    rng = np.random.RandomState(seed)
    n_keep = max(1, int(round(len(labeled_idx_sorted) * frac)))
    perm = rng.permutation(len(labeled_idx_sorted))
    keep = np.sort(perm[:n_keep])
    return labeled_idx_sorted[keep]

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
        "illicit_precision": float(precision[0]), "illicit_recall": float(recall[0]),
        "illicit_f1": float(f1[0]), "micro_f1": float(micro_f1), "auc_pr": float(auc_pr),
        "n_illicit_true": int(support[0]), "n_total": int(len(y_true)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train(model, adj, X, labels, train_idx, device, mode, epochs, head_lr,
          encoder_lr, weight_decay, class_weighted=True, verbose=True,
          teacher_model=None, distill_weight=0.5, distill_temp=2.0,
          domain_discriminator=None, domain_adv_weight=0.1, time_step=None):

    params = []
    if mode == "probe":
        for p in model.encoder.parameters(): p.requires_grad = False
        params.append({"params": model.head.parameters(), "lr": head_lr})
    else:
        for p in model.encoder.parameters(): p.requires_grad = True
        params.append({"params": model.encoder.parameters(), "lr": encoder_lr})
        params.append({"params": model.head.parameters(), "lr": head_lr})

    if domain_discriminator is not None:
        params.append({"params": domain_discriminator.parameters(), "lr": head_lr})

    optimizer = torch.optim.Adam(params, weight_decay=weight_decay)

    y_train = labels[train_idx]
    if class_weighted:
        counts = torch.bincount(y_train, minlength=2).float()
        weights = (counts.sum() / (2.0 * counts.clamp(min=1))).to(device)
    else:
        weights = None

    if domain_discriminator is not None:
        pre_idx = (time_step < 43).nonzero(as_tuple=True)[0]
        post_idx = (time_step >= 43).nonzero(as_tuple=True)[0]

    for epoch in range(1, epochs + 1):
        model.train()
        if mode == "probe": model.encoder.eval()
        if domain_discriminator is not None: domain_discriminator.train()

        optimizer.zero_grad()
        logits, h = model(adj, X)

        # 1. Supervised Loss
        loss_sup = F.cross_entropy(logits[train_idx], y_train, weight=weights)
        total_loss = loss_sup

        # 2. Extension B1: Knowledge Distillation Loss
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits, _ = teacher_model(adj, X)
            loss_kd = F.kl_div(
                F.log_softmax(logits[train_idx] / distill_temp, dim=1),
                F.softmax(teacher_logits[train_idx] / distill_temp, dim=1),
                reduction='batchmean'
            ) * (distill_temp ** 2)
            total_loss += distill_weight * loss_kd

        # 3. Extension A2: Domain-Adversarial Loss
        if domain_discriminator is not None:
            p = float(epoch) / epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1.

            idx_pre = pre_idx[torch.randint(0, len(pre_idx), (1024,))]
            idx_post = post_idx[torch.randint(0, len(post_idx), (1024,))]
            dom_idx = torch.cat([idx_pre, idx_post])
            dom_labels = torch.cat([torch.zeros(1024, dtype=torch.long), torch.ones(1024, dtype=torch.long)]).to(device)

            dom_logits = domain_discriminator(h[dom_idx], alpha)
            loss_da = F.cross_entropy(dom_logits, dom_labels)
            total_loss += domain_adv_weight * loss_da

        total_loss.backward()
        optimizer.step()

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            with torch.no_grad():
                pred = logits[train_idx].argmax(dim=1)
                acc = (pred == y_train).float().mean().item()
            msg = f"  epoch {epoch:4d}  loss={total_loss.item():.4f}  train_acc={acc:.4f}"
            if teacher_model is not None: msg += f"  kd_loss={loss_kd.item():.4f}"
            if domain_discriminator is not None: msg += f"  da_loss={loss_da.item():.4f}"
            print(msg)

@torch.no_grad()
def evaluate(model, adj, X, labels, eval_idx):
    model.eval()
    logits, _ = model(adj, X)
    probs = F.softmax(logits[eval_idx], dim=1)
    y_true = labels[eval_idx].cpu().numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()
    y_score = probs[:, 1].cpu().numpy()
    return compute_metrics(y_true, y_pred, y_score)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_experiment(args, device=None, g=None, X=None, labels_raw=None, adj=None, verbose=True):
    if args.checkpoint is None and args.mode == "probe":
        args.mode = "finetune"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if g is None:
        g, X, labels_raw, adj = prepare_graph(args.graph, device, verbose=verbose)

    num_nodes = g["X"].shape[0]
    time_step = torch.tensor(g["time_step"], dtype=torch.long, device=device)
    train_mask = g["train_mask"]
    test_mask = torch.tensor(g["test_mask"])
    pre_mask = torch.tensor(g["pre_shutdown_mask"])
    post_mask = torch.tensor(g["post_shutdown_mask"])

    labeled_train_idx = np.sort(np.nonzero(train_mask & (g["labels"] >= 0))[0])
    sub_idx = subsample_labeled_indices(labeled_train_idx, args.frac, args.seed)
    train_idx = torch.tensor(sub_idx, dtype=torch.long, device=device)

    test_labeled = (test_mask & (labels_raw >= 0)).nonzero(as_tuple=True)[0].to(device)
    pre_labeled = (pre_mask & (labels_raw >= 0)).nonzero(as_tuple=True)[0].to(device)
    post_labeled = (post_mask & (labels_raw >= 0)).nonzero(as_tuple=True)[0].to(device)

    labels = labels_raw.clamp(min=0).to(device)

    if args.checkpoint is not None:
        encoder, hidden_dim = build_encoder_from_checkpoint(args.checkpoint, X.shape[1], device)
        ckpt_tag = os.path.splitext(os.path.basename(args.checkpoint))[0]
    else:
        hidden_dim = args.hidden_dim
        encoder = GCNEncoder(in_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
        ckpt_tag = "scratch"

    head = ClassifierHead(in_dim=hidden_dim, num_classes=2).to(device)
    model = ProbeFineTuneModel(encoder, head).to(device)

    # Extension B1: Initialize Teacher
    teacher_model = None
    if args.teacher_ckpt:
        teacher_model = GCNBaseline(in_dim=X.shape[1], hidden_dim=args.teacher_hidden_dim, num_classes=2).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device, weights_only=True))
        teacher_model.eval()
        if verbose: print(f"Loaded Teacher Model from {args.teacher_ckpt}")

    # Extension A2: Initialize Domain Discriminator
    domain_discriminator = None
    if args.use_domain_adv:
        domain_discriminator = DomainDiscriminator(in_dim=hidden_dim).to(device)
        if verbose: print("Enabled Domain-Adversarial Training (A2)")

    tag = f"{ckpt_tag}_{args.mode}_frac{args.frac:g}_seed{args.seed}"

    train(
        model, adj, X, labels, train_idx, device, mode=args.mode,
        epochs=args.epochs, head_lr=args.head_lr, encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay, class_weighted=not args.no_class_weight,
        verbose=verbose,
        teacher_model=teacher_model, distill_weight=args.distill_weight, distill_temp=args.distill_temp,
        domain_discriminator=domain_discriminator, domain_adv_weight=args.domain_adv_weight, time_step=time_step
    )

    test_metrics = evaluate(model, adj, X, labels, test_labeled)
    pre_metrics = evaluate(model, adj, X, labels, pre_labeled)
    post_metrics = evaluate(model, adj, X, labels, post_labeled)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {
        "tag": tag, "checkpoint": args.checkpoint, "mode": args.mode, "frac": args.frac,
        "test_35_49": test_metrics, "test_pre_shutdown_35_42": pre_metrics, "test_post_shutdown_43_49": post_metrics,
    }
    out_json = os.path.join(args.out_dir, f"step4_{tag}_metrics.json")
    with open(out_json, "w") as f: json.dump(results, f, indent=2)

    if verbose:
        print("\n=== Test metrics (time steps 35-49) ===")
        for k, v in test_metrics.items():
            print(f"  {k:20s}: {v}")
        print("\n=== Pre-shutdown (35-42) vs Post-shutdown (43-49) illicit-F1 ===")
        print(f"  pre : F1={pre_metrics['illicit_f1']:.4f}")
        print(f"  post: F1={post_metrics['illicit_f1']:.4f}")

    return results
def prepare_graph(graph_path, device, verbose=True):
    g = load_graph(graph_path)
    X = torch.tensor(g["X"], dtype=torch.float32, device=device)
    labels_raw = torch.tensor(g["labels"], dtype=torch.long)
    adj = build_normalized_adj(g["edge_index"], g["X"].shape[0]).to(device)
    return g, X, labels_raw, adj

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/elliptic_graph.npz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["probe", "finetune"], default="finetune")
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--head_lr", type=float, default=0.01)
    parser.add_argument("--encoder_lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--no_class_weight", action="store_true")
    parser.add_argument("--out_dir", default="outputs")

    # Extension B1 Args
    parser.add_argument("--teacher_ckpt", default=None, help="Path to Step 2 supervised baseline .pt")
    parser.add_argument("--teacher_hidden_dim", type=int, default=100, help="Hidden dim of Step 2 baseline")
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--distill_temp", type=float, default=2.0)

    # Extension A2 Args
    parser.add_argument("--use_domain_adv", action="store_true", help="Enable domain adversarial training")
    parser.add_argument("--domain_adv_weight", type=float, default=0.1)

    return parser

def main():
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args() # Modified line
    run_experiment(args)

if __name__ == "__main__":
    main()

!python step4_probe_finetune.py \
  --graph /content/drive/MyDrive/EllipticProject/elliptic_graph.npz \
  --checkpoint /content/drive/MyDrive/EllipticProject/step3_contrastive_trainonly.pt \
  --teacher_ckpt /content/drive/MyDrive/EllipticProject/step2_baseline_gcn.pt \
  --mode finetune \
  --frac 0.10 \
  --use_domain_adv \
  --out_dir /content/drive/MyDrive/EllipticProject/outputs    