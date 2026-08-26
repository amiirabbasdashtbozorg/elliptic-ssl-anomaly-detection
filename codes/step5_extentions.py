%%writefile step5_label_efficiency_sweep.py
"""
Step 5 — Label-Efficiency Evaluation (R5)
+ EXTENSIONS A2 & B1 Integration
"""
import argparse
import csv
import os
import matplotlib.pyplot as plt
import torch

from step4_probe_finetune import build_arg_parser, run_experiment, prepare_graph

def make_args(base_defaults, **overrides):
    args = argparse.Namespace(**vars(base_defaults))
    for k, v in overrides.items():
        setattr(args, k, v)
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="outputs/elliptic_graph.npz")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.10])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--teacher_ckpt", default=None, help="Add B1 extension arm if provided")
    parser.add_argument("--use_domain_adv", action="store_true", help="Add A2 extension arm if flag is used")

    sweep_args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    g, X, labels_raw, adj = prepare_graph(sweep_args.graph, device)

    step4_defaults, _ = build_arg_parser().parse_known_args([])
    step4_defaults.graph = sweep_args.graph
    step4_defaults.seed = sweep_args.seed
    step4_defaults.epochs = sweep_args.epochs
    step4_defaults.out_dir = sweep_args.out_dir

    arms = [
        ("baseline", None, "finetune", {}),
        ("ssl_finetune", sweep_args.checkpoint, "finetune", {}),
    ]

    if sweep_args.use_domain_adv:
        arms.append(("ssl_finetune_A2", sweep_args.checkpoint, "finetune", {"use_domain_adv": True}))
    if sweep_args.teacher_ckpt:
        arms.append(("ssl_finetune_B1", sweep_args.checkpoint, "finetune", {"teacher_ckpt": sweep_args.teacher_ckpt}))
    if sweep_args.use_domain_adv and sweep_args.teacher_ckpt:
        arms.append(("ssl_finetune_A2_B1", sweep_args.checkpoint, "finetune", {"use_domain_adv": True, "teacher_ckpt": sweep_args.teacher_ckpt}))

    rows = []
    total_runs = len(arms) * len(sweep_args.fractions)
    run_num = 0

    for frac in sweep_args.fractions:
        for arm_name, checkpoint, mode, extra_kwargs in arms:
            run_num += 1
            print(f"\n{'='*70}\n[{run_num}/{total_runs}] arm={arm_name}  frac={frac}  mode={mode}\n{'='*70}")

            run_args = make_args(step4_defaults, checkpoint=checkpoint, mode=mode, frac=frac, **extra_kwargs)
            results = run_experiment(run_args, device=device, g=g, X=X, labels_raw=labels_raw, adj=adj, verbose=False)

            test_m = results["test_35_49"]
            pre_m = results["test_pre_shutdown_35_42"]
            post_m = results["test_post_shutdown_43_49"]

            print(f"  -> test_F1: {test_m['illicit_f1']:.4f} | pre_F1: {pre_m['illicit_f1']:.4f} | post_F1: {post_m['illicit_f1']:.4f}")

            rows.append({
                "arm": arm_name,
                "frac": frac,
                "n_train_labeled_used": results.get("n_train_labeled_used", 0),
                "test_illicit_f1": test_m["illicit_f1"],
                "test_illicit_precision": test_m["illicit_precision"],
                "test_illicit_recall": test_m["illicit_recall"],
                "test_auc_pr": test_m["auc_pr"],
                "pre_illicit_f1": pre_m["illicit_f1"],
                "post_illicit_f1": post_m["illicit_f1"],
            })

    os.makedirs(sweep_args.out_dir, exist_ok=True)
    csv_path = os.path.join(sweep_args.out_dir, "step5_label_efficiency_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_path = os.path.join(sweep_args.out_dir, "step5_label_efficiency_curve.png")
    plot_label_efficiency(rows, sweep_args.fractions, plot_path)
    print(f"\nSaved headline plot -> {plot_path}")

def plot_label_efficiency(rows, fractions, out_path):
    arm_order = []
    for r in rows:
        if r["arm"] not in arm_order:
            arm_order.append(r["arm"])

    style = {
        "baseline": dict(color="#6c757d", linestyle="--", marker="o", label="Supervised baseline"),
        "ssl_finetune": dict(color="#e63946", linestyle="-", marker="^", label="SSL: fine-tuned"),
        "ssl_finetune_A2": dict(color="#2a9d8f", linestyle="-", marker="s", label="SSL + A2 (Domain Adv)"),
        "ssl_finetune_B1": dict(color="#f4a261", linestyle="-", marker="D", label="SSL + B1 (Distillation)"),
        "ssl_finetune_A2_B1": dict(color="#264653", linestyle="-", marker="v", label="SSL + A2 + B1"),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for arm in arm_order:
        arm_rows = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["frac"])
        x = [r["frac"] * 100 for r in arm_rows]
        y = [r["test_illicit_f1"] for r in arm_rows]
        kwargs = style.get(arm, dict(marker="o", label=arm))
        ax.plot(x, y, **kwargs)

    ax.set_xscale("log")
    ax.set_xlabel("Labeled data used (%)")
    ax.set_ylabel("Illicit-class F1 (test, time steps 35-49)")
    ax.set_title("Label efficiency: illicit-F1 vs. fraction of labels used")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()

!python step5_label_efficiency_sweep.py \
  --graph /content/drive/MyDrive/EllipticProject/elliptic_graph.npz \
  --checkpoint /content/drive/MyDrive/EllipticProject/step3_contrastive_trainonly.pt \
  --teacher_ckpt /content/drive/MyDrive/EllipticProject/step2_baseline_gcn.pt \
  --use_domain_adv \
  --out_dir /content/drive/MyDrive/EllipticProject/outputs