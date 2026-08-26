%%writefile step6_temporal_robustness.py
"""
Step 6 — Temporal Robustness Evaluation (R6)
(Auto-loops through all available fractions in the Step 5 CSV)
"""
import argparse
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step5_csv", default="outputs/step5_label_efficiency_results.csv")
    parser.add_argument("--out_dir", default="outputs")
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.step5_csv):
        print(f"Error: Could not find {args.step5_csv}. Please run Step 5 first.")
        return

    df = pd.read_csv(args.step5_csv)
    fractions = sorted(df['frac'].unique(), reverse=True)

    os.makedirs(args.out_dir, exist_ok=True)

    for frac in fractions:
        df_frac = df[np.isclose(df['frac'], frac)]
        if df_frac.empty:
            continue

        rows = []
        for _, row in df_frac.iterrows():
            pre_f1 = row['pre_illicit_f1']
            post_f1 = row['post_illicit_f1']
            degradation_abs = pre_f1 - post_f1
            degradation_pct = degradation_abs / pre_f1 if pre_f1 > 0 else float('nan')

            rows.append({
                "model": row['arm'],
                "pre_illicit_f1": pre_f1,
                "post_illicit_f1": post_f1,
                "degradation_abs": degradation_abs,
                "degradation_pct": degradation_pct
            })

        print(f"\n=== Temporal Robustness Analysis (Fraction: {frac}) ===")
        print(f"{'Model':25s} {'Pre-F1':>8s} {'Post-F1':>8s} {'Degr.(abs)':>11s} {'Degr.(%)':>9s}")
        print("-" * 66)
        for r in rows:
            pct_str = f"{r['degradation_pct']*100:8.1f}%" if r['degradation_pct'] == r['degradation_pct'] else "     n/a"
            print(f"{r['model']:25s} {r['pre_illicit_f1']:8.4f} {r['post_illicit_f1']:8.4f} "
                  f"{r['degradation_abs']:11.4f} {pct_str}")

       
        csv_path = os.path.join(args.out_dir, f"step6_temporal_robustness_table_frac{frac}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

       
        models = [r["model"] for r in rows]
        pre_vals = [r["pre_illicit_f1"] for r in rows]
        post_vals = [r["post_illicit_f1"] for r in rows]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, pre_vals, width, label="Pre-shutdown (35-42)", color="#2a9d8f")
        ax.bar(x + width / 2, post_vals, width, label="Post-shutdown (43-49)", color="#e63946")

        ax.set_ylabel("Illicit-class F1")
        ax.set_title(f"Temporal Robustness: Pre- vs. Post-shutdown (Label Frac: {frac})")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

       
        plot_path = os.path.join(args.out_dir, f"step6_temporal_robustness_frac{frac}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print(f"\nAll tables and plots saved successfully to {args.out_dir}")

if __name__ == "__main__":
    main()


!python step6_temporal_robustness.py \
  --step5_csv /content/drive/MyDrive/EllipticProject/outputs/step5_label_efficiency_results.csv \
  --out_dir /content/drive/MyDrive/EllipticProject/outputs