
# python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def read_max_f1(csv_path: Path):
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            max_f1 =  0.0
            selected_mean = None
            selected_p25 = None
            selected_p975 = None
            for row in reader:
                raw = row.get("MAX")
                if raw is None: continue
                val_max = float(raw)
                val_mean = float(row.get("Mean", 0.0))
                val_p25  = float(row.get("Percentile_2.5", 0.0))
                val_p975 = float(row.get("Percentile_97.5", 0.0))
                if val_max > max_f1:
                    max_f1 = val_max
                    selected_mean = val_mean
                    selected_p25 = val_p25
                    selected_p975 = val_p975
            return max_f1, selected_mean, selected_p25, selected_p975
    except FileNotFoundError:
        print(f"ADVERTENCIA: No existe {csv_path!s}. Se omite.", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gráfica del F1 máximo por iteración (N=0..8).")
    parser.add_argument(
        "--modelname",
        type=Path,
        default="QWEN",
        help="Directorio con archivos scores_iterN.csv. Por defecto: 'rejectionSampling/QWEN/train'.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default="",
        help="Directorio raíz del proyecto.",
    )
    parser.add_argument(
        "--legend",
        default=False,
        action="store_true",
        help="Incluir leyenda en la gráfica.",
    )


    args = parser.parse_args()
    path = args.path
    modelname = args.modelname
    legend = args.legend

    train_dir = f"{path}/rejectionSampling/{modelname}/train"


    xs_train = []
    ys_max, ys_mean, ys_p25, ys_p975 = [], [], [], []
    max_n = 0
    for f in Path(train_dir).glob("scores_iter*.csv"):
        n = int(f.stem.replace("scores_iter", ""))
        max_n = max(max_n, n)
    for n in range(0, max_n + 1):
        csv_train = Path(train_dir) / f"scores_iter{n}.csv"
        result = read_max_f1(csv_train)
        if result is not None:
            max_f1, mean_f1, p25_f1, p975_f1 = result
            xs_train.append(n)
            ys_max.append(max_f1)
            ys_mean.append(mean_f1)
            ys_p25.append(p25_f1)
            ys_p975.append(p975_f1)

    if not xs_train:
        print("ERROR: No se encontraron datos para graficar.", file=sys.stderr)
        sys.exit(2)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 17
    })
    plt.figure(figsize=(9.0, 5.0), dpi=300)
    plt.plot(xs_train, ys_max, marker="o", linewidth=2, color="#1f77b4", label="Max F1")
    plt.plot(xs_train, ys_mean, marker="s", linewidth=2, color="#2ca02c", label="Mean F1")
    plt.fill_between(
        xs_train,
        ys_p25,
        ys_p975,
        alpha=0.2, color="#2ca02c", label="95% CI")
    #plt.title("F1 per iteration (Train)")
    plt.ylabel("F1")
    plt.ylim(15.0,80.0)
    plt.xlabel("Iteration")
    plt.xticks(xs_train)
    if legend:
        plt.legend(fancybox=True)
    plt.tight_layout()
    plt.savefig(f"irudiak/iterazioak/iteration_max_f1_{modelname}.pdf")
    plt.savefig(f"irudiak/iterazioak/iteration_max_f1_{modelname}.png")


if __name__ == "__main__":
    main()
