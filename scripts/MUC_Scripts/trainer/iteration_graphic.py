
# python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Se requiere 'matplotlib'. Instale con: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_train_dir() -> Path:
    return default_repo_root() / "rejectionSampling" / "QWEN" / "train"


def default_dev_dir() -> Path:
    return default_repo_root() / "rejectionSampling" / "QWEN" / "dev"


def _find_f1_header(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    for name in fieldnames:
        if name == "F1":
            return name
    for name in fieldnames:
        if name.lower() == "f1":
            return name
    return None


def read_max_f1(csv_path: Path) -> float | None:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            f1_col = _find_f1_header(reader.fieldnames)
            if not f1_col:
                print(f"ADVERTENCIA: No se encontró columna 'F1' en {csv_path!s}.", file=sys.stderr)
                return None

            max_f1: float | None = None
            for row in reader:
                raw = row.get(f1_col)
                if not raw:
                    continue
                val_s = raw.strip().replace(",", ".")
                try:
                    val = float(val_s)
                except ValueError:
                    continue
                if max_f1 is None or val > max_f1:
                    max_f1 = val
            return max_f1
    except FileNotFoundError:
        print(f"ADVERTENCIA: No existe {csv_path!s}. Se omite.", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Gráfica del F1 máximo por iteración (N=0..8).")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=default_train_dir(),
        help="Directorio con archivos scores_iterN.csv. Por defecto: 'rejectionSampling/QWEN/train'.",
    )
    parser.add_argument(
        "--dev-dir",
        type=Path,
        default=default_dev_dir(),
        help="Directorio con archivos N.csv de dev. Por defecto: 'rejectionSampling/QWEN/dev'.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra la gráfica en pantalla además de guardarla.",
    )
    args = parser.parse_args()

    out_path = "irudiak/iteration_max_f1_QWEN.png"

    xs_train, ys_train = [], []
    xs_dev, ys_dev = [], []

    for n in range(0, 15):
        csv_train = args.train_dir / f"scores_iter{n}.csv"
        max_f1_train = read_max_f1(csv_train)
        if max_f1_train is not None:
            xs_train.append(n)
            ys_train.append(max_f1_train)

        csv_dev = args.dev_dir / f"{n}.csv"
        max_f1_dev = read_max_f1(csv_dev)
        if max_f1_dev is not None:
            xs_dev.append(n)
            ys_dev.append(max_f1_dev)

    if not xs_train and not xs_dev:
        print("ERROR: No se encontraron datos para graficar.", file=sys.stderr)
        sys.exit(2)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(8, 4.5), dpi=120)
    if xs_train:
        plt.plot(xs_train, ys_train, marker="o", linewidth=2, color="#1f77b4", label="Train")
    if xs_dev:
        plt.plot(xs_dev, ys_dev, marker="s", linewidth=2, color="#d62728", label="Dev")
    plt.title("F1 máximo por iteración")
    plt.xlabel("Iteración (N)")
    plt.ylabel("F1 máximo")
    plt.xticks(sorted(set(xs_train + xs_dev)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    if args.show:
        plt.show()
    print(f"Gráfica guardada en: {out_path}")


if __name__ == "__main__":
    main()
