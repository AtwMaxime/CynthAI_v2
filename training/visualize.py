"""
CynthAI_v2 Visualize — plot training metrics from a CSV log file.

Usage:
    python -m training.visualize --csv checkpoints/metrics.csv
    python -m training.visualize --csv checkpoints/metrics.csv --overlay --save plots/overview.png

The CSV is expected to have columns matching the self_play.py log:
    update, win_rate, policy, value, entropy, pred, total, lr

Options:
    --csv      path to metrics CSV (default: checkpoints/metrics.csv)
    --save     output image path (optional)
    --overlay  single comprehensive figure instead of 2×3 grid (store_true)
    --show     display interactive window (default: False when --save given)
"""

from __future__ import annotations

import csv
from pathlib import Path


def _read_csv(csv_path: str) -> dict[str, list[float]]:
    """Read a metrics CSV into a dict of column_name → list of float values."""
    data: dict[str, list[float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass
    return data


def _save_or_show(fig, save_path: str | None, show: bool) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to {save_path}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()


def plot_metrics(csv_path: str, save_path: str | None = None, show: bool = True) -> None:
    """
    2×3 grid of training metric plots.
    Panels: win_rate, policy loss, value loss, entropy, pred loss, total loss.
    """
    import matplotlib.pyplot as plt

    data    = _read_csv(csv_path)
    updates = data.get("update", list(range(len(next(iter(data.values()))))))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("CynthAI_v2 Training Metrics", fontsize=14)

    metrics = [
        ("win_rate",  "Win Rate (%)",  "Win Rate",   None,  None),
        ("policy",    "Loss",          "Policy",     None,  None),
        ("value",     "Loss",          "Value",      None,  None),
        ("entropy",   "Entropy",       "Entropy",    None,  None),
        ("pred",      "Loss",          "Pred",       None,  None),
        ("total",     "Loss",          "Total",      None,  None),
    ]

    for ax, (col, ylabel, title, ymin, ymax) in zip(axes.flat, metrics):
        if col not in data:
            ax.set_visible(False)
            continue
        vals = data[col]
        if col == "win_rate":
            vals = [v * 100 for v in vals]
        ax.plot(updates[:len(vals)], vals, linewidth=0.8)
        if col == "win_rate":
            ax.axhline(50, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ymin is not None or ymax is not None:
            ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Update")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def plot_overlay(csv_path: str, save_path: str | None = None, show: bool = True) -> None:
    """
    Single comprehensive figure with twinx axes:
      left  → win_rate (green #2ecc71)
      right → value loss (blue #3498db)
    Overlaid lines: policy (#e74c3c), pred (#9b59b6), entropy (#f39c12)
    """
    import matplotlib.pyplot as plt

    data    = _read_csv(csv_path)
    updates = data.get("update", list(range(len(next(iter(data.values()))))))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.suptitle("CynthAI_v2 Training Overview", fontsize=14)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    if "win_rate" in data:
        wr = [v * 100 for v in data["win_rate"]]
        ax1.plot(updates[:len(wr)], wr, color="#2ecc71", linewidth=1.2, label="Win Rate %")
        ax1.axhline(50, color="#2ecc71", linestyle="--", linewidth=0.6, alpha=0.5)
    if "value" in data:
        ax2.plot(updates[:len(data["value"])], data["value"], color="#3498db", linewidth=0.8, label="Value")
    if "policy" in data:
        ax2.plot(updates[:len(data["policy"])], data["policy"], color="#e74c3c", linewidth=0.8, label="Policy")
    if "pred" in data:
        ax2.plot(updates[:len(data["pred"])], data["pred"], color="#9b59b6", linewidth=0.8, label="Pred")
    if "entropy" in data:
        ax3.plot(updates[:len(data["entropy"])], data["entropy"], color="#f39c12", linewidth=0.8, label="Entropy")

    ax1.set_xlabel("Update")
    ax1.set_ylabel("Win Rate (%)", color="#2ecc71")
    ax2.set_ylabel("Loss", color="#3498db")
    ax3.set_ylabel("Entropy", color="#f39c12")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper left", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path, show)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="checkpoints/metrics.csv")
    parser.add_argument("--save",    default=None, help="output image path")
    parser.add_argument("--overlay", action="store_true", help="Single comprehensive figure")
    parser.add_argument("--show",    action="store_true", default=False)
    args = parser.parse_args()

    if args.overlay:
        plot_overlay(args.csv, save_path=args.save, show=args.show)
    else:
        plot_metrics(args.csv, save_path=args.save, show=args.show)
