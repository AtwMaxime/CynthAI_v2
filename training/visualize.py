"""
CynthAI_v2 Visualize — training metrics + eval plots from CSV logs.

Usage:
    python -m training.visualize --dir checkpoints/curriculum_max_20260510_1234
    python -m training.visualize --dir checkpoints/curriculum_max_20260510_1234 --save plots/overview.png
    python -m training.visualize --dir checkpoints/curriculum_max_20260510_1234 --smooth 20

Without --dir, scans checkpoints/ for the most recent run.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv(csv_path: str) -> dict[str, list[float]]:
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


def _ewma(values: list[float], alpha: float = 0.3) -> list[float]:
    """Exponentially weighted moving average. alpha=1 → no smoothing."""
    smoothed = []
    prev = values[0]
    for v in values:
        prev = alpha * prev + (1 - alpha) * v
        smoothed.append(prev)
    return smoothed


def _running_mean(values: list[float], window: int) -> list[float]:
    """Simple running mean with given window size."""
    if window <= 1:
        return values
    arr = np.array(values)
    cum = np.cumsum(np.pad(arr, (window - 1, 0), mode="edge"))
    return (cum[window - 1:] - cum[:1 - window]) / window


def _find_most_recent_run(base_dir: str = "checkpoints") -> str | None:
    runs = sorted(Path(base_dir).iterdir()) if Path(base_dir).exists() else []
    # Filter directories that look like run folders (contain metrics.csv)
    runs = [r for r in runs if r.is_dir() and (r / "metrics.csv").exists()]
    return str(runs[-1]) if runs else None


def _load_phase_config(csv_path: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Try to load P1/P2 phase config from the run dir (saved in checkpoint config).
    Returns (mask_breakpoints, mask_values, dense_breakpoints, dense_values).
    """
    run_dir = Path(csv_path).parent
    # Try to read config from the first checkpoint .pt
    ckpts = sorted(run_dir.glob("agent_*.pt"))
    if ckpts:
        try:
            import torch
            ckpt = torch.load(ckpts[0], map_location="cpu", weights_only=True)
            cfg = ckpt.get("config", {})
            return (
                list(cfg.get("mask_phase_breakpoints", ())),
                list(cfg.get("mask_phase_values", ())),
                list(cfg.get("dense_phase_breakpoints", ())),
                list(cfg.get("dense_phase_values", ())),
            )
        except Exception:
            pass
    return [], [], [], []


def _add_phase_spans(ax, breakpoints, values, total_updates: int, alpha=0.08, colormap="RdYlGn"):
    """Add vertical shaded regions for curriculum phases."""
    if not breakpoints or not values:
        return
    boundaries = [0] + list(breakpoints) + [total_updates]
    colors = plt.cm.get_cmap(colormap, len(values))
    for i in range(len(values)):
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   alpha=alpha, color=colors(i / max(len(values) - 1, 1)),
                   zorder=-10)
        # Phase label at midpoint
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95,
                f"Phase {['I','II','III','IV','V'][i] if i < 5 else str(i)}",
                ha="center", va="top", fontsize=7, alpha=0.5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


# ── Main plot function ────────────────────────────────────────────────────────

def plot_all(
    run_dir: str,
    save_path: str | None = None,
    show: bool = False,
    smooth: int = 0,
    figsize: tuple = (16, 10),
) -> None:
    """
    Full training dashboard: 6-panel figure with curriculum phase overlays.

    Panels:
      1. Win Rate (raw + smoothed + eval points)
      2. Policy / Value / Pred loss
      3. Entropy
      4. Total loss
      5. Grad norm / Clip frac / Explained variance
      6. LR + mask_ratio + dense_scale (curriculum overview)
    """
    metrics_path = Path(run_dir) / "metrics.csv"
    eval_path    = Path(run_dir) / "eval.csv"

    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found")
        return

    data = _read_csv(str(metrics_path))
    eval_data = _read_csv(str(eval_path)) if eval_path.exists() else {}
    updates = data.get("update", list(range(len(next(iter(data.values()))))))

    # Phase config from checkpoint
    mask_bp, mask_vals, dense_bp, dense_vals = _load_phase_config(str(metrics_path))
    total_updates = int(data["update"][-1]) if "update" in data else len(updates)

    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f"Training Dashboard — {Path(run_dir).name}", fontsize=14, fontweight="bold")

    # Utility: apply smoothing
    def _smooth(vals):
        if smooth > 1:
            return _running_mean(vals, smooth)
        return _ewma(vals, alpha=0.1) if len(vals) > 50 else vals

    # ── Panel 1: Win Rate ──────────────────────────────────────────────────
    ax = axes[0, 0]
    if "win_rate" in data:
        wr_raw = [v * 100 for v in data["win_rate"]]
        ax.plot(updates[:len(wr_raw)], wr_raw, color="#b2df8a", linewidth=0.5, alpha=0.5, label="raw")
        wr_smooth = _smooth(wr_raw)
        ax.plot(updates[:len(wr_smooth)], wr_smooth, color="#33a02c", linewidth=1.5, label=f"smoothed")
        ax.axhline(50, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        # Evaluation overlay
        for label, color, marker in [("random_wr", "#1f78b4", "o"), ("fulloff_wr", "#e31a1c", "s"), ("pool_wr", "#ff7f00", "^")]:
            if label in eval_data:
                ax.scatter(eval_data["update"], [v * 100 for v in eval_data[label]],
                           color=color, marker=marker, s=20, zorder=5, alpha=0.8,
                           label={"_wr": "", "random_wr": "vs Random", "fulloff_wr": "vs FullOff", "pool_wr": "vs Pool"}.get(label, label))
        _add_phase_spans(ax, mask_bp, mask_vals, total_updates)
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Policy / Value / Pred ─────────────────────────────────────
    ax = axes[0, 1]
    for col, color, label in [("policy", "#e74c3c", "Policy"), ("value", "#3498db", "Value"),
                               ("pred", "#9b59b6", "Pred")]:
        if col in data:
            vals = _smooth(data[col])
            ax.plot(updates[:len(vals)], vals, color=color, linewidth=1.0, label=label)
    _add_phase_spans(ax, mask_bp, mask_vals, total_updates)
    ax.set_ylabel("Loss")
    ax.set_title("Policy / Value / Pred")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Entropy ───────────────────────────────────────────────────
    ax = axes[1, 0]
    if "entropy" in data:
        entropy_raw = data["entropy"]
        ax.plot(updates[:len(entropy_raw)], entropy_raw, color="#f39c12", linewidth=0.5, alpha=0.4, label="raw")
        if len(entropy_raw) > 10:
            ax.plot(updates[:len(entropy_raw)], _smooth(entropy_raw), color="#e67e22", linewidth=1.5, label="smoothed")
    _add_phase_spans(ax, mask_bp, mask_vals, total_updates)
    ax.set_ylabel("Entropy")
    ax.set_title("Policy Entropy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Total Loss ────────────────────────────────────────────────
    ax = axes[1, 1]
    if "total" in data:
        total_raw = data["total"]
        ax.plot(updates[:len(total_raw)], total_raw, color="#7f8c8d", linewidth=0.5, alpha=0.4, label="raw")
        if len(total_raw) > 10:
            ax.plot(updates[:len(total_raw)], _smooth(total_raw), color="#2c3e50", linewidth=1.5, label="smoothed")
    _add_phase_spans(ax, mask_bp, mask_vals, total_updates)
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Grad norm / Clip frac / Explained variance ────────────────
    ax = axes[2, 0]
    for col, color, label in [("grad_norm", "#e74c3c", "Grad Norm"),
                               ("clip_frac", "#3498db", "Clip Frac"),
                               ("explained_variance", "#2ecc71", "Expl. Var.")]:
        if col in data:
            vals = _smooth(data[col]) if len(data[col]) > 10 else data[col]
            ax.plot(updates[:len(vals)], vals, color=color, linewidth=1.0, label=label)
    _add_phase_spans(ax, mask_bp, mask_vals, total_updates)
    ax.set_ylabel("Magnitude")
    ax.set_title("Training Signals")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Curriculum overview (LR + mask_ratio + dense_scale) ───────
    ax = axes[2, 1]
    if "lr" in data:
        lr_vals = [v * 1e5 for v in data["lr"]]  # scale for readability
        ax.plot(updates[:len(lr_vals)], lr_vals, color="#e74c3c", linewidth=1.2, label="LR (×1e-5)")
    if "mask_ratio" in data:
        ax.plot(updates[:len(data["mask_ratio"])], data["mask_ratio"],
                color="#8e44ad", linewidth=1.5, label="Mask Ratio", linestyle="--")
    if "dense_scale" in data:
        ax.plot(updates[:len(data["dense_scale"])], data["dense_scale"],
                color="#2980b9", linewidth=1.5, label="Dense Scale", linestyle=":")
    _add_phase_spans(ax, mask_bp, mask_vals, total_updates)
    ax.set_ylabel("Value")
    ax.set_title("Curriculum Schedule")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def _save_or_show(fig, save_path: str | None, show: bool) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 Training Visualizer")
    parser.add_argument("--dir", default="", help="Run directory (auto-detect if empty)")
    parser.add_argument("--save", default=None, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    parser.add_argument("--smooth", type=int, default=0, help="Running mean window (0 = EWMA)")
    args = parser.parse_args()

    run_dir = args.dir or _find_most_recent_run()
    if not run_dir:
        print("No run directory found. Use --dir to specify one.")
        exit(1)

    plot_all(run_dir, save_path=args.save, show=args.show, smooth=args.smooth)