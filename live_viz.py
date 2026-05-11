"""
Live training dashboard — polls metrics.csv every 5s and updates plots.

Usage:
    python live_viz.py
    python live_viz.py --dir checkpoints/curriculum_max_YYYYMMDD_HHMM
    python live_viz.py --interval 3    # faster refresh
"""

import os
os.environ["MPLBACKEND"] = "TkAgg"

import sys
import time
import csv
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
# Force TkAgg again after all imports that might override it
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
import numpy as np


# ── Helpers (copied from training/visualize to avoid importing its .use("Agg")) ──

def _read_csv(csv_path):
    data = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data.setdefault(k, []).append(float(v))
                except (ValueError, TypeError):
                    pass
    return data


def _running_mean(values, window):
    if window <= 1:
        return values
    arr = np.array(values)
    cum = np.cumsum(np.pad(arr, (window - 1, 0), mode="edge"))
    return (cum[window - 1:] - cum[:1 - window]) / window


def _ewma(values, alpha=0.3):
    smoothed = []
    prev = values[0]
    for v in values:
        prev = alpha * prev + (1 - alpha) * v
        smoothed.append(prev)
    return smoothed


def _find_most_recent_run(base_dir="checkpoints"):
    base = Path(base_dir)
    if not base.exists():
        return None
    # Find all subdirs with metrics.csv, sorted by modification time (newest first)
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "metrics.csv").exists()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(runs[0]) if runs else None


def _load_phase_config(csv_path):
    run_dir = Path(csv_path).parent
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


def _add_phase_spans(ax, breakpoints, values, total_updates, alpha=0.08, colormap="RdYlGn"):
    if not breakpoints or not values:
        return
    boundaries = [0] + list(breakpoints) + [total_updates]
    colors = plt.cm.get_cmap(colormap, len(values))
    for i in range(len(values)):
        ax.axvspan(boundaries[i], boundaries[i + 1],
                   alpha=alpha, color=colors(i / max(len(values) - 1, 1)),
                   zorder=-10)
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95,
                f"Phase {['I','II','III','IV','V'][i] if i < 5 else str(i)}",
                ha="center", va="top", fontsize=7, alpha=0.5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


# ── Live dashboard ─────────────────────────────────────────────────────────────

def live_dashboard(run_dir: str, interval: float = 5.0, smooth: int = 10):
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found")
        sys.exit(1)

    mask_bp, mask_vals, dense_bp, dense_vals = _load_phase_config(str(metrics_path))

    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.suptitle(f"Live Training Dashboard — {Path(run_dir).name}", fontsize=14, fontweight="bold")

    known_rows = 0
    last_check = 0.0
    first = True

    while plt.fignum_exists(fig.number):
        now = time.time()
        if now - last_check < interval:
            plt.pause(0.5)
            continue
        last_check = now

        data = _read_csv(str(metrics_path))
        eval_path = Path(run_dir) / "eval.csv"
        eval_data = _read_csv(str(eval_path)) if eval_path.exists() else {}
        updates = data.get("update", list(range(len(next(iter(data.values()))))))
        total_updates = int(data["update"][-1]) if "update" in data else len(updates)

        try:
            with open(metrics_path) as f:
                known_rows = sum(1 for _ in f) - 1
        except Exception:
            pass

        def _smooth(vals):
            if smooth > 1:
                return _running_mean(vals, smooth)
            return _ewma(vals, alpha=0.1) if len(vals) > 50 else vals

        for ax in axes.flat:
            ax.clear()

        # ── Panel 1: Win Rate ──────────────────────────────────────────────────
        ax = axes[0, 0]
        if "win_rate" in data:
            wr_raw = [v * 100 for v in data["win_rate"]]
            ax.plot(updates[:len(wr_raw)], wr_raw, color="#b2df8a", linewidth=0.5, alpha=0.5, label="raw")
            wr_smooth = _smooth(wr_raw)
            ax.plot(updates[:len(wr_smooth)], wr_smooth, color="#33a02c", linewidth=1.5, label="smoothed")
            ax.axhline(50, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            for label, color, marker in [("random_wr", "#1f78b4", "o"), ("fulloff_wr", "#e31a1c", "s"), ("pool_wr", "#ff7f00", "^")]:
                if label in eval_data:
                    ax.scatter(eval_data["update"], [v * 100 for v in eval_data[label]],
                               color=color, marker=marker, s=20, zorder=5, alpha=0.8)
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

        # ── Panel 6: Curriculum overview ───────────────────────────────────────
        ax = axes[2, 1]
        if "lr" in data:
            lr_vals = [v * 1e5 for v in data["lr"]]
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
        fig.canvas.draw()
        fig.canvas.flush_events()

        if first:
            print(f"Dashboard live! Watching {run_dir}")
            first = False

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CynthAI_v2 Live Dashboard")
    parser.add_argument("--dir", default="", help="Run directory (auto-detect)")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval (seconds)")
    parser.add_argument("--smooth", type=int, default=10, help="Running mean window")
    args = parser.parse_args()

    run_dir = args.dir or _find_most_recent_run()
    if not run_dir:
        print("No run directory found. Use --dir to specify one.")
        sys.exit(1)

    print(f"Watching: {run_dir}")
    print(f"Refresh: every {args.interval}s  |  Smooth: window={args.smooth}")
    print("Close the matplotlib window to stop.")
    live_dashboard(run_dir, interval=args.interval, smooth=args.smooth)