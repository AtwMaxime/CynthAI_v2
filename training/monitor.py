"""
CynthAI_v2 Monitoring — diagnostic plots from evaluation data.

Generated at each checkpoint alongside attention maps:
  1. Action distribution histograms per opponent type
  2. Battle length distribution per opponent type
  3. Reward decomposition (stacked bar, per opponent)
  4. Value function calibration (predicted V(s) vs actual return)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Action labels ──────────────────────────────────────────────────────────────

ACTION_LABELS = [
    "Move 1", "Move 2", "Move 3", "Move 4",
    "Tera 1", "Tera 2", "Tera 3", "Tera 4",
    "Switch 1", "Switch 2", "Switch 3", "Switch 4", "Switch 5",
]

ACTION_SHORT = ["M1", "M2", "M3", "M4", "T1", "T2", "T3", "T4",
                "S1", "S2", "S3", "S4", "S5"]


def _save_fig(fig, path_stem: str):
    """Save figure as both PNG and PDF."""
    for ext in [".png", ".pdf"]:
        fig.savefig(f"{path_stem}{ext}", dpi=150, bbox_inches="tight")


# ── Main entry point ──────────────────────────────────────────────────────────

def save_eval_plots(
    eval_results: dict[str, dict],
    save_dir: str,
    tag: str = "",
) -> None:
    """Generate all monitoring plots from evaluation data and save as PNGs.

    Saves into subdirectories:
        eval_data/   — JSON raw data
        actions/     — action distribution histograms
        battle_len/  — battle length histograms
        reward/      — reward decomposition
        value_calib/ — value function calibration
        cross_attn/  — cross-attention heatmaps (if captured)
    """
    base = Path(save_dir)

    subdirs = {
        "data":        base / "eval_data",
        "actions":     base / "actions",
        "battle_len":  base / "battle_len",
        "reward":      base / "reward",
        "value_calib": base / "value_calib",
        "cross_attn":  base / "cross_attn",
        "cos_sim":     base / "cos_sim",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)

    _save_eval_json(eval_results, subdirs["data"], tag)
    _plot_action_hist(eval_results, subdirs["actions"], tag)
    _plot_battle_length(eval_results, subdirs["battle_len"], tag)
    _plot_reward_decomp(eval_results, subdirs["reward"], tag)
    _plot_value_calib(eval_results, subdirs["value_calib"], tag)
    _plot_cross_attention(eval_results, subdirs["cross_attn"], tag)
    _plot_cos_sim(eval_results, subdirs["cos_sim"], tag)

    # Accuracy-over-time plot from metrics.csv (if available)
    _plot_prediction_accuracy(str(base / "metrics.csv"), subdirs["data"], tag)

    print(f"  -> eval plots saved to {base}/  [{tag}]")


def _save_eval_json(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """Save raw evaluation data as JSON for later analysis."""
    serializable = {}
    for opp, data in eval_results.items():
        entry = {}
        for k, v in data.items():
            if isinstance(v, (list, dict, float, int)):
                entry[k] = v
            elif isinstance(v, np.integer):
                entry[k] = int(v)
            elif isinstance(v, np.floating):
                entry[k] = float(v)
            else:
                continue  # skip non-serializable (tensors, etc.)
        serializable[opp] = entry

    with open(save_path / f"eval_data_{tag}.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)


# ── Panel 1: Action distribution histogram ────────────────────────────────────

def _plot_action_hist(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """One histogram per opponent type, or grouped bar chart."""
    opp_labels = list(eval_results.keys())
    """One histogram per opponent type, or grouped bar chart."""
    opp_labels = list(eval_results.keys())
    n_opp = len(opp_labels)

    has_data = any(
        len(eval_results[opp].get("action_histogram", [])) == 13
        for opp in opp_labels
    )
    if not has_data:
        return

    fig, axes = plt.subplots(1, n_opp, figsize=(5 * n_opp, 4), squeeze=False)
    fig.suptitle(f"Action Distribution — {tag}", fontsize=12, fontweight="bold")

    for idx, opp in enumerate(opp_labels):
        ax = axes[0, idx]
        hist = eval_results[opp].get("action_histogram", [0] * 13)
        colors = ["#2ecc71"] * 4 + ["#f39c12"] * 4 + ["#3498db"] * 5
        ax.bar(range(13), hist, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(13))
        ax.set_xticklabels(ACTION_SHORT, rotation=45, fontsize=7)
        ax.set_title(f"vs {opp}", fontsize=10)
        ax.set_ylabel("Count")
        # Add value labels on top of bars
        for i, v in enumerate(hist):
            if v > 0:
                ax.text(i, v + max(hist) * 0.02, str(v),
                        ha="center", va="bottom", fontsize=6, alpha=0.7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_fig(fig, save_path / tag)
    plt.close(fig)


# ── Panel 2: Battle length distribution ───────────────────────────────────────

def _plot_battle_length(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """Histogram of episode lengths per opponent type."""
    opp_labels = list(eval_results.keys())
    has_data = any(
        len(eval_results[opp].get("battle_lengths", [])) > 0
        for opp in opp_labels
    )
    if not has_data:
        return

    fig, axes = plt.subplots(1, len(opp_labels),
                             figsize=(5 * len(opp_labels), 4), squeeze=False)
    fig.suptitle(f"Battle Length Distribution — {tag}", fontsize=12, fontweight="bold")

    for idx, opp in enumerate(opp_labels):
        ax = axes[0, idx]
        lengths = eval_results[opp].get("battle_lengths", [])
        if lengths:
            ax.hist(lengths, bins=min(30, max(lengths) - min(lengths) + 1),
                    color="#3498db", edgecolor="white", alpha=0.8)
            mean_len = np.mean(lengths)
            median_len = np.median(lengths)
            ax.axvline(mean_len, color="#e74c3c", linestyle="--",
                       linewidth=1.5, label=f"Mean={mean_len:.0f}")
            ax.axvline(median_len, color="#f39c12", linestyle=":",
                       linewidth=1.5, label=f"Median={median_len:.0f}")
            ax.legend(fontsize=7)
        ax.set_title(f"vs {opp} (n={len(lengths)})", fontsize=10)
        ax.set_xlabel("Battle Length (steps)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_path / tag)
    plt.close(fig)


# ── Panel 3: Reward decomposition ─────────────────────────────────────────────

REWARD_COLORS = {"ko_opp": "#2ecc71", "ko_own": "#e74c3c",
                 "hp_adv": "#3498db", "count_adv": "#9b59b6",
                 "status": "#e67e22", "hazard": "#1abc9c",
                 "hazard_remove": "#8e44ad",
                 "terminal": "#f39c12"}
REWARD_LABELS = {"ko_opp": "KO opp", "ko_own": "KO own",
                 "hp_adv": "HP adv", "count_adv": "Count adv",
                 "status": "Status", "hazard": "Hazard",
                 "hazard_remove": "Haz remove",
                 "terminal": "Terminal"}


def _plot_reward_decomp(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """Stacked bar chart: average reward per component per opponent type."""
    opp_labels = list(eval_results.keys())
    has_data = any(
        eval_results[opp].get("reward_decomp_avg", {})
        for opp in opp_labels
    )
    if not has_data:
        return

    components = ["ko_opp", "ko_own", "hp_adv", "count_adv", "status", "hazard", "hazard_remove", "terminal"]
    n_opp = len(opp_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Reward Decomposition — {tag}", fontsize=12, fontweight="bold")

    # Left: stacked bar per opponent
    x = np.arange(n_opp)
    bottom = np.zeros(n_opp)
    for comp in components:
        vals = [
            eval_results[opp].get("reward_decomp_avg", {}).get(comp, 0.0)
            for opp in opp_labels
        ]
        ax1.bar(x, vals, bottom=bottom, color=REWARD_COLORS[comp],
                label=REWARD_LABELS[comp], edgecolor="white", width=0.5)
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(opp_labels, fontsize=9)
    ax1.set_ylabel("Avg Reward per Step")
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: proportion bar (stacked to 100%)
    bottom = np.zeros(n_opp)
    for comp in components:
        vals = np.array([
            abs(eval_results[opp].get("reward_decomp_avg", {}).get(comp, 0.0))
            for opp in opp_labels
        ])
        total = vals.sum(axis=0)
        props = np.divide(vals, total, out=np.zeros_like(vals),
                          where=total > 0)
        ax2.bar(x, props, bottom=bottom, color=REWARD_COLORS[comp],
                label=REWARD_LABELS[comp], edgecolor="white", width=0.5)
        bottom += props

    ax2.set_xticks(x)
    ax2.set_xticklabels(opp_labels, fontsize=9)
    ax2.set_ylabel("Proportion")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_fig(fig, save_path / tag)
    plt.close(fig)


# ── Panel 4: Value calibration ────────────────────────────────────────────────

def _plot_value_calib(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """Scatter plot of predicted V(s) vs actual return, per opponent."""
    opp_labels = list(eval_results.keys())
    has_data = any(
        len(eval_results[opp].get("value_preds", [])) > 0
        for opp in opp_labels
    )
    if not has_data:
        return

    fig, axes = plt.subplots(1, len(opp_labels),
                             figsize=(5 * len(opp_labels), 4.5), squeeze=False)
    fig.suptitle(f"Value Function Calibration — {tag}", fontsize=12, fontweight="bold")

    for idx, opp in enumerate(opp_labels):
        ax = axes[0, idx]
        preds = eval_results[opp].get("value_preds", [])
        returns = eval_results[opp].get("value_returns", [])

        if preds and returns:
            ax.scatter(preds, returns, s=3, alpha=0.4, color="#3498db", edgecolors="none")

            # Perfect calibration line
            all_vals = preds + returns
            lo, hi = min(all_vals), max(all_vals)
            margin = (hi - lo) * 0.1 or 0.1
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    color="#e74c3c", linestyle="--", linewidth=1, alpha=0.6,
                    label="Perfect")

            # Linear fit
            if len(preds) > 5:
                try:
                    coeffs = np.polyfit(preds, returns, 1)
                    fit_line = np.polyval(coeffs, sorted(preds))
                    ax.plot(sorted(preds), fit_line, color="#2ecc71",
                            linewidth=1.5, alpha=0.8,
                            label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
                except np.linalg.LinAlgError:
                    pass

            # Correlation
            corr = np.corrcoef(preds, returns)[0, 1]
            ax.set_title(f"vs {opp}  (r={corr:.3f})", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
        else:
            ax.set_title(f"vs {opp} (no data)", fontsize=10)

        ax.set_xlabel("Predicted V(s)")
        ax.set_ylabel("Actual Return")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, save_path / tag)
    plt.close(fig)


# ── Cross-attention heatmap ──────────────────────────────────────────

_STATE_LABELS = [
    "OWN0", "OWN1", "OWN2", "OWN3", "OWN4", "OWN5",
    "OPP0", "OPP1", "OPP2", "OPP3", "OPP4", "OPP5",
    "FIELD",
]


def _plot_cross_attention(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """
    Generate cross-attention heatmaps for opponents that have cross_attn_stats.

    Shows average attention (over all 4 heads) from 13 action queries (y-axis)
    to 13 state tokens (x-axis). Brighter = more attention.
    """
    opp_labels = [k for k, v in eval_results.items()
                  if "cross_attn_stats" in v]
    if not opp_labels:
        return

    n_opp = len(opp_labels)
    fig, axes = plt.subplots(1, n_opp, figsize=(6 * n_opp, 5), squeeze=False)
    fig.suptitle(f"Cross-attention: Action Queries → State Tokens — {tag}",
                 fontsize=12, fontweight="bold")

    for idx, opp in enumerate(opp_labels):
        ax = axes[0, idx]
        stats = eval_results[opp]["cross_attn_stats"]
        mean_w = stats["mean"]  # [H, 13, 13]
        # Average over heads
        avg_w = mean_w.mean(dim=0).numpy()  # [13, 13]

        im = ax.imshow(avg_w, cmap="YlOrRd", aspect="auto", vmin=0, vmax=avg_w.max())

        ax.set_xticks(range(13))
        ax.set_xticklabels(_STATE_LABELS, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(13))
        ax.set_yticklabels(ACTION_SHORT, fontsize=7)

        ax.set_xlabel("State tokens (keys)")
        ax.set_ylabel("Action queries")
        ax.set_title(f"vs {opp}  (n={stats['n']})", fontsize=10)

        # Annotate top-3 attended tokens per action group
        for row in range(13):
            top3 = avg_w[row].argsort()[-3:][::-1]
            label = ", ".join(_STATE_LABELS[c] for c in top3)
            ax.text(12.5, row, f"  {label}", fontsize=5, va="center", ha="left",
                    color="gray")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    _save_fig(fig, save_path / tag)
    plt.close(fig)


def _plot_prediction_accuracy(
    metrics_path: str,
    save_path: Path,
    tag: str,
) -> None:
    """
    Plot prediction head accuracy over training updates from metrics.csv.
    Generated at each eval step alongside other eval plots.
    """
    import csv
    try:
        with open(metrics_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get("item_acc")]
    except (FileNotFoundError, IOError):
        return
    if not rows:
        return

    updates = [int(r["update"]) for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    fig.suptitle(f"Prediction Head Accuracy — {tag}", fontsize=12, fontweight="bold")

    for key, label, color in [
        ("item_acc",    "Item",    "#3498db"),
        ("ability_acc", "Ability", "#e67e22"),
        ("tera_acc",    "Tera",    "#2ecc71"),
        ("move_recall", "Move@4",  "#e74c3c"),
    ]:
        vals = [float(r[key]) * 100 for r in rows]
        ax.plot(updates, vals, label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Update")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    _save_fig(fig, save_path / "prediction_accuracy")
    plt.close(fig)


# ── Cosine similarity heatmaps ────────────────────────────────────────────────

_COS_LABELS = {
    "A_AT": ("Pre-transformer (A @ A^T)", ["O0","O1","O2","O3","O4","O5",
                                            "P0","P1","P2","P3","P4","P5","FL"]),
    "B_BT": ("Post-transformer / keys (B @ B^T)", ["O0","O1","O2","O3","O4","O5",
                                                     "P0","P1","P2","P3","P4","P5","FL"]),
    "C_CT": ("Action queries (C @ C^T)",
             ["M1","M2","M3","M4","T1","T2","T3","T4","S0","S1","S2","S3","S4"]),
    "B_CT": ("Keys vs Queries (B @ C^T)",
             ["M1","M2","M3","M4","T1","T2","T3","T4","S0","S1","S2","S3","S4"],
             ["O0","O1","O2","O3","O4","O5","P0","P1","P2","P3","P4","P5","FL"]),
    "A_CT": ("Pre-transformer vs Queries (A @ C^T)",
             ["M1","M2","M3","M4","T1","T2","T3","T4","S0","S1","S2","S3","S4"],
             ["O0","O1","O2","O3","O4","O5","P0","P1","P2","P3","P4","P5","FL"]),
}


def _plot_cos_sim(
    eval_results: dict[str, dict],
    save_path: Path,
    tag: str,
) -> None:
    """
    Save 5 cosine similarity heatmaps from the first opponent with cos_sim_matrices.
    """
    opp = next((k for k, v in eval_results.items() if "cos_sim_matrices" in v), None)
    if opp is None:
        return

    matrices = eval_results[opp]["cos_sim_matrices"]
    scalars  = eval_results[opp].get("cos_sim_scalars", {})

    for key, info in _COS_LABELS.items():
        mat = matrices.get(key)
        if mat is None:
            continue

        title = info[0]
        row_labels = info[1]
        col_labels = info[2] if len(info) > 2 else info[1]

        is_square = row_labels == col_labels
        fig_h = max(4, len(row_labels) * 0.45)
        fig_w = max(5, len(col_labels) * 0.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat.numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels(row_labels, fontsize=7)

        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = mat[i, j].item()
                if is_square and i == j:
                    continue  # skip diagonal for self-sim
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5, color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("cosine similarity", fontsize=8)

        # Add scalar metrics to title
        scalar_notes = ""
        if scalars:
            if key == "B_BT":
                scalar_notes = f"  off-diag={scalars.get('cos_post_offdiag', 0):.3f}"
            elif key == "C_CT":
                scalar_notes = f"  off-diag={scalars.get('cos_query_offdiag', 0):.3f}"
            elif key == "B_CT":
                scalar_notes = (f"  mean={scalars.get('cos_keys_queries_mean', 0):.3f}  "
                               f"n_keys={scalars.get('cos_n_unique_keys', 0)}")

        ax.set_title(f"{title}{scalar_notes} — {tag}", fontsize=9)

        fig.tight_layout()
        _save_fig(fig, save_path / f"{tag}_{key}")
        plt.close(fig)