"""
probing_svd.py — PCA & SVD on CynthAI_v2 backbone token representations.

Loads a pre-computed token cache (seq_all_cache.pt) and analyses:
  1. PCA (TruncatedSVD) at three levels of aggregation — how many independent
     directions exist in the token representations?
  2. SVD per state — what fraction of energy is captured by the top singular
     vectors of the 13-token matrix for each individual state?

No rollout, no checkpoint, no sklearn probes — only the cached tensors.

Usage:
    # Generate cache first (if not done by probing.py):
    python diag/make_token_cache.py \\
        --checkpoint checkpoints/cheater_v7/agent_000300.pt \\
        --n_envs 32 --min_steps 8192 --device cpu

    # Run SVD/PCA analysis:
    python diag/probing_svd.py --cache diag/seq_all_cache.pt

Output:
    diag/probing_svd_pca.png / .pdf
    diag/probing_svd_state.png / .pdf
    diag/probing_svd.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.backbone import D_MODEL, N_SLOTS, K_TURNS, SEQ_LEN


# ─────────────────────────────────────────────────────────────────────────────
# PCA (TruncatedSVD)
# ─────────────────────────────────────────────────────────────────────────────

def _pca_thresholds(cumvar: np.ndarray) -> dict[str, int]:
    """Return number of components needed to reach 80/90/95/99% variance."""
    result = {}
    for pct, key in [(0.80, "n80"), (0.90, "n90"), (0.95, "n95"), (0.99, "n99")]:
        idx = int(np.searchsorted(cumvar, pct))
        result[key] = min(idx + 1, len(cumvar))   # 1-indexed count
    return result


def run_pca(seq_all: torch.Tensor) -> dict:
    """
    TruncatedSVD at three levels on the mean-subtracted data.

    Levels:
      token_emb  : [N*52, 256]   — every individual token embedding
      state_cur  : [N, 13*256]   — current-turn tokens concatenated
      state_all  : [N, 52*256]   — all-turn tokens concatenated

    Returns dict with cumvar arrays and threshold dicts for each level.
    """
    N = seq_all.shape[0]
    results = {}

    configs = [
        ("token_emb", seq_all.reshape(N * SEQ_LEN, D_MODEL).numpy(),          200),
        ("state_cur", seq_all[:, -N_SLOTS:, :].reshape(N, N_SLOTS * D_MODEL).numpy(), 200),
        ("state_all", seq_all.reshape(N, SEQ_LEN * D_MODEL).numpy(),           200),
    ]

    for name, X, n_components in configs:
        n_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
        print(f"  PCA {name}: X={X.shape}  n_components={n_components} ...")

        # Mean-subtract
        X = X - X.mean(axis=0, keepdims=True)

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(X)

        cumvar = np.cumsum(svd.explained_variance_ratio_)
        thresholds = _pca_thresholds(cumvar)
        results[name] = {
            "cumvar":           cumvar.tolist(),
            "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
            **thresholds,
        }
        print(f"    n80={thresholds['n80']}  n90={thresholds['n90']}  "
              f"n95={thresholds['n95']}  n99={thresholds['n99']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SVD per state
# ─────────────────────────────────────────────────────────────────────────────

def run_svd_per_state(seq_all: torch.Tensor) -> dict:
    """
    For each state n, compute singular values of the token matrix and derive
    the energy distribution.

    current turn : M = seq_all[n, -13:, :]  shape [13, 256]  → s shape [13]
    all turns    : M = seq_all[n, :,   :]   shape [52, 256]  → s shape [52]

    Metrics per state:
      top-k energy = sum of top-k normalised squared singular values
      effective_rank = exp(-Σ p_i * log(p_i))  ∈ [1, n_tokens]

    Returns dict with per-metric mean/std and spectrum arrays.
    """
    results = {}

    for name, tokens in [
        ("current_turn", seq_all[:, -N_SLOTS:, :]),   # [N, 13, 256]
        ("all_turns",    seq_all),                      # [N, 52, 256]
    ]:
        M = tokens.float()
        n_tok = M.shape[1]
        print(f"  SVD per state — {name}: {tuple(M.shape)} ...")

        # Batch SVD: [N, n_tok, D_MODEL] → singular values [N, n_tok]
        s = torch.linalg.svdvals(M)                               # [N, n_tok]
        s2 = s ** 2
        energy = s2 / s2.sum(dim=-1, keepdim=True)                # [N, n_tok]

        # Effective rank (Shannon entropy of energy spectrum)
        eff_rank = torch.exp(
            -(energy * torch.log(energy.clamp(min=1e-10))).sum(dim=-1)
        )   # [N]

        spectrum_mean = energy.mean(dim=0).tolist()
        spectrum_std  = energy.std(dim=0).tolist()

        def _stat(x: torch.Tensor) -> tuple[float, float]:
            return float(x.mean()), float(x.std())

        top1  = energy[:, :1].sum(-1)
        top2  = energy[:, :2].sum(-1)
        top3  = energy[:, :3].sum(-1)
        top5  = energy[:, :5].sum(-1)

        entry: dict = {
            "top1_mean":            round(float(top1.mean()),     5),
            "top1_std":             round(float(top1.std()),      5),
            "top2_mean":            round(float(top2.mean()),     5),
            "top2_std":             round(float(top2.std()),      5),
            "top3_mean":            round(float(top3.mean()),     5),
            "top3_std":             round(float(top3.std()),      5),
            "top5_mean":            round(float(top5.mean()),     5),
            "top5_std":             round(float(top5.std()),      5),
            "effective_rank_mean":  round(float(eff_rank.mean()), 5),
            "effective_rank_std":   round(float(eff_rank.std()),  5),
            "spectrum_mean":        [round(v, 6) for v in spectrum_mean],
            "spectrum_std":         [round(v, 6) for v in spectrum_std],
        }

        if name == "all_turns":
            top10 = energy[:, :10].sum(-1)
            top20 = energy[:, :20].sum(-1)
            entry["top10_mean"] = round(float(top10.mean()), 5)
            entry["top10_std"]  = round(float(top10.std()),  5)
            entry["top20_mean"] = round(float(top20.mean()), 5)
            entry["top20_std"]  = round(float(top20.std()),  5)

        print(f"    top1={entry['top1_mean']:.4f}±{entry['top1_std']:.4f}  "
              f"eff_rank={entry['effective_rank_mean']:.3f}±{entry['effective_rank_std']:.3f}")

        results[name] = {
            "energy": energy,        # keep tensors for plotting
            "eff_rank": eff_rank,
            "metrics": entry,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: PCA cumvar — 2×2
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_pca(pca: dict, out_path: Path) -> None:
    """
    2×2 figure:
      [0,0] cumvar — token embeddings (256 dims)
      [0,1] cumvar — state current-turn (3328 dims)
      [1,0] cumvar — state all-turns (13312 dims)
      [1,1] summary table
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CynthAI_v2 — PCA / TruncatedSVD: effective dimensionality",
                 fontsize=12)

    THRESHOLD_COLORS = {
        "n80": ("#4CAF50", "80%"),
        "n90": ("#FF9800", "90%"),
        "n95": ("#F44336", "95%"),
        "n99": ("#9C2CF5", "99%"),
    }

    def _plot_cumvar(ax, name, title, max_x):
        d = pca[name]
        cv = np.array(d["cumvar"])
        x  = np.arange(1, len(cv) + 1)
        ax.plot(x, cv, color="#2196F3", lw=1.5, label="cumulative variance")
        ax.fill_between(x, cv, alpha=0.12, color="#2196F3")
        for key, (color, label) in THRESHOLD_COLORS.items():
            n = d[key]
            ax.axhline(float(key[1:]) / 100, color=color, ls="--", lw=0.9, alpha=0.6)
            ax.axvline(n, color=color, ls=":", lw=1.0, alpha=0.8,
                       label=f"{label}: n={n}")
        ax.set_xlim(1, max_x)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("# components")
        ax.set_ylabel("cumulative variance explained")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, lw=0.4, alpha=0.4)

    _plot_cumvar(axes[0, 0], "token_emb",
                 f"Token embeddings  [N×52, {D_MODEL}]",
                 max_x=D_MODEL)
    _plot_cumvar(axes[0, 1], "state_cur",
                 f"State current-turn  [N, {N_SLOTS}×{D_MODEL}={N_SLOTS*D_MODEL}]",
                 max_x=200)
    _plot_cumvar(axes[1, 0], "state_all",
                 f"State all-turns  [N, {SEQ_LEN}×{D_MODEL}={SEQ_LEN*D_MODEL}]",
                 max_x=200)

    # [1,1]: summary table
    ax = axes[1, 1]
    ax.axis("off")
    headers = ["Level", "80%", "90%", "95%", "99%"]
    rows = []
    for name, label in [
        ("token_emb", f"Token emb ({D_MODEL}D)"),
        ("state_cur", f"State cur ({N_SLOTS*D_MODEL}D)"),
        ("state_all", f"State all ({SEQ_LEN*D_MODEL}D)"),
    ]:
        d = pca[name]
        rows.append([label, d["n80"], d["n90"], d["n95"], d["n99"]])

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.3, 1.0, 0.5],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(weight="bold")

    ax.set_title("Summary: components needed", fontsize=9, pad=4)
    note = (
        "n_components to reach threshold\n"
        "(TruncatedSVD, mean-subtracted data)\n\n"
        "token_emb: each of the N×52 tokens\n"
        "state_cur: concat of 13 current tokens\n"
        "state_all: concat of 52 all-turn tokens"
    )
    ax.text(0.05, 0.25, note, transform=ax.transAxes, fontsize=7.5,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#BDBDBD"))

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = out_path.with_suffix(ext)
        plt.savefig(p, dpi=130, bbox_inches="tight")
        print(f"PCA figure {ext[1:].upper()}: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: SVD per state — 2×3
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_state(svd_res: dict, out_path: Path) -> None:
    """
    2×3 figure:
      [0,0] Spectrum mean±std — current turn (13 tokens)
      [0,1] Spectrum mean±std — all turns   (52 tokens)
      [0,2] Distribution top-1 energy — current turn
      [1,0] Violin: top-1/2/3/5 cumulative energy — current turn
      [1,1] Distribution top-1 energy — all turns
      [1,2] Violin: top-1/5/10/20 cumulative energy — all turns
    """
    cur = svd_res["current_turn"]
    all_ = svd_res["all_turns"]

    energy_cur  = cur["energy"].numpy()    # [N, 13]
    energy_all  = all_["energy"].numpy()   # [N, 52]
    eff_rank_cur = cur["eff_rank"].numpy() # [N]
    eff_rank_all = all_["eff_rank"].numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CynthAI_v2 — SVD per state: effective rank of token representations",
                 fontsize=12)

    # ── [0,0] Spectrum current turn ───────────────────────────────────────────
    ax = axes[0, 0]
    n_cur = energy_cur.shape[1]   # 13
    x_cur = np.arange(1, n_cur + 1)
    mean_cur = energy_cur.mean(axis=0)
    std_cur  = energy_cur.std(axis=0)
    ax.bar(x_cur, mean_cur, color="#2196F3", alpha=0.7, label="mean energy")
    ax.errorbar(x_cur, mean_cur, yerr=std_cur, fmt="none",
                color="#1565C0", capsize=3, lw=1.0)
    ax.axhline(1.0 / n_cur, color="#F44336", ls="--", lw=1.2,
               label=f"uniform = 1/{n_cur}={1/n_cur:.3f}")
    ax.set_xlabel("singular value rank")
    ax.set_ylabel("energy fraction")
    ax.set_title(f"Spectrum — current turn ({n_cur} tokens)\n"
                 f"eff_rank={cur['metrics']['effective_rank_mean']:.2f}"
                 f"±{cur['metrics']['effective_rank_std']:.2f}",
                 fontsize=9)
    ax.set_xticks(x_cur)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)

    # ── [0,1] Spectrum all turns ───────────────────────────────────────────────
    ax = axes[0, 1]
    n_all = energy_all.shape[1]   # 52
    x_all = np.arange(1, n_all + 1)
    mean_all = energy_all.mean(axis=0)
    std_all  = energy_all.std(axis=0)
    ax.bar(x_all, mean_all, color="#4CAF50", alpha=0.7, width=0.9, label="mean energy")
    ax.fill_between(x_all, mean_all - std_all, mean_all + std_all,
                    alpha=0.2, color="#4CAF50")
    ax.axhline(1.0 / n_all, color="#F44336", ls="--", lw=1.2,
               label=f"uniform = 1/{n_all}={1/n_all:.4f}")
    ax.set_xlabel("singular value rank")
    ax.set_ylabel("energy fraction")
    ax.set_title(f"Spectrum — all turns ({n_all} tokens)\n"
                 f"eff_rank={all_['metrics']['effective_rank_mean']:.2f}"
                 f"±{all_['metrics']['effective_rank_std']:.2f}",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)

    # ── [0,2] Distribution top-1 current turn ────────────────────────────────
    ax = axes[0, 2]
    top1_cur = energy_cur[:, 0]
    ax.hist(top1_cur, bins=60, density=True, color="#2196F3", alpha=0.7,
            edgecolor="none", label="top-1 energy")
    # KDE overlay
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(top1_cur)
        xg = np.linspace(top1_cur.min(), top1_cur.max(), 300)
        ax.plot(xg, kde(xg), color="#1565C0", lw=1.5)
    except ImportError:
        pass
    ax.axvline(top1_cur.mean(), color="#F44336", ls="--", lw=1.2,
               label=f"mean={top1_cur.mean():.3f}")
    ax.axvline(1.0 / n_cur, color="#9E9E9E", ls=":", lw=1.0,
               label=f"uniform={1/n_cur:.3f}")
    ax.set_xlabel("top-1 energy fraction")
    ax.set_ylabel("density")
    ax.set_title(f"Top-1 energy dist — current turn\n"
                 f"mean={top1_cur.mean():.3f}  std={top1_cur.std():.3f}",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, lw=0.4, alpha=0.4)

    # ── [1,0] Violin: cumulative top-k — current turn ─────────────────────────
    ax = axes[1, 0]
    ks_cur = [1, 2, 3, 5]
    cum_data_cur = [energy_cur[:, :k].sum(axis=-1) for k in ks_cur]
    refs_cur     = [k / n_cur for k in ks_cur]
    labels_cur   = [f"top-{k}" for k in ks_cur]

    parts = ax.violinplot(cum_data_cur, positions=range(1, len(ks_cur) + 1),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2196F3")
        pc.set_alpha(0.5)
    for i, (med_data, ref, lbl) in enumerate(zip(cum_data_cur, refs_cur, labels_cur), 1):
        ax.axhline(ref, color="#F44336", ls="--", lw=0.9, alpha=0.7)
        ax.text(i + 0.35, np.median(med_data) + 0.01,
                f"med={np.median(med_data):.3f}", fontsize=6.5, ha="left")
        ax.text(i, 0.01 + ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.01,
                f"eff={np.mean(eff_rank_cur):.2f}", fontsize=6, ha="center",
                color="gray")
    ax.set_xticks(range(1, len(ks_cur) + 1))
    ax.set_xticklabels(labels_cur)
    ax.set_ylabel("cumulative energy")
    ax.set_title(f"Cumulative top-k energy — current turn\n"
                 f"(dashed = uniform reference, 13 tokens)",
                 fontsize=9)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    # Annotate effective rank as text
    ax.text(0.98, 0.98, f"eff_rank mean={np.mean(eff_rank_cur):.2f}",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="#FFF3E0", ec="#FF9800"))

    # ── [1,1] Distribution top-1 all turns ───────────────────────────────────
    ax = axes[1, 1]
    top1_all = energy_all[:, 0]
    ax.hist(top1_all, bins=60, density=True, color="#4CAF50", alpha=0.7,
            edgecolor="none", label="top-1 energy")
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(top1_all)
        xg = np.linspace(top1_all.min(), top1_all.max(), 300)
        ax.plot(xg, kde(xg), color="#2E7D32", lw=1.5)
    except ImportError:
        pass
    ax.axvline(top1_all.mean(), color="#F44336", ls="--", lw=1.2,
               label=f"mean={top1_all.mean():.4f}")
    ax.axvline(1.0 / n_all, color="#9E9E9E", ls=":", lw=1.0,
               label=f"uniform={1/n_all:.4f}")
    ax.set_xlabel("top-1 energy fraction")
    ax.set_ylabel("density")
    ax.set_title(f"Top-1 energy dist — all turns\n"
                 f"mean={top1_all.mean():.4f}  std={top1_all.std():.4f}",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, lw=0.4, alpha=0.4)

    # ── [1,2] Violin: cumulative top-k — all turns ────────────────────────────
    ax = axes[1, 2]
    ks_all = [1, 5, 10, 20]
    cum_data_all = [energy_all[:, :k].sum(axis=-1) for k in ks_all]
    refs_all     = [k / n_all for k in ks_all]
    labels_all   = [f"top-{k}" for k in ks_all]

    parts = ax.violinplot(cum_data_all, positions=range(1, len(ks_all) + 1),
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4CAF50")
        pc.set_alpha(0.5)
    for i, (med_data, ref, lbl) in enumerate(zip(cum_data_all, refs_all, labels_all), 1):
        ax.axhline(ref, color="#F44336", ls="--", lw=0.9, alpha=0.7)
        ax.text(i + 0.35, np.median(med_data) + 0.01,
                f"med={np.median(med_data):.3f}", fontsize=6.5, ha="left")
    ax.set_xticks(range(1, len(ks_all) + 1))
    ax.set_xticklabels(labels_all)
    ax.set_ylabel("cumulative energy")
    ax.set_title(f"Cumulative top-k energy — all turns\n"
                 f"(dashed = uniform reference, {n_all} tokens)",
                 fontsize=9)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)
    ax.text(0.98, 0.98, f"eff_rank mean={np.mean(eff_rank_all):.2f}",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="#F1F8E9", ec="#8BC34A"))

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        p = out_path.with_suffix(ext)
        plt.savefig(p, dpi=130, bbox_inches="tight")
        print(f"State figure {ext[1:].upper()}: {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────────────────

def save_json(N: int, pca: dict, svd_res: dict, out_path: Path) -> None:
    pca_out = {}
    for name in ("token_emb", "state_cur", "state_all"):
        d = pca[name]
        pca_out[name] = {k: d[k] for k in ("n80", "n90", "n95", "n99")}

    svd_cur = svd_res["current_turn"]["metrics"]
    svd_all = svd_res["all_turns"]["metrics"]

    data = {
        "n_transitions": N,
        "pca": pca_out,
        "svd_current_turn": svd_cur,
        "svd_all_turns":    svd_all,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCA & SVD analysis on cached backbone token representations."
    )
    parser.add_argument("--cache", default="diag/seq_all_cache.pt",
                        help="Path to seq_all_cache.pt produced by make_token_cache.py "
                             "or probing.py")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        print("Run first: python diag/make_token_cache.py --checkpoint <ckpt>")
        sys.exit(1)

    out_dir = Path(__file__).parent

    # ── 1. Load cache ─────────────────────────────────────────────────────────
    print(f"Loading cache: {cache_path}")
    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
    seq_all = saved["seq_all"]                   # [N, 52, 256]
    N       = int(saved.get("n_transitions", seq_all.shape[0]))
    print(f"  seq_all: {tuple(seq_all.shape)}   n_transitions={N}")

    # ── 2. PCA ────────────────────────────────────────────────────────────────
    print("\nRunning PCA (TruncatedSVD) ...")
    pca = run_pca(seq_all)

    # ── 3. SVD per state ──────────────────────────────────────────────────────
    print("\nRunning SVD per state ...")
    svd_res = run_svd_per_state(seq_all)

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\nSaving figures ...")
    save_figure_pca(pca,     out_dir / "probing_svd_pca")
    save_figure_state(svd_res, out_dir / "probing_svd_state")

    # ── 5. JSON ───────────────────────────────────────────────────────────────
    save_json(N, pca, svd_res, out_dir / "probing_svd.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
