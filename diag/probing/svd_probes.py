"""
svd_probes — PCA & SVD analysis on token representations.

Analyses:
  1. PCA (TruncatedSVD) at 3 levels: token_emb, state_cur, state_all
  2. SVD per state: energy spectrum, effective rank

Required cache keys: seq_all
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

from model.backbone import D_MODEL, N_SLOTS, K_TURNS, SEQ_LEN


REQUIRED_KEYS = {"seq_all"}


def can_run(cache: dict) -> bool:
    return REQUIRED_KEYS.issubset(cache.keys())


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────

def _pca_thresholds(cumvar):
    result = {}
    for pct, key in [(0.80, "n80"), (0.90, "n90"), (0.95, "n95"), (0.99, "n99")]:
        idx = int(np.searchsorted(cumvar, pct))
        result[key] = min(idx + 1, len(cumvar))
    return result


def _run_pca(seq_all):
    N = seq_all.shape[0]
    results = {}

    configs = [
        ("token_emb", seq_all.reshape(N * SEQ_LEN, D_MODEL).numpy(), 200),
        ("state_cur", seq_all[:, -N_SLOTS:, :].reshape(N, N_SLOTS * D_MODEL).numpy(), 200),
        ("state_all", seq_all.reshape(N, SEQ_LEN * D_MODEL).numpy(), 200),
    ]

    for name, X, n_components in configs:
        n_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
        print(f"  PCA {name}: X={X.shape}  n_components={n_components} ...")
        X = X - X.mean(axis=0, keepdims=True)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        svd.fit(X)
        cumvar = np.cumsum(svd.explained_variance_ratio_)
        thresholds = _pca_thresholds(cumvar)
        results[name] = {
            "cumvar": cumvar.tolist(),
            "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
            **thresholds,
        }
        print(f"    n80={thresholds['n80']}  n90={thresholds['n90']}  "
              f"n95={thresholds['n95']}  n99={thresholds['n99']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SVD per state
# ─────────────────────────────────────────────────────────────────────────────

def _run_svd_per_state(seq_all):
    results = {}

    for name, tokens in [
        ("current_turn", seq_all[:, -N_SLOTS:, :]),
        ("all_turns", seq_all),
    ]:
        M = tokens.float()
        n_tok = M.shape[1]
        print(f"  SVD per state — {name}: {tuple(M.shape)} ...")

        s = torch.linalg.svdvals(M)
        s2 = s ** 2
        energy = s2 / s2.sum(dim=-1, keepdim=True)

        eff_rank = torch.exp(
            -(energy * torch.log(energy.clamp(min=1e-10))).sum(dim=-1))

        def _stat(x): return float(x.mean()), float(x.std())

        entry = {}
        for k_val in [1, 2, 3, 5]:
            topk = energy[:, :k_val].sum(-1)
            entry[f"top{k_val}_mean"] = round(float(topk.mean()), 5)
            entry[f"top{k_val}_std"] = round(float(topk.std()), 5)
        entry["effective_rank_mean"] = round(float(eff_rank.mean()), 5)
        entry["effective_rank_std"] = round(float(eff_rank.std()), 5)
        entry["spectrum_mean"] = [round(v, 6) for v in energy.mean(dim=0).tolist()]
        entry["spectrum_std"] = [round(v, 6) for v in energy.std(dim=0).tolist()]

        if name == "all_turns":
            for k_val in [10, 20]:
                topk = energy[:, :k_val].sum(-1)
                entry[f"top{k_val}_mean"] = round(float(topk.mean()), 5)
                entry[f"top{k_val}_std"] = round(float(topk.std()), 5)

        print(f"    top1={entry['top1_mean']:.4f}±{entry['top1_std']:.4f}  "
              f"eff_rank={entry['effective_rank_mean']:.3f}±{entry['effective_rank_std']:.3f}")

        results[name] = {"energy": energy, "eff_rank": eff_rank, "metrics": entry}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _save_pca_figure(pca, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CynthAI_v2 — PCA / TruncatedSVD: effective dimensionality", fontsize=12)

    THRESHOLD_COLORS = {
        "n80": ("#4CAF50", "80%"), "n90": ("#FF9800", "90%"),
        "n95": ("#F44336", "95%"), "n99": ("#9C2CF5", "99%"),
    }

    def _plot(ax, name, title, max_x):
        d = pca[name]
        cv = np.array(d["cumvar"])
        x = np.arange(1, len(cv) + 1)
        ax.plot(x, cv, color="#2196F3", lw=1.5)
        ax.fill_between(x, cv, alpha=0.12, color="#2196F3")
        for key, (color, label) in THRESHOLD_COLORS.items():
            n = d[key]
            ax.axhline(float(key[1:]) / 100, color=color, ls="--", lw=0.9, alpha=0.6)
            ax.axvline(n, color=color, ls=":", lw=1.0, alpha=0.8, label=f"{label}: n={n}")
        ax.set_xlim(1, max_x); ax.set_ylim(0, 1.02)
        ax.set_xlabel("# components"); ax.set_ylabel("cumulative variance")
        ax.set_title(title, fontsize=9); ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, lw=0.4, alpha=0.4)

    _plot(axes[0, 0], "token_emb", f"Token embeddings [{D_MODEL}D]", D_MODEL)
    _plot(axes[0, 1], "state_cur", f"State cur [{N_SLOTS*D_MODEL}D]", 200)
    _plot(axes[1, 0], "state_all", f"State all [{SEQ_LEN*D_MODEL}D]", 200)

    ax = axes[1, 1]
    ax.axis("off")
    headers = ["Level", "80%", "90%", "95%", "99%"]
    rows = [[label, pca[name]["n80"], pca[name]["n90"], pca[name]["n95"], pca[name]["n99"]]
            for name, label in [("token_emb", f"Token ({D_MODEL}D)"),
                                ("state_cur", f"Cur ({N_SLOTS*D_MODEL}D)"),
                                ("state_all", f"All ({SEQ_LEN*D_MODEL}D)")]]
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center",
                   bbox=[0.0, 0.3, 1.0, 0.5])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    ax.set_title("Summary: components needed", fontsize=9, pad=4)

    fig.tight_layout()
    plt.savefig(out_path / "svd_pca.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "svd_pca.pdf", bbox_inches="tight")
    print(f"  PCA figure: {out_path / 'svd_pca.png'}")
    plt.close(fig)


def _save_state_figure(svd_res, out_path):
    cur = svd_res["current_turn"]
    all_ = svd_res["all_turns"]
    energy_cur = cur["energy"].numpy()
    energy_all = all_["energy"].numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CynthAI_v2 — SVD per state: effective rank of token representations", fontsize=12)

    # Spectrum current turn
    ax = axes[0, 0]
    n_cur = energy_cur.shape[1]
    x_cur = np.arange(1, n_cur + 1)
    mean_cur = energy_cur.mean(axis=0)
    std_cur = energy_cur.std(axis=0)
    ax.bar(x_cur, mean_cur, color="#2196F3", alpha=0.7)
    ax.errorbar(x_cur, mean_cur, yerr=std_cur, fmt="none", color="#1565C0", capsize=3, lw=1.0)
    ax.axhline(1.0 / n_cur, color="#F44336", ls="--", lw=1.2, label=f"uniform=1/{n_cur}")
    ax.set_xlabel("singular value rank"); ax.set_ylabel("energy fraction")
    ax.set_title(f"Spectrum — current turn ({n_cur} tokens)", fontsize=9)
    ax.set_xticks(x_cur); ax.legend(fontsize=7)

    # Spectrum all turns
    ax = axes[0, 1]
    n_all = energy_all.shape[1]
    x_all = np.arange(1, n_all + 1)
    mean_all = energy_all.mean(axis=0)
    ax.bar(x_all, mean_all, color="#4CAF50", alpha=0.7, width=0.9)
    ax.axhline(1.0 / n_all, color="#F44336", ls="--", lw=1.2, label=f"uniform=1/{n_all}")
    ax.set_xlabel("singular value rank"); ax.set_ylabel("energy fraction")
    ax.set_title(f"Spectrum — all turns ({n_all} tokens)", fontsize=9); ax.legend(fontsize=7)

    # Top-1 distribution current turn
    ax = axes[0, 2]
    top1_cur = energy_cur[:, 0]
    ax.hist(top1_cur, bins=60, density=True, color="#2196F3", alpha=0.7, edgecolor="none")
    ax.axvline(top1_cur.mean(), color="#F44336", ls="--", lw=1.2, label=f"mean={top1_cur.mean():.3f}")
    ax.set_xlabel("top-1 energy"); ax.set_ylabel("density")
    ax.set_title("Top-1 energy dist — current turn", fontsize=9); ax.legend(fontsize=7)

    # Violin current turn
    ax = axes[1, 0]
    ks = [1, 2, 3, 5]
    cum_data = [energy_cur[:, :k].sum(axis=-1) for k in ks]
    parts = ax.violinplot(cum_data, positions=range(1, len(ks) + 1), showmedians=True, showextrema=False)
    for pc in parts["bodies"]: pc.set_facecolor("#2196F3"); pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(ks) + 1)); ax.set_xticklabels([f"top-{k}" for k in ks])
    ax.set_ylabel("cumulative energy"); ax.set_title("Cumulative top-k — current turn", fontsize=9)

    # Top-1 distribution all turns
    ax = axes[1, 1]
    top1_all = energy_all[:, 0]
    ax.hist(top1_all, bins=60, density=True, color="#4CAF50", alpha=0.7, edgecolor="none")
    ax.axvline(top1_all.mean(), color="#F44336", ls="--", lw=1.2, label=f"mean={top1_all.mean():.4f}")
    ax.set_xlabel("top-1 energy"); ax.set_ylabel("density")
    ax.set_title("Top-1 energy dist — all turns", fontsize=9); ax.legend(fontsize=7)

    # Violin all turns
    ax = axes[1, 2]
    ks_all = [1, 5, 10, 20]
    cum_data_all = [energy_all[:, :k].sum(axis=-1) for k in ks_all]
    parts = ax.violinplot(cum_data_all, positions=range(1, len(ks_all) + 1), showmedians=True, showextrema=False)
    for pc in parts["bodies"]: pc.set_facecolor("#4CAF50"); pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(ks_all) + 1)); ax.set_xticklabels([f"top-{k}" for k in ks_all])
    ax.set_ylabel("cumulative energy"); ax.set_title("Cumulative top-k — all turns", fontsize=9)

    fig.tight_layout()
    plt.savefig(out_path / "svd_state.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "svd_state.pdf", bbox_inches="tight")
    print(f"  SVD state figure: {out_path / 'svd_state.png'}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cache: dict, train_idx: list[int], val_idx: list[int], out_dir: Path) -> dict:
    if not can_run(cache):
        print(f"  [SKIP] svd_probes: missing 'seq_all'")
        return {}

    seq_all = cache["seq_all"]
    N = seq_all.shape[0]

    print(f"\n{'='*60}")
    print("SVD PROBES (PCA + per-state SVD)")
    print(f"{'='*60}")

    print("\nRunning PCA (TruncatedSVD) ...")
    pca = _run_pca(seq_all)

    print("\nRunning SVD per state ...")
    svd_res = _run_svd_per_state(seq_all)

    _save_pca_figure(pca, out_dir)
    _save_state_figure(svd_res, out_dir)

    # JSON (strip tensor data)
    pca_out = {}
    for name in ("token_emb", "state_cur", "state_all"):
        pca_out[name] = {k: pca[name][k] for k in ("n80", "n90", "n95", "n99")}

    output = {
        "n_transitions": N,
        "pca": pca_out,
        "svd_current_turn": svd_res["current_turn"]["metrics"],
        "svd_all_turns": svd_res["all_turns"]["metrics"],
    }

    json_path = out_dir / "svd_probes.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON: {json_path}")

    return output
