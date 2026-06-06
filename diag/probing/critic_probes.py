"""
critic_probes — Linear probes + PCA on IndependentCritic representations.

Analyses:
  1. Linear probes on critic CLS + 52 tokens + mean pools
  2. PCA of critic CLS (scatter by value/win, explained variance)
  3. Effective rank (CLS, all tokens, per-turn)
  4. Per-dimension correlations (CLS vs value, CLS vs win)

Required cache keys: critic_cls, critic_seq, critic_values, + labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.backbone import K_TURNS, N_SLOTS, D_MODEL, SEQ_LEN

from diag.probing._common import (
    SEQ_TOKEN_LABELS, SLOT_NAMES, BAR_COLORS_52, TURN_SEPS,
    fit_regression, fit_classification, effective_rank, per_dim_correlation,
    draw_turn_vlines, numpy_from_cache, get_labels_from_cache,
)


REQUIRED_KEYS = {"critic_cls", "critic_seq", "critic_values",
                 "y_return", "y_win", "y_type1", "y_item", "y_ability", "y_hp", "y_stats"}


def can_run(cache: dict) -> bool:
    return REQUIRED_KEYS.issubset(cache.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def _run_probes(critic_seq, critic_cls, labels, train_idx, val_idx):
    """Linear probes on critic's 52 tokens + CLS + mean pools."""
    y_return = labels["y_return"]
    y_win = labels["y_win"]
    y_type1 = labels["y_type1"]
    y_item = labels["y_item"]
    y_ability = labels["y_ability"]
    y_hp = labels["y_hp"]
    y_stats = labels["y_stats"]

    win_mask = y_win >= 0
    win_tr = [i for i in train_idx if win_mask[i]]
    win_val = [i for i in val_idx if win_mask[i]]

    # CLS probes
    print("\nCritic CLS probes:")
    X_cls = critic_cls

    cls_ret_r2, cls_ret_corr = fit_regression(
        X_cls[train_idx], y_return[train_idx], X_cls[val_idx], y_return[val_idx])
    cls_win_acc, cls_win_auc = 0.5, 0.5
    if len(win_tr) > 20 and len(win_val) > 10:
        cls_win_acc, cls_win_auc = fit_classification(
            X_cls[win_tr], y_win[win_tr].astype(np.int64),
            X_cls[win_val], y_win[win_val].astype(np.int64),
            max_iter=500, compute_auc=True)
    print(f"  CLS        | ret_r2={cls_ret_r2:+.3f}  ret_corr={cls_ret_corr:.3f}  "
          f"win_acc={cls_win_acc:.3f}  win_auc={cls_win_auc:.3f}")

    cls_results = {
        "return_r2": round(cls_ret_r2, 4), "return_corr": round(cls_ret_corr, 4),
        "win_acc": round(cls_win_acc, 4), "win_auc": round(cls_win_auc, 4),
    }

    # Per-token probes
    print("\nCritic per-token probes:")
    per_token = {}
    for tok_i in range(SEQ_LEN):
        t = tok_i // N_SLOTS
        s = tok_i % N_SLOTS
        poke_idx = t * 12 + s
        X = critic_seq[:, tok_i, :]

        ret_r2, ret_corr = fit_regression(
            X[train_idx], y_return[train_idx], X[val_idx], y_return[val_idx])
        win_acc, win_auc = 0.5, 0.5
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = fit_classification(
                X[win_tr], y_win[win_tr].astype(np.int64),
                X[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True)

        tok_res = {
            "return_r2": round(ret_r2, 4), "return_corr": round(ret_corr, 4),
            "win_acc": round(win_acc, 4), "win_auc": round(win_auc, 4),
        }

        if s < 12:
            type1_acc, _ = fit_classification(
                X[train_idx], y_type1[train_idx, poke_idx],
                X[val_idx], y_type1[val_idx, poke_idx], max_iter=500)
            item_acc, _ = fit_classification(
                X[train_idx], y_item[train_idx, poke_idx],
                X[val_idx], y_item[val_idx, poke_idx], max_iter=1000)
            ability_acc, _ = fit_classification(
                X[train_idx], y_ability[train_idx, poke_idx],
                X[val_idx], y_ability[val_idx, poke_idx], max_iter=1000)
            hp_r2, _ = fit_regression(
                X[train_idx], y_hp[train_idx, poke_idx],
                X[val_idx], y_hp[val_idx, poke_idx])
            stat_r2s = []
            for stat in range(5):
                r2, _ = fit_regression(
                    X[train_idx], y_stats[train_idx, poke_idx, stat],
                    X[val_idx], y_stats[val_idx, poke_idx, stat])
                stat_r2s.append(r2)
            bstats_r2 = float(np.mean(stat_r2s))
            tok_res.update({
                "type1_acc": round(type1_acc, 4), "item_acc": round(item_acc, 4),
                "ability_acc": round(ability_acc, 4), "hp_r2": round(hp_r2, 4),
                "basestats_r2": round(bstats_r2, 4),
            })
            print(f"  {SEQ_TOKEN_LABELS[tok_i]:8s} | "
                  f"ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}  "
                  f"type1={type1_acc:.3f}  item={item_acc:.3f}  "
                  f"ability={ability_acc:.3f}  hp={hp_r2:.3f}  bstats={bstats_r2:.3f}")
        else:
            tok_res.update({
                "type1_acc": None, "item_acc": None, "ability_acc": None,
                "hp_r2": None, "basestats_r2": None,
            })
            print(f"  {SEQ_TOKEN_LABELS[tok_i]:8s} | "
                  f"ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")

        per_token[str(tok_i)] = tok_res

    # Mean pool probes
    print("\nCritic mean pool probes:")
    pools = {}
    for label, X_pool in [
        ("mean_cur", critic_seq[:, -N_SLOTS:, :].mean(axis=1)),
        ("mean_all", critic_seq.mean(axis=1)),
    ]:
        ret_r2, ret_corr = fit_regression(
            X_pool[train_idx], y_return[train_idx], X_pool[val_idx], y_return[val_idx])
        win_acc, win_auc = 0.5, 0.5
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = fit_classification(
                X_pool[win_tr], y_win[win_tr].astype(np.int64),
                X_pool[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True)
        pools[label] = {
            "return_r2": round(ret_r2, 4), "return_corr": round(ret_corr, 4),
            "win_acc": round(win_acc, 4), "win_auc": round(win_auc, 4),
        }
        print(f"  {label:12s} | ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")

    return {"cls": cls_results, "per_token": per_token, "pools": pools}


def _run_effective_rank(critic_cls, critic_seq):
    """Effective rank of CLS, mean-pool, and per-turn tokens."""
    erank_cls = effective_rank(critic_cls)
    erank_all = effective_rank(critic_seq.reshape(-1, D_MODEL))
    erank_cur = effective_rank(critic_seq[:, -N_SLOTS:, :].reshape(-1, D_MODEL))

    print(f"\nEffective rank:")
    print(f"  CLS:         {erank_cls:.2f} / {D_MODEL}")
    print(f"  all tokens:  {erank_all:.2f} / {D_MODEL}")
    print(f"  cur tokens:  {erank_cur:.2f} / {D_MODEL}")

    per_turn = {}
    for t in range(K_TURNS):
        start = t * N_SLOTS
        end = start + N_SLOTS
        er = effective_rank(critic_seq[:, start:end, :].reshape(-1, D_MODEL))
        per_turn[f"T{t}"] = round(er, 2)
        print(f"  turn {t}:      {er:.2f} / {D_MODEL}")

    return {"cls": round(erank_cls, 2), "all": round(erank_all, 2),
            "cur": round(erank_cur, 2), "per_turn": per_turn, "d_model": D_MODEL}


def _run_dim_correlations(critic_cls, values, y_win):
    """Per-dimension Pearson r of CLS with V(s) and win label."""
    corr_value = per_dim_correlation(critic_cls, values)

    win_mask = y_win >= 0
    corr_win = np.zeros(D_MODEL)
    if win_mask.sum() > 50:
        corr_win = per_dim_correlation(critic_cls[win_mask], y_win[win_mask])

    profile_corr = (
        float(np.corrcoef(corr_value, corr_win)[0, 1])
        if np.std(corr_win) > 1e-10 else 0.0
    )

    print(f"\nPer-dim correlation CLS vs value/win:")
    print(f"  max |corr(dim, V)|:    {np.abs(corr_value).max():.4f}")
    print(f"  max |corr(dim, win)|:  {np.abs(corr_win).max():.4f}")
    print(f"  profile corr: {profile_corr:.4f}")

    return {
        "max_abs_corr_value": round(float(np.abs(corr_value).max()), 4),
        "max_abs_corr_win": round(float(np.abs(corr_win).max()), 4),
        "mean_abs_corr_value": round(float(np.abs(corr_value).mean()), 4),
        "mean_abs_corr_win": round(float(np.abs(corr_win).mean()), 4),
        "profile_corr": round(profile_corr, 4),
        "corr_value": corr_value.tolist(),
        "corr_win": corr_win.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _save_probes_figure(results, out_path):
    """Per-token probe bar charts for critic."""
    per = results["per_token"]
    cls = results["cls"]
    x = np.arange(SEQ_LEN)

    def _arr(key):
        return np.array([per[str(i)][key] if per[str(i)][key] is not None else np.nan
                         for i in range(SEQ_LEN)])

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("IndependentCritic -- per-token linear probes", fontsize=12)

    for ax in axes:
        draw_turn_vlines(ax)

    ax = axes[0]
    ax.bar(x, _arr("return_r2"), color=BAR_COLORS_52, width=0.8)
    ax.axhline(cls["return_r2"], color="purple", ls="--", lw=1.2, alpha=0.8,
               label=f'CLS={cls["return_r2"]:.3f}')
    ax.set_ylabel("Return R2"); ax.set_title("Return R2 per token"); ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x, _arr("win_auc"), color=BAR_COLORS_52, width=0.8)
    ax.axhline(cls["win_auc"], color="purple", ls="--", lw=1.2, alpha=0.8,
               label=f'CLS={cls["win_auc"]:.3f}')
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("Win AUC"); ax.set_title("Win AUC per token"); ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(x, _arr("hp_r2"), color=BAR_COLORS_52, width=0.8)
    ax.set_ylabel("HP R2"); ax.set_title("HP R2 per token")
    ax.set_xlabel("Token index"); ax.set_xticks(x)
    ax.set_xticklabels(SEQ_TOKEN_LABELS, rotation=90, fontsize=5)

    fig.tight_layout()
    plt.savefig(out_path / "critic_probes.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "critic_probes.pdf", bbox_inches="tight")
    print(f"  Critic probes figure: {out_path / 'critic_probes.png'}")
    plt.close(fig)


def _save_pca_figure(critic_cls, values, y_win, out_path):
    """PCA on critic CLS."""
    X_c = critic_cls - critic_cls.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n = min(50, len(S))
    var = S ** 2 / (critic_cls.shape[0] - 1)
    evr = var / var.sum()
    proj = U[:, :n] * S[:n]
    win_mask = y_win >= 0
    cumvar = np.cumsum(evr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("IndependentCritic CLS -- PCA", fontsize=12)

    ax = axes[0, 0]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=values, cmap="coolwarm", s=4, alpha=0.5, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="V(s)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("CLS PCA -- colored by V(s)")

    ax = axes[0, 1]
    if win_mask.any():
        idx_w = np.where(win_mask)[0]
        sc2 = ax.scatter(proj[idx_w, 0], proj[idx_w, 1], c=y_win[idx_w], cmap="RdYlGn",
                         s=4, alpha=0.5, edgecolors="none", vmin=0, vmax=1)
        fig.colorbar(sc2, ax=ax, label="Win (1) / Loss (0)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("CLS PCA -- colored by win/loss")

    ax = axes[1, 0]
    n_show = min(30, len(evr))
    ax.bar(range(n_show), evr[:n_show] * 100, color="#42A5F5")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Explained variance  (top {n_show} PCs)"); ax.set_xlim(-0.5, n_show - 0.5)

    ax = axes[1, 1]
    ax.plot(range(len(cumvar)), cumvar * 100, "o-", markersize=3, color="#1565C0")
    ax.axhline(90, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(95, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of PCs"); ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative explained variance")
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    ax.annotate(f"90% @ PC{n90}", xy=(n90, 90), fontsize=8, color="gray")
    ax.annotate(f"95% @ PC{n95}", xy=(n95, 95), fontsize=8, color="gray")

    fig.tight_layout()
    plt.savefig(out_path / "critic_pca.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "critic_pca.pdf", bbox_inches="tight")
    print(f"  Critic PCA figure: {out_path / 'critic_pca.png'}")
    plt.close(fig)


def _save_corr_figure(corr_results, out_path):
    """Per-dimension correlation figure."""
    corr_value = np.array(corr_results["corr_value"])
    corr_win = np.array(corr_results["corr_win"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("IndependentCritic CLS -- per-dimension correlations", fontsize=12)

    ax = axes[0, 0]
    ax.bar(range(D_MODEL), corr_value[np.argsort(corr_value)[::-1]], color="#1976D2", width=1.0)
    ax.set_xlabel("Dimension (sorted)"); ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs V(s)"); ax.axhline(0, color="black", lw=0.5)

    ax = axes[0, 1]
    ax.bar(range(D_MODEL), corr_win[np.argsort(corr_win)[::-1]], color="#388E3C", width=1.0)
    ax.set_xlabel("Dimension (sorted)"); ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs win label"); ax.axhline(0, color="black", lw=0.5)

    ax = axes[1, 0]
    ax.scatter(corr_value, corr_win, s=8, alpha=0.6, color="#7B1FA2")
    ax.set_xlabel("corr(dim, V)"); ax.set_ylabel("corr(dim, win)")
    ax.set_title(f"Value vs Win (profile r={corr_results['profile_corr']:.3f})")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5); ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    lim = max(np.abs(corr_value).max(), np.abs(corr_win).max()) * 1.1
    if lim > 0:
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    ax = axes[1, 1]
    ax.hist(np.abs(corr_value), bins=30, alpha=0.6, label="|corr(dim, V)|", color="#1976D2")
    ax.hist(np.abs(corr_win), bins=30, alpha=0.6, label="|corr(dim, win)|", color="#388E3C")
    ax.set_xlabel("|Pearson r|"); ax.set_ylabel("Count")
    ax.set_title("Distribution of |correlation| per dimension"); ax.legend(fontsize=8)

    fig.tight_layout()
    plt.savefig(out_path / "critic_corr.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "critic_corr.pdf", bbox_inches="tight")
    print(f"  Critic correlation figure: {out_path / 'critic_corr.png'}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cache: dict, train_idx: list[int], val_idx: list[int], out_dir: Path) -> dict:
    """Run all critic probes. Returns results dict."""
    if not can_run(cache):
        missing = REQUIRED_KEYS - set(cache.keys())
        print(f"  [SKIP] critic_probes: missing keys {missing}")
        return {}

    critic_cls = numpy_from_cache(cache, "critic_cls", np.float32)
    critic_seq = numpy_from_cache(cache, "critic_seq", np.float32)
    values = numpy_from_cache(cache, "critic_values", np.float32).squeeze(-1)
    labels = get_labels_from_cache(cache)
    N = critic_cls.shape[0]

    print(f"\n{'='*60}")
    print("CRITIC PROBES (IndependentCritic)")
    print(f"{'='*60}")

    # 1. Linear probes
    probe_results = _run_probes(critic_seq, critic_cls, labels, train_idx, val_idx)

    # 2. Effective rank
    erank_results = _run_effective_rank(critic_cls, critic_seq)

    # 3. Per-dim correlations
    corr_results = _run_dim_correlations(critic_cls, values, labels["y_win"])

    # 4. Figures
    _save_probes_figure(probe_results, out_dir)
    _save_pca_figure(critic_cls, values, labels["y_win"], out_dir)
    _save_corr_figure(corr_results, out_dir)

    # 5. JSON
    all_results = {
        "n_transitions": N,
        "probes": probe_results,
        "effective_rank": erank_results,
        "dim_correlations": {k: v for k, v in corr_results.items()
                             if k not in ("corr_value", "corr_win")},
    }
    json_path = out_dir / "critic_probes.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  JSON: {json_path}")

    return all_results


def run_light(cache: dict, train_idx: list[int], val_idx: list[int], out_dir: Path) -> dict:
    """Light version: CLS probes + effective rank only (no per-token, no figures)."""
    if not can_run(cache):
        missing = REQUIRED_KEYS - set(cache.keys())
        print(f"  [SKIP] critic_probes (light): missing keys {missing}")
        return {}

    critic_cls = numpy_from_cache(cache, "critic_cls", np.float32)
    critic_seq = numpy_from_cache(cache, "critic_seq", np.float32)
    values = numpy_from_cache(cache, "critic_values", np.float32).squeeze(-1)
    labels = get_labels_from_cache(cache)
    N = critic_cls.shape[0]

    print(f"\n{'='*60}")
    print("CRITIC PROBES (light — CLS + effective rank)")
    print(f"{'='*60}")

    y_return = labels["y_return"]
    y_win = labels["y_win"]
    win_mask = y_win >= 0
    win_tr = [i for i in train_idx if win_mask[i]]
    win_val = [i for i in val_idx if win_mask[i]]

    # CLS probes
    X_cls = critic_cls
    cls_ret_r2, cls_ret_corr = fit_regression(
        X_cls[train_idx], y_return[train_idx], X_cls[val_idx], y_return[val_idx])
    cls_win_acc, cls_win_auc = 0.5, 0.5
    if len(win_tr) > 20 and len(win_val) > 10:
        cls_win_acc, cls_win_auc = fit_classification(
            X_cls[win_tr], y_win[win_tr].astype(np.int64),
            X_cls[win_val], y_win[win_val].astype(np.int64),
            max_iter=500, compute_auc=True)
    print(f"  CLS | ret_r2={cls_ret_r2:+.3f}  win_auc={cls_win_auc:.3f}")

    cls_results = {
        "return_r2": round(cls_ret_r2, 4), "return_corr": round(cls_ret_corr, 4),
        "win_acc": round(cls_win_acc, 4), "win_auc": round(cls_win_auc, 4),
    }

    # Effective rank
    erank_results = _run_effective_rank(critic_cls, critic_seq)

    output = {
        "n_transitions": N,
        "probes": {"cls": cls_results},
        "effective_rank": erank_results,
    }

    json_path = out_dir / "critic_light.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON: {json_path}")

    return output
