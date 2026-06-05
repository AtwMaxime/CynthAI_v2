"""
probing_critic.py -- Linear probes + representation analysis on IndependentCritic.

Analyses performed on the critic's CLS token and 52 post-Transformer tokens:

  1. Linear probes (same targets as probing.py):
     - CLS:  return R2, win AUC
     - Per-token (52):  return R2, win AUC, type1/item/ability acc, HP R2, base stats R2
     - Mean pool (current 13, all 52): return R2, win AUC

  2. PCA of CLS token:
     - 2D scatter colored by value, by win label
     - Explained variance ratio curve (first 50 components)

  3. Effective rank of CLS and per-token representations:
     - erank = exp(H(sigma)) where H is Shannon entropy of normalised singular values

  4. Per-dimension correlation of CLS with value and win:
     - Pearson r for each of the D_MODEL dimensions vs V(s) and vs win label
     - Sorted bar plots

Usage:
    python diag/probing_critic.py --cache diag/seq_all_cache.pt
    python diag/probing_critic.py --cache diag/seq_all_cache.pt --output diag/probing_critic

Output: probing_critic.json, probing_critic.png, probing_critic_pca.png
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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.backbone import K_TURNS, N_SLOTS, D_MODEL, SEQ_LEN


# ── Token labels (same as probing.py) ───────────────────────────────────────

SEQ_TOKEN_LABELS = []
for _t in range(K_TURNS):
    for _s in range(N_SLOTS):
        if _s < 6:
            SEQ_TOKEN_LABELS.append(f"T{_t}:O{_s}")
        elif _s < 12:
            SEQ_TOKEN_LABELS.append(f"T{_t}:P{_s-6}")
        else:
            SEQ_TOKEN_LABELS.append(f"T{_t}:F")


# ── Probe helpers (mirror probing.py) ───────────────────────────────────────

def _fit_regression(X_tr, y_tr, X_val, y_val):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    m = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
    pred = m.predict(X_val_s)
    ss_res = float(np.sum((y_val - pred) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    if ss_tot < 1e-6:
        return float("nan"), 0.0
    r2 = max(-1.0, float(1.0 - ss_res / ss_tot))
    corr = (
        float(np.corrcoef(pred, y_val)[0, 1])
        if np.std(pred) > 1e-10 and np.std(y_val) > 1e-10
        else 0.0
    )
    return r2, corr


def _fit_classification(X_tr, y_tr, X_val, y_val, max_iter=500, compute_auc=False):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    classes = np.unique(y_tr)
    if len(classes) < 2:
        return float(np.mean(np.full(len(y_val), classes[0]) == y_val)), 0.5
    m = LogisticRegression(max_iter=max_iter).fit(X_tr_s, y_tr)
    pred = m.predict(X_val_s)
    acc = float(np.mean(pred == y_val))
    auc = 0.5
    if compute_auc:
        try:
            all_cls = np.unique(np.concatenate([y_tr, y_val]))
            if len(all_cls) == 2:
                auc = float(roc_auc_score(y_val, m.predict_proba(X_val_s)[:, 1]))
            else:
                auc = float(roc_auc_score(
                    y_val, m.predict_proba(X_val_s),
                    multi_class="ovr", average="macro", labels=m.classes_,
                ))
        except Exception:
            auc = 0.5
    return acc, auc


# ── Effective rank ──────────────────────────────────────────────────────────

def effective_rank(X: np.ndarray) -> float:
    """
    Effective rank = exp(H(sigma_norm)) where sigma are singular values of X
    and H is Shannon entropy of the normalised spectrum.
    Measures the "intrinsic dimensionality" of the representation.
    """
    # Center
    X_c = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X_c, compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


# ── Per-dimension correlation ───────────────────────────────────────────────

def per_dim_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pearson r between each column of X and y.
    Returns array of shape [D].
    """
    D = X.shape[1]
    corrs = np.zeros(D, dtype=np.float64)
    y_std = np.std(y)
    if y_std < 1e-10:
        return corrs
    for d in range(D):
        if np.std(X[:, d]) < 1e-10:
            corrs[d] = 0.0
        else:
            corrs[d] = np.corrcoef(X[:, d], y)[0, 1]
    return corrs


# ── Linear probes on critic tokens ──────────────────────────────────────────

def run_critic_probes(
    critic_seq: np.ndarray,   # [N, 52, D_MODEL]
    critic_cls: np.ndarray,   # [N, D_MODEL]
    labels: dict,
    train_idx: list[int],
    val_idx: list[int],
) -> dict:
    """Run linear probes on critic's 52 tokens + CLS + mean pools."""

    y_return  = labels["y_return"]
    y_win     = labels["y_win"]
    y_type1   = labels["y_type1"]
    y_item    = labels["y_item"]
    y_ability = labels["y_ability"]
    y_hp      = labels["y_hp"]
    y_stats   = labels["y_stats"]

    win_mask = y_win >= 0
    win_tr   = [i for i in train_idx if win_mask[i]]
    win_val  = [i for i in val_idx   if win_mask[i]]

    # ── CLS probes ──────────────────────────────────────────────────────────
    print("\nCritic CLS probes:")
    X_cls = critic_cls

    cls_ret_r2, cls_ret_corr = _fit_regression(
        X_cls[train_idx], y_return[train_idx],
        X_cls[val_idx],   y_return[val_idx],
    )
    cls_win_acc, cls_win_auc = 0.5, 0.5
    if len(win_tr) > 20 and len(win_val) > 10:
        cls_win_acc, cls_win_auc = _fit_classification(
            X_cls[win_tr], y_win[win_tr].astype(np.int64),
            X_cls[win_val], y_win[win_val].astype(np.int64),
            max_iter=500, compute_auc=True,
        )
    print(f"  CLS        | ret_r2={cls_ret_r2:+.3f}  ret_corr={cls_ret_corr:.3f}  "
          f"win_acc={cls_win_acc:.3f}  win_auc={cls_win_auc:.3f}")

    cls_results = {
        "return_r2":   round(cls_ret_r2,   4),
        "return_corr": round(cls_ret_corr, 4),
        "win_acc":     round(cls_win_acc,  4),
        "win_auc":     round(cls_win_auc,  4),
    }

    # ── Per-token probes (52 tokens) ────────────────────────────────────────
    print("\nCritic per-token probes:")
    per_token = {}
    for tok_i in range(SEQ_LEN):
        t = tok_i // N_SLOTS
        s = tok_i % N_SLOTS
        poke_idx = t * 12 + s

        X = critic_seq[:, tok_i, :]

        ret_r2, ret_corr = _fit_regression(
            X[train_idx], y_return[train_idx],
            X[val_idx],   y_return[val_idx],
        )
        win_acc, win_auc = 0.5, 0.5
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = _fit_classification(
                X[win_tr], y_win[win_tr].astype(np.int64),
                X[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True,
            )

        tok_res = {
            "return_r2":   round(ret_r2,   4),
            "return_corr": round(ret_corr, 4),
            "win_acc":     round(win_acc,  4),
            "win_auc":     round(win_auc,  4),
        }

        if s < 12:
            type1_acc, _ = _fit_classification(
                X[train_idx], y_type1[train_idx, poke_idx],
                X[val_idx],   y_type1[val_idx, poke_idx], max_iter=500)
            item_acc, _ = _fit_classification(
                X[train_idx], y_item[train_idx, poke_idx],
                X[val_idx],   y_item[val_idx, poke_idx], max_iter=1000)
            ability_acc, _ = _fit_classification(
                X[train_idx], y_ability[train_idx, poke_idx],
                X[val_idx],   y_ability[val_idx, poke_idx], max_iter=1000)
            hp_r2, _ = _fit_regression(
                X[train_idx], y_hp[train_idx, poke_idx],
                X[val_idx],   y_hp[val_idx, poke_idx])
            stat_r2s = []
            for stat in range(5):
                r2, _ = _fit_regression(
                    X[train_idx], y_stats[train_idx, poke_idx, stat],
                    X[val_idx],   y_stats[val_idx, poke_idx, stat])
                stat_r2s.append(r2)
            bstats_r2 = float(np.mean(stat_r2s))
            tok_res.update({
                "type1_acc":    round(type1_acc, 4),
                "item_acc":     round(item_acc, 4),
                "ability_acc":  round(ability_acc, 4),
                "hp_r2":        round(hp_r2, 4),
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

    # ── Mean pool probes ────────────────────────────────────────────────────
    print("\nCritic mean pool probes:")
    pools = {}
    for label, X_pool in [
        ("mean_cur", critic_seq[:, -N_SLOTS:, :].mean(axis=1)),
        ("mean_all", critic_seq.mean(axis=1)),
    ]:
        ret_r2, ret_corr = _fit_regression(
            X_pool[train_idx], y_return[train_idx],
            X_pool[val_idx],   y_return[val_idx])
        win_acc, win_auc = 0.5, 0.5
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = _fit_classification(
                X_pool[win_tr], y_win[win_tr].astype(np.int64),
                X_pool[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True)
        pools[label] = {
            "return_r2":   round(ret_r2, 4),
            "return_corr": round(ret_corr, 4),
            "win_acc":     round(win_acc, 4),
            "win_auc":     round(win_auc, 4),
        }
        print(f"  {label:12s} | ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")

    return {
        "cls": cls_results,
        "per_token": per_token,
        "pools": pools,
    }


# ── PCA + scatter ───────────────────────────────────────────────────────────

def run_pca(X: np.ndarray, n_components: int = 50):
    """PCA on centered data. Returns (projected, explained_var_ratio, components)."""
    X_c = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n = min(n_components, len(S))
    var = S ** 2 / (X.shape[0] - 1)
    evr = var / var.sum()
    proj = U[:, :n] * S[:n]   # [N, n]
    return proj, evr[:n], Vt[:n]


def save_pca_figure(
    critic_cls: np.ndarray,
    values: np.ndarray,
    y_win: np.ndarray,
    out_path: Path,
) -> dict:
    """
    PCA on critic CLS:
      - 2D scatter colored by V(s) and by win label
      - Explained variance ratio bar chart (top 50 components)
    """
    proj, evr, _ = run_pca(critic_cls, n_components=50)

    win_mask = y_win >= 0
    cumvar = np.cumsum(evr)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("IndependentCritic CLS -- PCA", fontsize=12)

    # [0,0] PCA colored by value
    ax = axes[0, 0]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=values, cmap="coolwarm",
                    s=4, alpha=0.5, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="V(s)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("CLS PCA -- colored by V(s)")

    # [0,1] PCA colored by win label
    ax = axes[0, 1]
    if win_mask.any():
        idx_w = np.where(win_mask)[0]
        sc2 = ax.scatter(proj[idx_w, 0], proj[idx_w, 1],
                         c=y_win[idx_w], cmap="RdYlGn", s=4, alpha=0.5,
                         edgecolors="none", vmin=0, vmax=1)
        fig.colorbar(sc2, ax=ax, label="Win (1) / Loss (0)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("CLS PCA -- colored by win/loss")

    # [1,0] Explained variance ratio
    ax = axes[1, 0]
    n_show = min(30, len(evr))
    ax.bar(range(n_show), evr[:n_show] * 100, color="#42A5F5")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Explained variance  (top {n_show} PCs)")
    ax.set_xlim(-0.5, n_show - 0.5)

    # [1,1] Cumulative variance
    ax = axes[1, 1]
    ax.plot(range(len(cumvar)), cumvar * 100, "o-", markersize=3, color="#1565C0")
    ax.axhline(90, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(95, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Cumulative explained variance")
    # Find 90% and 95% thresholds
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    ax.annotate(f"90% @ PC{n90}", xy=(n90, 90), fontsize=8, color="gray")
    ax.annotate(f"95% @ PC{n95}", xy=(n95, 95), fontsize=8, color="gray")

    fig.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_pca.png"), dpi=130, bbox_inches="tight")
    plt.savefig(out_path.with_name(out_path.stem + "_pca.pdf"), bbox_inches="tight")
    print(f"PCA figure: {out_path.with_name(out_path.stem + '_pca.png')}")

    return {
        "evr_top10":   evr[:10].tolist(),
        "cumvar_at_10": float(cumvar[min(9, len(cumvar)-1)]),
        "cumvar_at_30": float(cumvar[min(29, len(cumvar)-1)]) if len(cumvar) > 29 else None,
        "n_pc_90pct":   int(n90),
        "n_pc_95pct":   int(n95),
    }


# ── Effective rank analysis ─────────────────────────────────────────────────

def run_effective_rank(
    critic_cls: np.ndarray,
    critic_seq: np.ndarray,
) -> dict:
    """Effective rank of CLS, mean-pool, and per-turn tokens."""
    erank_cls = effective_rank(critic_cls)
    erank_all = effective_rank(critic_seq.reshape(-1, D_MODEL))
    erank_cur = effective_rank(critic_seq[:, -N_SLOTS:, :].reshape(-1, D_MODEL))

    print(f"\nEffective rank:")
    print(f"  CLS:         {erank_cls:.2f} / {D_MODEL}")
    print(f"  all tokens:  {erank_all:.2f} / {D_MODEL}")
    print(f"  cur tokens:  {erank_cur:.2f} / {D_MODEL}")

    # Per-turn effective rank
    per_turn = {}
    for t in range(K_TURNS):
        start = t * N_SLOTS
        end   = start + N_SLOTS
        er = effective_rank(critic_seq[:, start:end, :].reshape(-1, D_MODEL))
        per_turn[f"T{t}"] = round(er, 2)
        print(f"  turn {t}:      {er:.2f} / {D_MODEL}")

    return {
        "cls":       round(erank_cls, 2),
        "all":       round(erank_all, 2),
        "cur":       round(erank_cur, 2),
        "per_turn":  per_turn,
        "d_model":   D_MODEL,
    }


# ── Per-dim correlation ─────────────────────────────────────────────────────

def run_dim_correlations(
    critic_cls: np.ndarray,
    values: np.ndarray,
    y_win: np.ndarray,
    out_path: Path,
) -> dict:
    """
    Per-dimension Pearson r of CLS with V(s) and win label.
    Saves sorted bar plots.
    """
    corr_value = per_dim_correlation(critic_cls, values)

    win_mask = y_win >= 0
    if win_mask.sum() > 50:
        corr_win = per_dim_correlation(critic_cls[win_mask], y_win[win_mask])
    else:
        corr_win = np.zeros(D_MODEL)

    # Cross-correlation between the two correlation profiles
    profile_corr = float(np.corrcoef(corr_value, corr_win)[0, 1]) if np.std(corr_win) > 1e-10 else 0.0

    # Top dims
    top_value_dims = np.argsort(np.abs(corr_value))[::-1][:10]
    top_win_dims   = np.argsort(np.abs(corr_win))[::-1][:10]

    print(f"\nPer-dim correlation CLS vs value/win:")
    print(f"  max |corr(dim, V)|:    {np.abs(corr_value).max():.4f}  (dim {np.argmax(np.abs(corr_value))})")
    print(f"  max |corr(dim, win)|:  {np.abs(corr_win).max():.4f}  (dim {np.argmax(np.abs(corr_win))})")
    print(f"  mean |corr(dim, V)|:   {np.abs(corr_value).mean():.4f}")
    print(f"  mean |corr(dim, win)|: {np.abs(corr_win).mean():.4f}")
    print(f"  profile corr(value_profile, win_profile): {profile_corr:.4f}")

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("IndependentCritic CLS -- per-dimension correlations", fontsize=12)

    # [0,0] Sorted |corr| with value
    ax = axes[0, 0]
    order_v = np.argsort(corr_value)[::-1]
    ax.bar(range(D_MODEL), corr_value[order_v], color="#1976D2", width=1.0)
    ax.set_xlabel("Dimension (sorted by corr)")
    ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs V(s)")
    ax.axhline(0, color="black", lw=0.5)

    # [0,1] Sorted |corr| with win
    ax = axes[0, 1]
    order_w = np.argsort(corr_win)[::-1]
    ax.bar(range(D_MODEL), corr_win[order_w], color="#388E3C", width=1.0)
    ax.set_xlabel("Dimension (sorted by corr)")
    ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs win label")
    ax.axhline(0, color="black", lw=0.5)

    # [1,0] Scatter: corr_value vs corr_win per dimension
    ax = axes[1, 0]
    ax.scatter(corr_value, corr_win, s=8, alpha=0.6, color="#7B1FA2")
    ax.set_xlabel("corr(dim, V)")
    ax.set_ylabel("corr(dim, win)")
    ax.set_title(f"Value vs Win correlation per dim  (profile r={profile_corr:.3f})")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    # Identity line
    lim = max(np.abs(corr_value).max(), np.abs(corr_win).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # [1,1] Histogram of |corr|
    ax = axes[1, 1]
    ax.hist(np.abs(corr_value), bins=30, alpha=0.6, label="|corr(dim, V)|", color="#1976D2")
    ax.hist(np.abs(corr_win),   bins=30, alpha=0.6, label="|corr(dim, win)|", color="#388E3C")
    ax.set_xlabel("|Pearson r|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of |correlation| per dimension")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_corr.png"), dpi=130, bbox_inches="tight")
    plt.savefig(out_path.with_name(out_path.stem + "_corr.pdf"), bbox_inches="tight")
    print(f"Correlation figure: {out_path.with_name(out_path.stem + '_corr.png')}")

    return {
        "max_abs_corr_value":  round(float(np.abs(corr_value).max()), 4),
        "max_abs_corr_win":    round(float(np.abs(corr_win).max()), 4),
        "mean_abs_corr_value": round(float(np.abs(corr_value).mean()), 4),
        "mean_abs_corr_win":   round(float(np.abs(corr_win).mean()), 4),
        "profile_corr":        round(profile_corr, 4),
        "top10_value_dims":    top_value_dims.tolist(),
        "top10_win_dims":      top_win_dims.tolist(),
        "corr_value":          corr_value.tolist(),
        "corr_win":            corr_win.tolist(),
    }


# ── Main probes figure (per-token bar charts) ──────────────────────────────

def save_probes_figure(results: dict, out_path: Path) -> None:
    """Bar charts of per-token probing results (critic), same layout as probing.py."""
    per = results["per_token"]
    cls = results["cls"]

    x = np.arange(SEQ_LEN)

    def _arr(key):
        return np.array([
            per[str(i)][key] if per[str(i)][key] is not None else np.nan
            for i in range(SEQ_LEN)
        ])

    ret_r2  = _arr("return_r2")
    win_auc = _arr("win_auc")
    hp_arr  = _arr("hp_r2")

    _slot_colors = ["#2196F3"] * 6 + ["#E53935"] * 6 + ["#9E9E9E"]
    bar_colors = _slot_colors * K_TURNS
    turn_seps = [N_SLOTS * t - 0.5 for t in range(1, K_TURNS)]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle("IndependentCritic -- per-token linear probes", fontsize=12)

    for ax in axes:
        for xs in turn_seps:
            ax.axvline(xs, color="gray", lw=0.8, ls="--", alpha=0.4)

    # Return R2
    ax = axes[0]
    ax.bar(x, ret_r2, color=bar_colors, width=0.8)
    ax.axhline(cls["return_r2"], color="purple", ls="--", lw=1.2, alpha=0.8,
               label=f'CLS={cls["return_r2"]:.3f}')
    ax.set_ylabel("Return R2")
    ax.set_title("Return R2 per token")
    ax.legend(fontsize=8)

    # Win AUC
    ax = axes[1]
    ax.bar(x, win_auc, color=bar_colors, width=0.8)
    ax.axhline(cls["win_auc"], color="purple", ls="--", lw=1.2, alpha=0.8,
               label=f'CLS={cls["win_auc"]:.3f}')
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_ylabel("Win AUC")
    ax.set_title("Win AUC per token")
    ax.legend(fontsize=8)

    # HP R2
    ax = axes[2]
    ax.bar(x, hp_arr, color=bar_colors, width=0.8)
    ax.set_ylabel("HP R2")
    ax.set_title("HP R2 per token")
    ax.set_xlabel("Token index")
    ax.set_xticks(x)
    ax.set_xticklabels(SEQ_TOKEN_LABELS, rotation=90, fontsize=5)

    fig.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Probes figure: {out_path.with_suffix('.png')}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Probing on IndependentCritic representations.")
    parser.add_argument("--cache",  required=True, help="Path to seq_all_cache.pt")
    parser.add_argument("--output", default="diag/probing_critic",
                        help="Output prefix (without extension)")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load cache ──────────────────────────────────────────────────────────
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)

    for key in ("critic_cls", "critic_seq", "critic_values"):
        if key not in cache:
            print(f"ERROR: '{key}' not in cache. Re-run make_token_cache.py with a checkpoint "
                  f"that has an independent critic.")
            sys.exit(1)

    critic_cls = cache["critic_cls"].numpy()       # [N, D_MODEL]
    critic_seq = cache["critic_seq"].numpy()       # [N, 52, D_MODEL]
    values     = cache["critic_values"].squeeze(-1).numpy()  # [N]
    N = critic_cls.shape[0]
    print(f"  N={N}  D_MODEL={critic_cls.shape[1]}")

    # Labels
    labels = {
        "y_return":  cache["y_return"],
        "y_win":     cache["y_win"],
        "y_type1":   cache["y_type1"],
        "y_item":    cache["y_item"],
        "y_ability": cache["y_ability"],
        "y_hp":      cache["y_hp"],
        "y_stats":   cache["y_stats"],
    }
    # Ensure numpy
    for k, v in labels.items():
        if isinstance(v, torch.Tensor):
            labels[k] = v.numpy()

    # ── Train/val split ─────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N)
    n_val = int(N * args.val_frac)
    val_idx   = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    # ── 1. Linear probes ────────────────────────────────────────────────────
    probe_results = run_critic_probes(critic_seq, critic_cls, labels, train_idx, val_idx)

    # ── 2. PCA ──────────────────────────────────────────────────────────────
    pca_results = save_pca_figure(critic_cls, values, labels["y_win"], out_path)

    # ── 3. Effective rank ───────────────────────────────────────────────────
    erank_results = run_effective_rank(critic_cls, critic_seq)

    # ── 4. Per-dim correlations ─────────────────────────────────────────────
    corr_results = run_dim_correlations(critic_cls, values, labels["y_win"], out_path)

    # ── 5. Save probes figure ───────────────────────────────────────────────
    save_probes_figure(probe_results, out_path)

    # ── Save JSON ───────────────────────────────────────────────────────────
    all_results = {
        "n_transitions": N,
        "probes":        probe_results,
        "pca":           pca_results,
        "effective_rank": erank_results,
        "dim_correlations": {
            k: v for k, v in corr_results.items()
            if k not in ("corr_value", "corr_win")  # skip large arrays in summary
        },
        "dim_correlations_full": {
            "corr_value": corr_results["corr_value"],
            "corr_win":   corr_results["corr_win"],
        },
    }
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON saved: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
