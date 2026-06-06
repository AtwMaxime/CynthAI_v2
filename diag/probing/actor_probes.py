"""
actor_probes — Linear probes on backbone (actor) Transformer tokens.

Analyses:
  1. Per-token probes (52 tokens): return R², win AUC, type1/item/ability acc, HP R², base stats R²
  2. Mean pool probes (current turn, all turns)
  3. Per-slot cross-turn pool probes
  4. Cross-token 12×12 matrix (current turn)
  5. Backbone CLS probes + PCA

Required cache keys: seq_all, y_return, y_win, y_type1, y_item, y_ability, y_hp, y_stats
Optional: backbone_cls, backbone_values
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
    extract_aggregate_labels, draw_turn_vlines,
    numpy_from_cache, get_labels_from_cache, check_labels,
)


# ─────────────────────────────────────────────────────────────────────────────
# Required cache keys
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"seq_all", "y_return", "y_win", "y_type1",
                 "y_item", "y_ability", "y_hp", "y_stats"}


def can_run(cache: dict) -> bool:
    return REQUIRED_KEYS.issubset(cache.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def _run_token_probes(seq_all, labels, train_idx, val_idx):
    """Run probes on all 52 tokens + mean pools + per-slot cross-turn pools."""
    seq_np = seq_all.numpy()

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

    results_per_token: dict[str, dict] = {}

    for tok_i in range(SEQ_LEN):
        t = tok_i // N_SLOTS
        s = tok_i % N_SLOTS
        poke_idx = t * 12 + s

        X = seq_np[:, tok_i, :]

        ret_r2, ret_corr = fit_regression(
            X[train_idx], y_return[train_idx], X[val_idx], y_return[val_idx])

        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = fit_classification(
                X[win_tr], y_win[win_tr].astype(np.int64),
                X[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True)
        else:
            win_acc, win_auc = 0.5, 0.5

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
            basestats_r2 = float(np.mean(stat_r2s))
            tok_res.update({
                "type1_acc": round(type1_acc, 4), "item_acc": round(item_acc, 4),
                "ability_acc": round(ability_acc, 4), "hp_r2": round(hp_r2, 4),
                "basestats_r2": round(basestats_r2, 4),
            })
            print(f"  {SEQ_TOKEN_LABELS[tok_i]:8s} | "
                  f"ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}  "
                  f"type1={type1_acc:.3f}  item={item_acc:.3f}  "
                  f"ability={ability_acc:.3f}  hp={hp_r2:.3f}  bstats={basestats_r2:.3f}")
        else:
            tok_res.update({
                "type1_acc": None, "item_acc": None, "ability_acc": None,
                "hp_r2": None, "basestats_r2": None,
            })
            print(f"  {SEQ_TOKEN_LABELS[tok_i]:8s} | "
                  f"ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")

        results_per_token[str(tok_i)] = tok_res

    # Mean pool probes
    def _probe_pool(X_pool, label):
        ret_r2, ret_corr = fit_regression(
            X_pool[train_idx], y_return[train_idx],
            X_pool[val_idx], y_return[val_idx])
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = fit_classification(
                X_pool[win_tr], y_win[win_tr].astype(np.int64),
                X_pool[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True)
        else:
            win_acc, win_auc = 0.5, 0.5
        print(f"  {label:20s} | ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")
        return {"return_r2": round(ret_r2, 4), "return_corr": round(ret_corr, 4),
                "win_acc": round(win_acc, 4), "win_auc": round(win_auc, 4)}

    X_mean_cur = seq_np[:, -N_SLOTS:, :].mean(axis=1)
    X_mean_all = seq_np.mean(axis=1)

    print("\nMean pool probes:")
    pool_cur = _probe_pool(X_mean_cur, "mean_pool_current")
    pool_all = _probe_pool(X_mean_all, "mean_pool_all")

    # Per-slot cross-turn pool probes
    per_slot_pool = {}
    print("\nPer-slot cross-turn pool probes:")
    for s in range(12):
        indices = [t * N_SLOTS + s for t in range(K_TURNS)]
        X_slot = seq_np[:, indices, :].mean(axis=1)
        poke_idx = (K_TURNS - 1) * 12 + s

        type1_acc, _ = fit_classification(
            X_slot[train_idx], y_type1[train_idx, poke_idx],
            X_slot[val_idx], y_type1[val_idx, poke_idx], max_iter=500)
        item_acc, _ = fit_classification(
            X_slot[train_idx], y_item[train_idx, poke_idx],
            X_slot[val_idx], y_item[val_idx, poke_idx], max_iter=1000)
        ability_acc, _ = fit_classification(
            X_slot[train_idx], y_ability[train_idx, poke_idx],
            X_slot[val_idx], y_ability[val_idx, poke_idx], max_iter=1000)
        hp_r2, _ = fit_regression(
            X_slot[train_idx], y_hp[train_idx, poke_idx],
            X_slot[val_idx], y_hp[val_idx, poke_idx])
        stat_r2s = []
        for stat in range(5):
            r2, _ = fit_regression(
                X_slot[train_idx], y_stats[train_idx, poke_idx, stat],
                X_slot[val_idx], y_stats[val_idx, poke_idx, stat])
            stat_r2s.append(r2)
        bstats_r2 = float(np.mean(stat_r2s))

        win_acc_s, win_auc_s = 0.5, 0.5
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc_s, win_auc_s = fit_classification(
                X_slot[win_tr], y_win[win_tr].astype(np.int64),
                X_slot[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True)

        per_slot_pool[SLOT_NAMES[s]] = {
            "type1_acc": round(type1_acc, 4), "item_acc": round(item_acc, 4),
            "ability_acc": round(ability_acc, 4), "hp_r2": round(hp_r2, 4),
            "basestats_r2": round(bstats_r2, 4),
            "win_acc": round(win_acc_s, 4), "win_auc": round(win_auc_s, 4),
        }
        print(f"  {SLOT_NAMES[s]:6s} | "
              f"type1={type1_acc:.3f}  item={item_acc:.3f}  "
              f"ability={ability_acc:.3f}  hp={hp_r2:.3f}  "
              f"bstats={bstats_r2:.3f}  win_auc={win_auc_s:.3f}")

    return {
        "per_token": results_per_token,
        "mean_pool_current": pool_cur,
        "mean_pool_all": pool_all,
        "per_slot_pool": per_slot_pool,
    }


def _run_cross_token_matrix(seq_all, labels, train_idx, val_idx):
    """12×12 cross-token probe matrix for current turn."""
    seq_np = seq_all.numpy()
    cur_turn_start = (K_TURNS - 1) * N_SLOTS
    poke_start = (K_TURNS - 1) * 12

    y_type1 = labels["y_type1"]
    y_item = labels["y_item"]
    y_ability = labels["y_ability"]
    y_hp = labels["y_hp"]
    y_stats = labels["y_stats"]

    n_slots = 12
    hp_r2_mat = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    type1_acc_mat = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    item_acc_mat = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    ability_acc_mat = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    bstats_r2_mat = np.full((n_slots, n_slots), np.nan, dtype=np.float32)

    total = n_slots * n_slots
    done = 0
    for i in range(n_slots):
        X = seq_np[:, cur_turn_start + i, :]
        X_tr = X[train_idx]
        X_val = X[val_idx]
        for j in range(n_slots):
            poke_idx = poke_start + j
            hp_r2_mat[i, j], _ = fit_regression(
                X_tr, y_hp[train_idx, poke_idx], X_val, y_hp[val_idx, poke_idx])
            type1_acc_mat[i, j], _ = fit_classification(
                X_tr, y_type1[train_idx, poke_idx], X_val, y_type1[val_idx, poke_idx], max_iter=500)
            item_acc_mat[i, j], _ = fit_classification(
                X_tr, y_item[train_idx, poke_idx], X_val, y_item[val_idx, poke_idx], max_iter=1000)
            ability_acc_mat[i, j], _ = fit_classification(
                X_tr, y_ability[train_idx, poke_idx], X_val, y_ability[val_idx, poke_idx], max_iter=1000)
            stat_r2s = []
            for stat in range(5):
                r2, _ = fit_regression(
                    X_tr, y_stats[train_idx, poke_idx, stat],
                    X_val, y_stats[val_idx, poke_idx, stat])
                stat_r2s.append(r2)
            bstats_r2_mat[i, j] = float(np.nanmean(stat_r2s))
            done += 1
        label = f"OWN{i}" if i < 6 else f"OPP{i-6}"
        print(f"  cross-token matrix: {done}/{total}  (source token {label} done)")

    return {
        "hp_r2_mat": hp_r2_mat.tolist(),
        "type1_acc_mat": type1_acc_mat.tolist(),
        "item_acc_mat": item_acc_mat.tolist(),
        "ability_acc_mat": ability_acc_mat.tolist(),
        "bstats_r2_mat": bstats_r2_mat.tolist(),
    }


def _run_cls_probes(backbone_cls, backbone_val, labels, train_idx, val_idx):
    """Linear probes on the backbone CLS token."""
    y_return = labels["y_return"]
    y_win = labels["y_win"]
    agg_labels = extract_aggregate_labels(labels)

    win_mask = y_win >= 0
    win_tr = [i for i in train_idx if win_mask[i]]
    win_val = [i for i in val_idx if win_mask[i]]

    X = backbone_cls
    print("\nBackbone CLS probes:")

    ret_r2, ret_corr = fit_regression(
        X[train_idx], y_return[train_idx], X[val_idx], y_return[val_idx])
    print(f"  return R2:   {ret_r2:+.4f}  corr: {ret_corr:.4f}")

    win_acc, win_auc = 0.5, 0.5
    if len(win_tr) > 20 and len(win_val) > 10:
        win_acc, win_auc = fit_classification(
            X[win_tr], y_win[win_tr].astype(np.int64),
            X[win_val], y_win[win_val].astype(np.int64),
            max_iter=500, compute_auc=True)
    print(f"  win AUC:     {win_auc:.4f}  acc: {win_acc:.4f}")

    agg_results = {}
    for key in ("mean_hp_own", "mean_hp_opp", "alive_own", "alive_opp"):
        y = agg_labels[key]
        r2, corr = fit_regression(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        agg_results[key + "_r2"] = round(r2, 4)
        print(f"  {key:15s} R2: {r2:+.4f}")

    erank = effective_rank(X)
    print(f"  effective rank: {erank:.2f} / {D_MODEL}")

    corr_value = per_dim_correlation(X, backbone_val)
    corr_win = np.zeros(D_MODEL)
    if win_mask.sum() > 50:
        corr_win = per_dim_correlation(X[win_mask], y_win[win_mask])

    profile_corr = 0.0
    if np.std(corr_value) > 1e-10 and np.std(corr_win) > 1e-10:
        profile_corr = float(np.corrcoef(corr_value, corr_win)[0, 1])

    return {
        "return_r2": round(ret_r2, 4), "return_corr": round(ret_corr, 4),
        "win_acc": round(win_acc, 4), "win_auc": round(win_auc, 4),
        **agg_results,
        "effective_rank": round(erank, 2),
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

def _save_token_figure(results, out_path):
    """Main 4×2 probing figure."""
    per = results["per_token"]
    pool_cur = results["mean_pool_current"]
    pool_all = results["mean_pool_all"]
    per_slot_pool = results["per_slot_pool"]

    x = np.arange(SEQ_LEN)

    def _arr(key):
        return np.array([per[str(i)][key] if per[str(i)][key] is not None else np.nan
                         for i in range(SEQ_LEN)])

    ret_r2_arr = _arr("return_r2")
    win_auc_arr = _arr("win_auc")
    type1_arr = _arr("type1_acc")
    item_arr = _arr("item_acc")
    ability_arr = _arr("ability_acc")
    hp_arr = _arr("hp_r2")
    bstats_arr = _arr("basestats_r2")

    xs12 = np.arange(12)
    slot_colors12 = ["#2196F3"] * 6 + ["#E53935"] * 6

    def _slot_arr(key):
        return np.array([per_slot_pool[n][key] for n in SLOT_NAMES], dtype=np.float32)

    sp_type1 = _slot_arr("type1_acc")
    sp_item = _slot_arr("item_acc")
    sp_ability = _slot_arr("ability_acc")
    sp_hp = _slot_arr("hp_r2")
    sp_bstats = _slot_arr("basestats_r2")
    sp_winauc = _slot_arr("win_auc")

    tick_kw = dict(rotation=90, ha="center", fontsize=5.5)

    def _heatmap(ax, data, cmap, vmin, vmax, ytick_labels, title):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        n_rows = data.shape[0]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(SEQ_TOKEN_LABELS, rotation=90, ha="center", fontsize=5.5)
        ax.set_title(title, fontsize=9)
        for xs in TURN_SEPS:
            ax.axvline(xs, color="white", lw=1.8, alpha=0.8)
        return im

    fig = plt.figure(figsize=(22, 15))
    fig.suptitle("CynthAI_v2 — Linear Probing (cheater, full info, 52 tokens × 4 turns)", fontsize=12)
    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.25)

    # [0,0] Heatmap: win AUC
    ax00 = fig.add_subplot(gs[0, 0])
    im00 = _heatmap(ax00, win_auc_arr.reshape(1, -1), cmap="viridis", vmin=0.5, vmax=1.0,
                    ytick_labels=["win AUC"], title="Win AUC — heatmap (vmin=0.5)")
    fig.colorbar(im00, ax=ax00, fraction=0.03, pad=0.04)

    # [0,1] Bar: return R²
    ax01 = fig.add_subplot(gs[0, 1])
    ylim_lo, ylim_hi = -0.12, 0.05
    clipped = ret_r2_arr < ylim_lo
    bar_vals = np.where(clipped, ylim_lo, ret_r2_arr)
    ax01.bar(x, bar_vals, color=BAR_COLORS_52, edgecolor="none", width=0.8)
    for xi, clip in zip(x, clipped):
        if clip:
            ax01.text(xi, ylim_lo + 0.005, "*", ha="center", va="bottom", fontsize=5, color="black")
    ax01.axhline(0, color="black", lw=0.6, ls="-", alpha=0.4)
    ax01.axhline(pool_cur["return_r2"], color="#FF9800", ls="--", lw=1.5,
                 label=f"mp_cur={pool_cur['return_r2']:.3f}")
    ax01.axhline(pool_all["return_r2"], color="#9C27B0", ls=":", lw=1.5,
                 label=f"mp_all={pool_all['return_r2']:.3f}")
    draw_turn_vlines(ax01)
    ax01.set_xticks(x); ax01.set_xticklabels(SEQ_TOKEN_LABELS, **tick_kw)
    ax01.set_ylabel("R²"); ax01.set_title("Return R² per token  (* = clipped < -0.12)", fontsize=9)
    ax01.set_ylim(ylim_lo, ylim_hi); ax01.legend(fontsize=7)

    # [1,0] Bar: win AUC
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.bar(x, win_auc_arr, color=BAR_COLORS_52, edgecolor="none", width=0.8)
    ax10.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance=0.5")
    ax10.axhline(pool_cur["win_auc"], color="#FF9800", ls="--", lw=1.5,
                 label=f"mp_cur={pool_cur['win_auc']:.3f}")
    ax10.axhline(pool_all["win_auc"], color="#9C27B0", ls=":", lw=1.5,
                 label=f"mp_all={pool_all['win_auc']:.3f}")
    draw_turn_vlines(ax10)
    ax10.set_xticks(x); ax10.set_xticklabels(SEQ_TOKEN_LABELS, **tick_kw)
    ax10.set_ylabel("AUC-ROC"); ax10.set_title("Win AUC per token", fontsize=9)
    ax10.set_ylim(0.5, 1.0); ax10.legend(fontsize=7)

    # [1,1] Heatmap: type1/item/ability
    ax11 = fig.add_subplot(gs[1, 1])
    hmap_id = np.array([type1_arr, item_arr, ability_arr])
    im11 = _heatmap(ax11, hmap_id, cmap="plasma", vmin=0.9, vmax=1.0,
                    ytick_labels=["type1", "item", "ability"],
                    title="Type1 / Item / Ability accuracy  (vmin=0.9)")
    fig.colorbar(im11, ax=ax11, fraction=0.03, pad=0.04)

    # [2,0] Heatmap: HP R²
    ax20 = fig.add_subplot(gs[2, 0])
    im20 = _heatmap(ax20, hp_arr.reshape(1, -1), cmap="plasma", vmin=0.0, vmax=1.0,
                    ytick_labels=["hp_ratio R²"], title="HP ratio R² per token")
    fig.colorbar(im20, ax=ax20, fraction=0.03, pad=0.04)

    # [2,1] Heatmap: base stats R²
    ax21 = fig.add_subplot(gs[2, 1])
    im21 = _heatmap(ax21, bstats_arr.reshape(1, -1), cmap="plasma", vmin=0.0, vmax=1.0,
                    ytick_labels=["bstats R²"],
                    title="Base stats R² per token (mean over 5 stats)")
    fig.colorbar(im21, ax=ax21, fraction=0.03, pad=0.04)

    # [3,0] Cross-turn slot pool — ID
    ax30 = fig.add_subplot(gs[3, 0])
    hmap_slot_id = np.array([sp_type1, sp_item, sp_ability])
    im30 = ax30.imshow(hmap_slot_id, aspect="auto", cmap="plasma",
                       vmin=0.9, vmax=1.0, interpolation="nearest")
    ax30.set_yticks([0, 1, 2])
    ax30.set_yticklabels(["type1", "item", "ability"], fontsize=8)
    ax30.set_xticks(xs12)
    ax30.set_xticklabels(SLOT_NAMES, rotation=45, ha="right", fontsize=7)
    ax30.axvline(5.5, color="white", lw=1.8, alpha=0.8)
    ax30.set_title("Cross-turn slot pool — ID accuracy (vmin=0.9)", fontsize=9)
    fig.colorbar(im30, ax=ax30, fraction=0.03, pad=0.04)

    # [3,1] Cross-turn slot pool — HP/bstats/win
    ax31 = fig.add_subplot(gs[3, 1])
    w = 0.25
    ax31.bar(xs12 - w, sp_hp, width=w, color="#00BCD4", label="hp R²")
    ax31.bar(xs12, sp_bstats, width=w, color="#8BC34A", label="bstats R²")
    ax31.bar(xs12 + w, sp_winauc, width=w, color="#FF5722", label="win AUC")
    ax31.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax31.axvline(5.5, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax31.set_xticks(xs12)
    ax31.set_xticklabels(SLOT_NAMES, rotation=45, ha="right", fontsize=7)
    ax31.set_ylim(0.0, 1.05)
    ax31.set_title("Cross-turn slot pool — HP R², bstats R², win AUC", fontsize=9)
    ax31.legend(fontsize=7)

    plt.savefig(out_path / "actor_probes.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "actor_probes.pdf", bbox_inches="tight")
    print(f"  Actor probes figure: {out_path / 'actor_probes.png'}")
    plt.close(fig)


def _save_cross_matrix_figure(cross, out_path):
    """2×3 cross-token matrix figure."""
    SLOT_LABELS = [f"OWN{s}" if s < 6 else f"OPP{s-6}" for s in range(12)]
    def _mat(key): return np.array(cross[key], dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("CynthAI_v2 — Cross-token probe matrix  (current turn, 12×12)", fontsize=11)
    sep = 5.5

    def _draw(ax, data, title, vmin, vmax, cmap="plasma"):
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest", origin="upper")
        ax.set_xticks(range(12)); ax.set_xticklabels(SLOT_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(12)); ax.set_yticklabels(SLOT_LABELS, fontsize=7)
        ax.set_xlabel("target slot attribute", fontsize=7)
        ax.set_ylabel("source token", fontsize=7)
        ax.axhline(sep, color="white", lw=1.5, alpha=0.8)
        ax.axvline(sep, color="white", lw=1.5, alpha=0.8)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw(axes[0, 0], _mat("hp_r2_mat"), "HP ratio R²", 0.0, 1.0)
    _draw(axes[0, 1], _mat("type1_acc_mat"), "Type1 accuracy", 0.0, 1.0)
    _draw(axes[0, 2], _mat("item_acc_mat"), "Item accuracy", 0.0, 1.0)
    _draw(axes[1, 0], _mat("ability_acc_mat"), "Ability accuracy", 0.0, 1.0)
    _draw(axes[1, 1], _mat("bstats_r2_mat"), "Base stats R² (mean×5)", 0.0, 1.0)
    axes[1, 2].axis("off")

    fig.tight_layout()
    plt.savefig(out_path / "actor_cross_matrix.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "actor_cross_matrix.pdf", bbox_inches="tight")
    print(f"  Cross-token matrix: {out_path / 'actor_cross_matrix.png'}")
    plt.close(fig)


def _save_cls_figure(backbone_cls, backbone_val, y_win, cls_results, out_path):
    """Backbone CLS PCA + correlation figure."""
    X_c = backbone_cls - backbone_cls.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n_pc = min(50, len(S))
    var = S ** 2 / (backbone_cls.shape[0] - 1)
    evr = var / var.sum()
    proj = U[:, :n_pc] * S[:n_pc]
    win_mask = y_win >= 0

    corr_value = np.array(cls_results["corr_value"])
    corr_win = np.array(cls_results["corr_win"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Backbone CLS token -- probing & PCA", fontsize=12)

    ax = axes[0, 0]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=backbone_val, cmap="coolwarm", s=4, alpha=0.5, edgecolors="none")
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

    ax = axes[0, 2]
    n_show = min(30, len(evr))
    ax.bar(range(n_show), evr[:n_show] * 100, color="#42A5F5")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Explained variance (top {n_show} PCs)")
    ax.set_xlim(-0.5, n_show - 0.5)

    ax = axes[1, 0]
    order_v = np.argsort(corr_value)[::-1]
    ax.bar(range(D_MODEL), corr_value[order_v], color="#1976D2", width=1.0)
    ax.set_xlabel("Dimension (sorted)"); ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs V(s)"); ax.axhline(0, color="black", lw=0.5)

    ax = axes[1, 1]
    order_w = np.argsort(corr_win)[::-1]
    ax.bar(range(D_MODEL), corr_win[order_w], color="#388E3C", width=1.0)
    ax.set_xlabel("Dimension (sorted)"); ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs win label"); ax.axhline(0, color="black", lw=0.5)

    ax = axes[1, 2]
    ax.scatter(corr_value, corr_win, s=8, alpha=0.6, color="#7B1FA2")
    ax.set_xlabel("corr(dim, V)"); ax.set_ylabel("corr(dim, win)")
    ax.set_title(f"Value vs Win corr per dim (profile r={cls_results['profile_corr']:.3f})")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5); ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    lim = max(np.abs(corr_value).max(), np.abs(corr_win).max()) * 1.1
    if lim > 0:
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    fig.tight_layout()
    plt.savefig(out_path / "actor_cls.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "actor_cls.pdf", bbox_inches="tight")
    print(f"  CLS figure: {out_path / 'actor_cls.png'}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cache: dict, train_idx: list[int], val_idx: list[int], out_dir: Path) -> dict:
    """Run all actor/backbone probes. Returns results dict."""
    if not can_run(cache):
        missing = REQUIRED_KEYS - set(cache.keys())
        print(f"  [SKIP] actor_probes: missing keys {missing}")
        return {}

    seq_all = cache["seq_all"]
    labels = get_labels_from_cache(cache)
    N = seq_all.shape[0]

    print(f"\n{'='*60}")
    print("ACTOR PROBES (backbone tokens)")
    print(f"{'='*60}")

    # 1. Token probes
    print(f"\nRunning probes ({SEQ_LEN} tokens + pools) ...")
    results = _run_token_probes(seq_all, labels, train_idx, val_idx)

    # 2. Cross-token matrix
    print(f"\nRunning cross-token matrix (12×12) ...")
    cross = _run_cross_token_matrix(seq_all, labels, train_idx, val_idx)
    results["cross_token_matrix"] = cross

    # 3. Backbone CLS probes
    backbone_cls = numpy_from_cache(cache, "backbone_cls", np.float32)
    backbone_val = numpy_from_cache(cache, "backbone_values", np.float32)
    if backbone_cls is not None and backbone_val is not None:
        backbone_val = backbone_val.squeeze(-1)
        cls_results = _run_cls_probes(backbone_cls, backbone_val, labels, train_idx, val_idx)
        cls_summary = {k: v for k, v in cls_results.items() if k not in ("corr_value", "corr_win")}
        results["backbone_cls"] = cls_summary
        _save_cls_figure(backbone_cls, backbone_val, labels["y_win"], cls_results, out_dir)
    else:
        print("  (no backbone_cls in cache — skipping CLS probes)")

    # 4. Figures
    _save_token_figure(results, out_dir)
    _save_cross_matrix_figure(cross, out_dir)

    # 5. JSON
    json_path = out_dir / "actor_probes.json"
    with open(json_path, "w") as f:
        json.dump({"n_transitions": N, **results}, f, indent=2)
    print(f"  JSON: {json_path}")

    return results
