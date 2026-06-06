"""
detr_probes — Linear probes on DETR queries from backbone.act().

Probes:
  1. Action chosen  (per-query AUC, mean-pool 13-class accuracy)
  2. Win probability (per-query, mean-pool, chosen-query AUC)
  3. dHP next turn   (Ridge R², active slot 0, no-switch filtered)
  4. KO next turn    (AUC, active slot 0, no-switch filtered)

Required cache keys: detr_queries, actions, next_hp_own, next_hp_opp,
                     no_switch_valid, opp_no_switch_valid
Optional: y_win, cur_hp_own, cur_hp_opp
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diag.probing._common import (
    QUERY_LABELS, QUERY_COLORS,
    fit_regression, fit_classification,
    numpy_from_cache,
)


REQUIRED_KEYS = {"detr_queries", "actions", "next_hp_own", "next_hp_opp",
                 "no_switch_valid", "opp_no_switch_valid"}


def can_run(cache: dict) -> bool:
    return REQUIRED_KEYS.issubset(cache.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers (thin wrappers for DETR-specific patterns)
# ─────────────────────────────────────────────────────────────────────────────

def _ridge_r2(X_tr, y_tr, X_val, y_val):
    r2, _ = fit_regression(X_tr, y_tr, X_val, y_val)
    return r2


def _logistic_auc(X_tr, y_tr, X_val, y_val, max_iter=500):
    _, auc = fit_classification(X_tr, y_tr, X_val, y_val, max_iter=max_iter, compute_auc=True)
    return auc


def _logistic_acc(X_tr, y_tr, X_val, y_val, max_iter=500):
    acc, _ = fit_classification(X_tr, y_tr, X_val, y_val, max_iter=max_iter)
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def _probe_action(detr_np, actions, train_idx, val_idx):
    per_query_auc = []
    for i in range(13):
        X = detr_np[:, i, :]
        y = (actions == i).astype(np.int64)
        auc = _logistic_auc(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
        per_query_auc.append(round(auc, 4))

    X_mean = detr_np.mean(axis=1)
    acc = _logistic_acc(X_mean[train_idx], actions[train_idx],
                        X_mean[val_idx], actions[val_idx])
    return {"per_query_auc": per_query_auc, "mean_pool_top1_acc": round(acc, 4)}


def _probe_win(detr_np, actions, y_win, train_idx, val_idx):
    win_mask = y_win >= 0
    win_tr = [i for i in train_idx if win_mask[i]]
    win_val = [i for i in val_idx if win_mask[i]]

    def _auc(X):
        if len(win_tr) < 20 or len(win_val) < 10:
            return 0.5
        return _logistic_auc(X[win_tr], y_win[win_tr].astype(np.int64),
                             X[win_val], y_win[win_val].astype(np.int64))

    per_query_auc = [round(_auc(detr_np[:, i, :]), 4) for i in range(13)]
    mean_pool_auc = round(_auc(detr_np.mean(axis=1)), 4)
    chosen = detr_np[np.arange(len(actions)), actions, :]
    chosen_auc = round(_auc(chosen), 4)

    return {"per_query_auc": per_query_auc, "mean_pool_auc": mean_pool_auc,
            "chosen_query_auc": chosen_auc}


def _probe_delta_hp(detr_np, actions, cur_hp_own, cur_hp_opp,
                    next_hp_own, next_hp_opp, no_switch_valid,
                    opp_no_switch_valid, train_idx, val_idx):
    own_tr = [i for i in train_idx if no_switch_valid[i]]
    own_val = [i for i in val_idx if no_switch_valid[i]]
    delta_own_active = next_hp_own[:, 0] - cur_hp_own[:, 0]

    opp_tr = [i for i in train_idx if opp_no_switch_valid[i]]
    opp_val = [i for i in val_idx if opp_no_switch_valid[i]]
    delta_opp_active = next_hp_opp[:, 0] - cur_hp_opp[:, 0]

    def _r2_own(X):
        if len(own_tr) < 20 or len(own_val) < 10: return float("nan")
        return round(_ridge_r2(X[own_tr], delta_own_active[own_tr],
                               X[own_val], delta_own_active[own_val]), 4)

    def _r2_opp(X):
        if len(opp_tr) < 20 or len(opp_val) < 10: return float("nan")
        return round(_ridge_r2(X[opp_tr], delta_opp_active[opp_tr],
                               X[opp_val], delta_opp_active[opp_val]), 4)

    per_query_own = [_r2_own(detr_np[:, i, :]) for i in range(13)]
    per_query_opp = [_r2_opp(detr_np[:, i, :]) for i in range(13)]

    X_mean = detr_np.mean(axis=1)
    X_chosen = detr_np[np.arange(len(actions)), actions, :]

    own_d = delta_own_active[no_switch_valid]
    opp_d = delta_opp_active[opp_no_switch_valid]

    return {
        "per_query_own_r2": per_query_own, "per_query_opp_r2": per_query_opp,
        "mean_pool_own_r2": _r2_own(X_mean), "mean_pool_opp_r2": _r2_opp(X_mean),
        "chosen_own_r2": _r2_own(X_chosen), "chosen_opp_r2": _r2_opp(X_chosen),
        "n_own_valid": int(no_switch_valid.sum()), "n_opp_valid": int(opp_no_switch_valid.sum()),
        "label_own_mean": round(float(own_d.mean()), 4), "label_own_std": round(float(own_d.std()), 4),
        "label_opp_mean": round(float(opp_d.mean()), 4), "label_opp_std": round(float(opp_d.std()), 4),
    }


def _probe_ko(detr_np, actions, cur_hp_own, cur_hp_opp,
              next_hp_own, next_hp_opp, no_switch_valid,
              opp_no_switch_valid, train_idx, val_idx):
    own_tr = [i for i in train_idx if no_switch_valid[i]]
    own_val = [i for i in val_idx if no_switch_valid[i]]
    opp_tr = [i for i in train_idx if opp_no_switch_valid[i]]
    opp_val = [i for i in val_idx if opp_no_switch_valid[i]]

    ko_own = ((cur_hp_own[:, 0] > 0.01) & (next_hp_own[:, 0] < 0.01)).astype(np.int64)
    ko_opp = ((cur_hp_opp[:, 0] > 0.01) & (next_hp_opp[:, 0] < 0.01)).astype(np.int64)

    def _auc_own(X):
        if len(own_tr) < 20 or len(own_val) < 10 or ko_own[own_tr].sum() < 2:
            return float("nan")
        return round(_logistic_auc(X[own_tr], ko_own[own_tr], X[own_val], ko_own[own_val]), 4)

    def _auc_opp(X):
        if len(opp_tr) < 20 or len(opp_val) < 10 or ko_opp[opp_tr].sum() < 2:
            return float("nan")
        return round(_logistic_auc(X[opp_tr], ko_opp[opp_tr], X[opp_val], ko_opp[opp_val]), 4)

    per_query_opp = [_auc_opp(detr_np[:, i, :]) for i in range(13)]
    per_query_own = [_auc_own(detr_np[:, i, :]) for i in range(13)]

    X_mean = detr_np.mean(axis=1)
    X_chosen = detr_np[np.arange(len(actions)), actions, :]

    return {
        "per_query_opp_auc": per_query_opp, "per_query_own_auc": per_query_own,
        "mean_pool_opp_auc": _auc_opp(X_mean), "mean_pool_own_auc": _auc_own(X_mean),
        "chosen_opp_auc": _auc_opp(X_chosen), "chosen_own_auc": _auc_own(X_chosen),
        "n_ko_own": int(ko_own[no_switch_valid].sum()),
        "n_ko_opp": int(ko_opp[opp_no_switch_valid].sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def _save_figure(results, out_path):
    act = results["probe_action"]
    win = results["probe_win"]
    dhp = results["probe_delta_hp"]
    ko = results["probe_ko"]
    x = np.arange(13)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CynthAI_v2 — DETR Query Probes (backbone.act() cross-attention output)", fontsize=11)

    ax = axes[0, 0]
    ax.bar(x, act["per_query_auc"], color=QUERY_COLORS, edgecolor="none")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance=0.5")
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.05); ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Probe 1 — Is query_i the chosen action?\n"
                 f"mean_pool 13-class acc={act['mean_pool_top1_acc']:.3f}", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.bar(x, win["per_query_auc"], color=QUERY_COLORS, edgecolor="none")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance=0.5")
    ax.axhline(win["mean_pool_auc"], color="#FF9800", ls="--", lw=1.5,
               label=f"mean_pool={win['mean_pool_auc']:.3f}")
    ax.axhline(win["chosen_query_auc"], color="#E53935", ls=":", lw=1.5,
               label=f"chosen_q={win['chosen_query_auc']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.05); ax.set_ylabel("AUC-ROC")
    ax.set_title("Probe 2 — Win probability", fontsize=9); ax.legend(fontsize=7)

    ax = axes[0, 2]
    w = 0.35
    opp_r2 = np.array(dhp["per_query_opp_r2"], dtype=float)
    own_r2 = np.array(dhp["per_query_own_r2"], dtype=float)
    ax.bar(x - w/2, opp_r2, width=w, color="#E53935", label="dHP_opp", edgecolor="none")
    ax.bar(x + w/2, own_r2, width=w, color="#2196F3", alpha=0.6, label="dHP_own", edgecolor="none")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²"); ax.set_title("Probe 3 — dHP active slot R²", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.bar(x, own_r2, color=QUERY_COLORS, edgecolor="none")
    ax.axhline(dhp["mean_pool_own_r2"], color="#2196F3", ls="--", lw=1.2,
               label=f"mean_pool={dhp['mean_pool_own_r2']:.3f}")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²"); ax.set_title("dHP_own R² per query", fontsize=9); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar(x, opp_r2, color=QUERY_COLORS, edgecolor="none")
    ax.axhline(dhp["mean_pool_opp_r2"], color="#E53935", ls="--", lw=1.2,
               label=f"mean_pool={dhp['mean_pool_opp_r2']:.3f}")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²"); ax.set_title("dHP_opp R² per query", fontsize=9); ax.legend(fontsize=7)

    ax = axes[1, 2]
    ko_opp_arr = np.array(ko["per_query_opp_auc"], dtype=float)
    ko_own_arr = np.array(ko["per_query_own_auc"], dtype=float)
    ax.bar(x - w/2, ko_opp_arr, width=w, color="#E53935", label="KO_opp", edgecolor="none")
    ax.bar(x + w/2, ko_own_arr, width=w, color="#2196F3", alpha=0.7, label="KO_own", edgecolor="none")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.05); ax.set_ylabel("AUC-ROC")
    ax.set_title("Probe 4 — KO next turn AUC", fontsize=9); ax.legend(fontsize=7)

    fig.tight_layout()
    plt.savefig(out_path / "detr_probes.png", dpi=130, bbox_inches="tight")
    plt.savefig(out_path / "detr_probes.pdf", bbox_inches="tight")
    print(f"  DETR probes figure: {out_path / 'detr_probes.png'}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(cache: dict, train_idx: list[int], val_idx: list[int], out_dir: Path) -> dict:
    if not can_run(cache):
        missing = REQUIRED_KEYS - set(cache.keys())
        print(f"  [SKIP] detr_probes: missing keys {missing}")
        return {}

    detr_np = numpy_from_cache(cache, "detr_queries", np.float32)
    actions = numpy_from_cache(cache, "actions", np.int64)
    next_hp_own = numpy_from_cache(cache, "next_hp_own", np.float32)
    next_hp_opp = numpy_from_cache(cache, "next_hp_opp", np.float32)
    no_switch_valid = numpy_from_cache(cache, "no_switch_valid", bool)
    opp_no_switch_valid = numpy_from_cache(cache, "opp_no_switch_valid", bool)

    N = len(actions)
    y_win = numpy_from_cache(cache, "y_win", np.float32)
    if y_win is None:
        y_win = np.full(N, -1.0, dtype=np.float32)

    cur_hp_own = numpy_from_cache(cache, "cur_hp_own", np.float32)
    cur_hp_opp = numpy_from_cache(cache, "cur_hp_opp", np.float32)
    if cur_hp_own is None or cur_hp_opp is None:
        print("  [warn] cur_hp not in cache — using zeros for dHP probes")
        cur_hp_own = np.zeros((N, 6), dtype=np.float32)
        cur_hp_opp = np.zeros((N, 6), dtype=np.float32)

    print(f"\n{'='*60}")
    print("DETR PROBES (backbone.act() queries)")
    print(f"{'='*60}")

    print("\nProbe 1 — action chosen ...")
    res_action = _probe_action(detr_np, actions, train_idx, val_idx)
    print(f"  mean_pool_top1_acc: {res_action['mean_pool_top1_acc']:.3f}")

    print("\nProbe 2 — win probability ...")
    res_win = _probe_win(detr_np, actions, y_win, train_idx, val_idx)
    print(f"  mean_pool_auc={res_win['mean_pool_auc']:.3f}  chosen_q={res_win['chosen_query_auc']:.3f}")

    print("\nProbe 3 — dHP next turn ...")
    res_dhp = _probe_delta_hp(detr_np, actions, cur_hp_own, cur_hp_opp,
                              next_hp_own, next_hp_opp, no_switch_valid,
                              opp_no_switch_valid, train_idx, val_idx)

    print("\nProbe 4 — KO next turn ...")
    res_ko = _probe_ko(detr_np, actions, cur_hp_own, cur_hp_opp,
                       next_hp_own, next_hp_opp, no_switch_valid,
                       opp_no_switch_valid, train_idx, val_idx)

    output = {
        "n_transitions": N,
        "probe_action": res_action,
        "probe_win": res_win,
        "probe_delta_hp": res_dhp,
        "probe_ko": res_ko,
    }

    _save_figure(output, out_dir)

    json_path = out_dir / "detr_probes.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  JSON: {json_path}")

    return output
