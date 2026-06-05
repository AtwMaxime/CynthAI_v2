"""
probing_detr.py — Linear probes on DETR queries from backbone.act().

Probes the 13 cross-attention output vectors (attn_out, shape [B, 13, D_MODEL])
that are produced AFTER action×context cross-attention and BEFORE the final
action_score linear layer.

Probes:
  1. Action chosen  — does query i self-identify as the chosen action? (AUC-ROC per query)
                      mean_pool → 13-class accuracy
  2. Win probability — does query i / mean_pool / chosen_query encode the outcome?
  3. dHP next turn  — Ridge R² for each of 12 HP deltas (own × 6, opp × 6)
  4. KO next turn   — does query i predict a KO event? (AUC-ROC)

Usage:
    # After running make_token_cache.py to generate the extended cache:
    python diag/probing_detr.py --cache diag/seq_all_cache.pt

Output:
    diag/probing_detr.png / .pdf
    diag/probing_detr.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers (self-contained — no sklearn import from probing.py needed)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ridge(X_tr, y_tr, X_val, y_val) -> float:
    """Ridge regression → val R²."""
    sc = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)
    pred   = model.predict(X_val_s)
    ss_res = float(np.sum((y_val - pred) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    if ss_tot < 1e-6:
        return float("nan")
    return max(-1.0, float(1.0 - ss_res / ss_tot))


def _fit_logistic_auc(X_tr, y_tr, X_val, y_val, max_iter=500) -> float:
    """Binary logistic regression → AUC-ROC. Returns 0.5 on failure."""
    sc = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    if len(np.unique(y_tr)) < 2:
        return 0.5
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_tr_s, y_tr)
    try:
        prob = model.predict_proba(X_val_s)[:, 1]
        return float(roc_auc_score(y_val, prob))
    except Exception:
        return 0.5


def _fit_logistic_acc(X_tr, y_tr, X_val, y_val, max_iter=500) -> float:
    """Multi-class logistic regression → top-1 accuracy."""
    sc = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)
    if len(np.unique(y_tr)) < 2:
        return float(np.mean(y_val == y_tr[0]))
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_tr_s, y_tr)
    return float(np.mean(model.predict(X_val_s) == y_val))


# ─────────────────────────────────────────────────────────────────────────────
# Probe 1 — Action chosen
# ─────────────────────────────────────────────────────────────────────────────

def probe_action(
    detr_np: np.ndarray,   # [N, 13, D]
    actions: np.ndarray,   # [N]  int64
    train_idx, val_idx,
) -> dict:
    """
    Per query i: binary AUC — is query i the chosen action?
    Mean-pool over 13 queries: 13-class top-1 accuracy.
    """
    per_query_auc = []
    for i in range(13):
        X = detr_np[:, i, :]
        y = (actions == i).astype(np.int64)
        auc = _fit_logistic_auc(
            X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
        )
        per_query_auc.append(round(auc, 4))

    X_mean = detr_np.mean(axis=1)
    acc = _fit_logistic_acc(
        X_mean[train_idx], actions[train_idx],
        X_mean[val_idx],   actions[val_idx],
    )
    return {"per_query_auc": per_query_auc, "mean_pool_top1_acc": round(acc, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Probe 2 — Win probability
# ─────────────────────────────────────────────────────────────────────────────

def probe_win(
    detr_np:  np.ndarray,   # [N, 13, D]
    actions:  np.ndarray,   # [N]
    y_win:    np.ndarray,   # [N]  float, -1 = unknown
    train_idx, val_idx,
) -> dict:
    win_mask = y_win >= 0
    win_tr  = [i for i in train_idx if win_mask[i]]
    win_val = [i for i in val_idx   if win_mask[i]]

    def _auc(X):
        if len(win_tr) < 20 or len(win_val) < 10:
            return 0.5
        return _fit_logistic_auc(
            X[win_tr],  y_win[win_tr].astype(np.int64),
            X[win_val], y_win[win_val].astype(np.int64),
        )

    per_query_auc = [round(_auc(detr_np[:, i, :]), 4) for i in range(13)]
    mean_pool_auc  = round(_auc(detr_np.mean(axis=1)), 4)

    # chosen_query: query at the action slot for each sample
    chosen = detr_np[np.arange(len(actions)), actions, :]   # [N, D]
    chosen_auc = round(_auc(chosen), 4)

    return {
        "per_query_auc":  per_query_auc,
        "mean_pool_auc":  mean_pool_auc,
        "chosen_query_auc": chosen_auc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe 3 — dHP next turn
# ─────────────────────────────────────────────────────────────────────────────

def probe_delta_hp(
    detr_np:           np.ndarray,   # [N, 13, D]
    actions:           np.ndarray,   # [N]
    cur_hp_own:        np.ndarray,   # [N, 6]
    cur_hp_opp:        np.ndarray,   # [N, 6]
    next_hp_own:       np.ndarray,   # [N, 6]  NaN where invalid
    next_hp_opp:       np.ndarray,   # [N, 6]  NaN where invalid
    no_switch_valid:   np.ndarray,   # [N]  bool — own no-switch & has next step
    opp_no_switch_valid: np.ndarray, # [N]  bool — above + opp likely didn't switch
    train_idx, val_idx,
) -> dict:
    """
    Probe only the ACTIVE mon HP delta (slot 0), using clean masks:
      - own: no_switch_valid  (own action was a move/mech, not a switch)
      - opp: opp_no_switch_valid  (additionally filters likely opp switches)

    Slot 1-5 are excluded because bench slot order resets on every switch,
    making those deltas dominated by 'which mon moved where' noise.
    """
    # Own active slot 0 — own didn't switch
    own_tr  = [i for i in train_idx if no_switch_valid[i]]
    own_val = [i for i in val_idx   if no_switch_valid[i]]
    delta_own_active = next_hp_own[:, 0] - cur_hp_own[:, 0]   # [N]

    # Opp active slot 0 — neither side switched
    opp_tr  = [i for i in train_idx if opp_no_switch_valid[i]]
    opp_val = [i for i in val_idx   if opp_no_switch_valid[i]]
    delta_opp_active = next_hp_opp[:, 0] - cur_hp_opp[:, 0]   # [N]

    def _r2_own(X):
        if len(own_tr) < 20 or len(own_val) < 10:
            return float("nan")
        return round(_fit_ridge(X[own_tr], delta_own_active[own_tr],
                                X[own_val], delta_own_active[own_val]), 4)

    def _r2_opp(X):
        if len(opp_tr) < 20 or len(opp_val) < 10:
            return float("nan")
        return round(_fit_ridge(X[opp_tr], delta_opp_active[opp_tr],
                                X[opp_val], delta_opp_active[opp_val]), 4)

    per_query_own = []
    per_query_opp = []
    for i in range(13):
        X = detr_np[:, i, :]
        per_query_own.append(_r2_own(X))
        per_query_opp.append(_r2_opp(X))

    X_mean   = detr_np.mean(axis=1)
    X_chosen = detr_np[np.arange(len(actions)), actions, :]

    # Stats on the label distributions
    own_d = delta_own_active[no_switch_valid]
    opp_d = delta_opp_active[opp_no_switch_valid]

    return {
        "per_query_own_r2":   per_query_own,
        "per_query_opp_r2":   per_query_opp,
        "mean_pool_own_r2":   _r2_own(X_mean),
        "mean_pool_opp_r2":   _r2_opp(X_mean),
        "chosen_own_r2":      _r2_own(X_chosen),
        "chosen_opp_r2":      _r2_opp(X_chosen),
        "n_own_valid":        int(no_switch_valid.sum()),
        "n_opp_valid":        int(opp_no_switch_valid.sum()),
        "label_own_mean":     round(float(own_d.mean()), 4),
        "label_own_std":      round(float(own_d.std()),  4),
        "label_opp_mean":     round(float(opp_d.mean()), 4),
        "label_opp_std":      round(float(opp_d.std()),  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe 4 — KO next turn
# ─────────────────────────────────────────────────────────────────────────────

def probe_ko(
    detr_np:           np.ndarray,   # [N, 13, D]
    actions:           np.ndarray,   # [N]
    cur_hp_own:        np.ndarray,   # [N, 6]
    cur_hp_opp:        np.ndarray,   # [N, 6]
    next_hp_own:       np.ndarray,   # [N, 6]
    next_hp_opp:       np.ndarray,   # [N, 6]
    no_switch_valid:   np.ndarray,   # [N]  bool
    opp_no_switch_valid: np.ndarray, # [N]  bool
    train_idx, val_idx,
) -> dict:
    """
    KO probe uses the same clean masks as dHP:
      - own KO: no_switch_valid  (slot 0 — active mon was alive and fainted)
      - opp KO: opp_no_switch_valid (slot 0 — active opp was alive and fainted)
    """
    own_tr  = [i for i in train_idx if no_switch_valid[i]]
    own_val = [i for i in val_idx   if no_switch_valid[i]]
    opp_tr  = [i for i in train_idx if opp_no_switch_valid[i]]
    opp_val = [i for i in val_idx   if opp_no_switch_valid[i]]

    # KO = was alive (>0.01) and now fainted (<0.01), slot 0 only
    ko_own = ((cur_hp_own[:, 0] > 0.01) & (next_hp_own[:, 0] < 0.01)).astype(np.int64)
    ko_opp = ((cur_hp_opp[:, 0] > 0.01) & (next_hp_opp[:, 0] < 0.01)).astype(np.int64)

    def _auc_own(X):
        if len(own_tr) < 20 or len(own_val) < 10 or ko_own[own_tr].sum() < 2:
            return float("nan")
        return round(_fit_logistic_auc(X[own_tr], ko_own[own_tr],
                                       X[own_val], ko_own[own_val]), 4)

    def _auc_opp(X):
        if len(opp_tr) < 20 or len(opp_val) < 10 or ko_opp[opp_tr].sum() < 2:
            return float("nan")
        return round(_fit_logistic_auc(X[opp_tr], ko_opp[opp_tr],
                                       X[opp_val], ko_opp[opp_val]), 4)

    per_query_opp_auc = []
    per_query_own_auc = []
    for i in range(13):
        X = detr_np[:, i, :]
        per_query_opp_auc.append(_auc_opp(X))
        per_query_own_auc.append(_auc_own(X))

    X_mean   = detr_np.mean(axis=1)
    X_chosen = detr_np[np.arange(len(actions)), actions, :]

    return {
        "per_query_opp_auc":  per_query_opp_auc,
        "per_query_own_auc":  per_query_own_auc,
        "mean_pool_opp_auc":  _auc_opp(X_mean),
        "mean_pool_own_auc":  _auc_own(X_mean),
        "chosen_opp_auc":     _auc_opp(X_chosen),
        "chosen_own_auc":     _auc_own(X_chosen),
        "n_ko_own":           int(ko_own[no_switch_valid].sum()),
        "n_ko_opp":           int(ko_opp[opp_no_switch_valid].sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

QUERY_LABELS = [f"M{i}" for i in range(4)] + [f"MM{i}" for i in range(4)] + [f"SW{i}" for i in range(5)]
QUERY_COLORS = ["#2196F3"] * 4 + ["#9C27B0"] * 4 + ["#4CAF50"] * 5


def save_figure(results: dict, out_path: Path, backbone_win_auc: float | None = None) -> None:
    """2×3 figure summarising all four probes."""
    act    = results["probe_action"]
    win    = results["probe_win"]
    dhp    = results["probe_delta_hp"]
    ko     = results["probe_ko"]
    n_q    = 13
    x      = np.arange(n_q)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("CynthAI_v2 — DETR Query Probes (backbone.act() cross-attention output)",
                 fontsize=11)

    # ── [0,0] Action AUC per query ───────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(x, act["per_query_auc"], color=QUERY_COLORS, edgecolor="none")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance=0.5")
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Probe 1 — Is query_i the chosen action?\n"
                 f"mean_pool 13-class acc={act['mean_pool_top1_acc']:.3f}", fontsize=9)
    ax.legend(fontsize=7)

    # ── [0,1] Win AUC per query ──────────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(x, win["per_query_auc"], color=QUERY_COLORS, edgecolor="none")
    ax.axhline(0.5, color="gray",   ls="--", lw=0.8, label="chance=0.5")
    ax.axhline(win["mean_pool_auc"],  color="#FF9800", ls="--", lw=1.5,
               label=f"mean_pool={win['mean_pool_auc']:.3f}")
    ax.axhline(win["chosen_query_auc"], color="#E53935", ls=":", lw=1.5,
               label=f"chosen_q={win['chosen_query_auc']:.3f}")
    if backbone_win_auc is not None:
        ax.axhline(backbone_win_auc, color="#00BCD4", ls="-.", lw=1.2,
                   label=f"backbone_mp={backbone_win_auc:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Probe 2 — Win probability", fontsize=9)
    ax.legend(fontsize=7)

    # ── [0,2] dHP R² per query (active slot 0, no-switch filtered) ──────────
    ax = axes[0, 2]
    w = 0.35
    opp_r2 = np.array(dhp["per_query_opp_r2"], dtype=float)
    own_r2 = np.array(dhp["per_query_own_r2"], dtype=float)
    ax.bar(x - w/2, opp_r2, width=w, color="#E53935", label="dHP_opp (active)", edgecolor="none")
    ax.bar(x + w/2, own_r2, width=w, color="#2196F3", alpha=0.6, label="dHP_own (active)", edgecolor="none")
    ax.axhline(dhp["mean_pool_opp_r2"], color="#E53935", ls="--", lw=1.0,
               label=f"mp_opp={dhp['mean_pool_opp_r2']:.3f}")
    ax.axhline(dhp["mean_pool_own_r2"], color="#2196F3", ls="--", lw=1.0,
               label=f"mp_own={dhp['mean_pool_own_r2']:.3f}")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²")
    n_own_v = dhp.get("n_own_valid", "?")
    n_opp_v = dhp.get("n_opp_valid", "?")
    own_mu  = dhp.get("label_own_mean", float("nan"))
    own_sig = dhp.get("label_own_std",  float("nan"))
    opp_mu  = dhp.get("label_opp_mean", float("nan"))
    opp_sig = dhp.get("label_opp_std",  float("nan"))
    ax.set_title(
        f"Probe 3 — dHP active slot R²  (no-switch filter)\n"
        f"own n={n_own_v}  mu={own_mu:+.3f} s={own_sig:.3f}   "
        f"opp n={n_opp_v}  mu={opp_mu:+.3f} s={opp_sig:.3f}",
        fontsize=8,
    )
    ax.legend(fontsize=7)

    # ── [1,0] dHP own R² per query (bar) ─────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x, own_r2, color=QUERY_COLORS, edgecolor="none")
    ax.axhline(dhp["mean_pool_own_r2"], color="#2196F3", ls="--", lw=1.2,
               label=f"mean_pool={dhp['mean_pool_own_r2']:.3f}")
    ax.axhline(dhp["chosen_own_r2"], color="#E53935", ls=":", lw=1.2,
               label=f"chosen_q={dhp['chosen_own_r2']:.3f}")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²"); ax.set_title("dHP_own R² per query (active, no-switch)", fontsize=9)
    ax.legend(fontsize=7)

    # ── [1,1] dHP opp R² per query (bar) ─────────────────────────────────────
    ax = axes[1, 1]
    ax.bar(x, opp_r2, color=QUERY_COLORS, edgecolor="none")
    ax.axhline(dhp["mean_pool_opp_r2"], color="#E53935", ls="--", lw=1.2,
               label=f"mean_pool={dhp['mean_pool_opp_r2']:.3f}")
    ax.axhline(dhp["chosen_opp_r2"], color="#9C27B0", ls=":", lw=1.2,
               label=f"chosen_q={dhp['chosen_opp_r2']:.3f}")
    ax.axhline(0, color="gray", ls=":", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("R²"); ax.set_title("dHP_opp R² per query (active, opp no-switch filter)", fontsize=9)
    ax.legend(fontsize=7)

    # ── [1,2] KO AUC grouped bars ────────────────────────────────────────────
    ax = axes[1, 2]
    ko_opp_arr = np.array(ko["per_query_opp_auc"], dtype=float)
    ko_own_arr = np.array(ko["per_query_own_auc"], dtype=float)
    w = 0.35
    ax.bar(x - w/2, ko_opp_arr, width=w, color="#E53935", label="KO_opp", edgecolor="none")
    ax.bar(x + w/2, ko_own_arr, width=w, color="#2196F3", alpha=0.7, label="KO_own", edgecolor="none")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(QUERY_LABELS, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.45, 1.05)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Probe 4 — KO next turn AUC", fontsize=9)
    ax.legend(fontsize=7)
    # Annotation
    note = (
        f"mean_pool  KO_opp={ko['mean_pool_opp_auc']:.3f}  KO_own={ko['mean_pool_own_auc']:.3f}\n"
        f"chosen_q   KO_opp={ko['chosen_opp_auc']:.3f}  KO_own={ko['chosen_own_auc']:.3f}"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="#BDBDBD", alpha=0.8))

    fig.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"PNG saved: {out_path.with_suffix('.png')}")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"PDF saved: {out_path.with_suffix('.pdf')}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Linear probes on DETR queries from backbone.act()."
    )
    parser.add_argument("--cache",  required=True,
                        help="Path to extended seq_all_cache.pt (from make_token_cache.py)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load cache ────────────────────────────────────────────────────────────
    cache_path = Path(args.cache)
    print(f"Loading cache: {cache_path}")
    data = torch.load(cache_path, map_location="cpu", weights_only=False)

    required = {"detr_queries", "actions", "next_hp_own", "next_hp_opp",
                "no_switch_valid", "opp_no_switch_valid"}
    missing  = required - set(data.keys())
    if missing:
        raise ValueError(
            f"Cache is missing keys: {missing}\n"
            "Re-run make_token_cache.py to generate an extended cache."
        )

    def _np(key, dtype=np.float32):
        v = data[key]
        return v.numpy().astype(dtype) if isinstance(v, torch.Tensor) else np.array(v, dtype=dtype)

    detr_np             = _np("detr_queries")
    actions             = _np("actions",             dtype=np.int64)
    next_hp_own         = _np("next_hp_own")
    next_hp_opp         = _np("next_hp_opp")
    no_switch_valid     = _np("no_switch_valid",     dtype=bool)
    opp_no_switch_valid = _np("opp_no_switch_valid", dtype=bool)

    N = len(actions)
    print(f"  N={N}  no_switch_valid={no_switch_valid.sum()}  opp_no_switch_valid={opp_no_switch_valid.sum()}")

    # ── Recover current-turn HP from seq_all cache (K_TURNS-1 turn, slots 36-47) ──
    # We need it to compute dHP and KO labels.
    # It lives in the buffer scalars, but cache may have been loaded without buffer.
    # Recompute from next_hp and delta: fallback — use 0.0 for cur HP when not available.
    # Better: store cur HP in cache. For now we load it from seq_all if it's there,
    # otherwise we use the data already stored as "next_hp" of the *previous* step, which
    # is not directly available. The simplest robust approach: assume cur HP ≈ 0.5 (flat prior)
    # and dHP labels are computed purely from (next_hp_own - cur_hp_own).
    # But since we have the cache, we can re-derive from the transition scalars stored in
    # next_hp_own at step n+1 vs step n. We need the raw scalars.
    #
    # Compromise: try to load y_win from a probing.json if available, and use
    # cur_hp = zeros (so delta = next_hp). This still makes dHP = HP_t+1 - 0 ≈ HP_t+1
    # which is informative but not the exact delta. Warn the user.
    #
    # If "cur_hp_own" is in the cache (future extension), use it directly.
    if "cur_hp_own" in data and "cur_hp_opp" in data:
        cur_hp_own = np.array(data["cur_hp_own"], dtype=np.float32)
        cur_hp_opp = np.array(data["cur_hp_opp"], dtype=np.float32)
    else:
        print("  [warn] cur_hp_own/opp not in cache — using next_hp as proxy for dHP probes.")
        print("         Re-run make_token_cache.py with updated code to get exact dHP.")
        cur_hp_own = np.zeros((N, 6), dtype=np.float32)
        cur_hp_opp = np.zeros((N, 6), dtype=np.float32)

    # ── Win labels ────────────────────────────────────────────────────────────
    if "y_win" in data:
        y_win = np.array(data["y_win"], dtype=np.float32)
        n_known = int((y_win >= 0).sum())
        print(f"  y_win: {n_known}/{N} known ({100*n_known/N:.1f}%)  "
              f"win_rate={float(y_win[y_win>=0].mean()):.3f}")
    else:
        y_win = np.full(N, -1.0, dtype=np.float32)
        print("  [warn] y_win not in cache — win probes will return AUC=0.5")

    # ── Backbone reference AUC from probing.json if available ─────────────────
    probing_json = cache_path.parent / "probing.json"
    backbone_win_auc = None
    if probing_json.exists():
        with open(probing_json) as f:
            pj = json.load(f)
        try:
            backbone_win_auc = pj["mean_pool_current"]["win_auc"]
            print(f"  backbone mean_pool_current win_auc = {backbone_win_auc:.3f} (from probing.json)")
        except KeyError:
            pass

    # ── Train / val split 80/20 ───────────────────────────────────────────────
    perm      = rng.permutation(N).tolist()
    n_train   = int(0.8 * N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    print(f"  train={len(train_idx)}  val={len(val_idx)}")

    # ── Run probes ────────────────────────────────────────────────────────────
    print("\nProbe 1 — action chosen ...")
    res_action = probe_action(detr_np, actions, train_idx, val_idx)
    print(f"  per_query_auc: {[f'{v:.3f}' for v in res_action['per_query_auc']]}")
    print(f"  mean_pool_top1_acc: {res_action['mean_pool_top1_acc']:.3f}")

    print("\nProbe 2 — win probability ...")
    res_win = probe_win(detr_np, actions, y_win, train_idx, val_idx)
    print(f"  per_query_auc: {[f'{v:.3f}' for v in res_win['per_query_auc']]}")
    print(f"  mean_pool_auc={res_win['mean_pool_auc']:.3f}  chosen_q={res_win['chosen_query_auc']:.3f}")

    print("\nProbe 3 — dHP next turn (active slot 0, no-switch filtered) ...")
    res_dhp = probe_delta_hp(
        detr_np, actions,
        cur_hp_own, cur_hp_opp,
        next_hp_own, next_hp_opp,
        no_switch_valid, opp_no_switch_valid,
        train_idx, val_idx,
    )
    print(f"  label own: mean={res_dhp['label_own_mean']:+.4f}  std={res_dhp['label_own_std']:.4f}  n={res_dhp['n_own_valid']}")
    print(f"  label opp: mean={res_dhp['label_opp_mean']:+.4f}  std={res_dhp['label_opp_std']:.4f}  n={res_dhp['n_opp_valid']}")
    print(f"  per_query_opp_r2 (mean): {np.nanmean(res_dhp['per_query_opp_r2']):.3f}")
    print(f"  per_query_own_r2 (mean): {np.nanmean(res_dhp['per_query_own_r2']):.3f}")
    print(f"  mean_pool_opp={res_dhp['mean_pool_opp_r2']:.3f}  mean_pool_own={res_dhp['mean_pool_own_r2']:.3f}")

    print("\nProbe 4 — KO next turn (active slot 0, no-switch filtered) ...")
    res_ko = probe_ko(
        detr_np, actions,
        cur_hp_own, cur_hp_opp,
        next_hp_own, next_hp_opp,
        no_switch_valid, opp_no_switch_valid,
        train_idx, val_idx,
    )
    print(f"  KO events: own={res_ko['n_ko_own']}  opp={res_ko['n_ko_opp']}")
    print(f"  per_query_opp_auc (mean): {np.nanmean(res_ko['per_query_opp_auc']):.3f}")
    print(f"  mean_pool_opp={res_ko['mean_pool_opp_auc']:.3f}  chosen_opp={res_ko['chosen_opp_auc']:.3f}")

    # ── Assemble per-query results dict ───────────────────────────────────────
    per_query = {}
    for i in range(13):
        per_query[str(i)] = {
            "label":          QUERY_LABELS[i],
            "action_auc":     res_action["per_query_auc"][i],
            "win_auc":        res_win["per_query_auc"][i],
            "delta_hp_own_r2": res_dhp["per_query_own_r2"][i],
            "delta_hp_opp_r2": res_dhp["per_query_opp_r2"][i],
            "ko_opp_auc":     res_ko["per_query_opp_auc"][i],
            "ko_own_auc":     res_ko["per_query_own_auc"][i],
        }

    output = {
        "n_transitions":      N,
        "n_no_switch_valid":  int(no_switch_valid.sum()),
        "n_opp_no_switch":    int(opp_no_switch_valid.sum()),
        "per_query":      per_query,
        "mean_pool": {
            "action_top1_acc": res_action["mean_pool_top1_acc"],
            "win_auc":         res_win["mean_pool_auc"],
            "delta_hp_own_r2": res_dhp["mean_pool_own_r2"],
            "delta_hp_opp_r2": res_dhp["mean_pool_opp_r2"],
            "ko_opp_auc":      res_ko["mean_pool_opp_auc"],
            "ko_own_auc":      res_ko["mean_pool_own_auc"],
        },
        "chosen_query": {
            "win_auc":         res_win["chosen_query_auc"],
            "delta_hp_own_r2": res_dhp["chosen_own_r2"],
            "delta_hp_opp_r2": res_dhp["chosen_opp_r2"],
            "ko_opp_auc":      res_ko["chosen_opp_auc"],
            "ko_own_auc":      res_ko["chosen_own_auc"],
        },
        "probe_action":   res_action,
        "probe_win":      res_win,
        "probe_delta_hp": res_dhp,
        "probe_ko":       res_ko,
    }

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir  = cache_path.parent
    json_out = out_dir / "probing_detr.json"
    with open(json_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON saved: {json_out}")

    # ── Save figure ───────────────────────────────────────────────────────────
    fig_out = out_dir / "probing_detr"
    save_figure(output, fig_out, backbone_win_auc=backbone_win_auc)


if __name__ == "__main__":
    main()
