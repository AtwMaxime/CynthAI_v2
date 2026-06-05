"""
probing.py — Linear probes on CynthAI_v2 Transformer token representations.

Probes all K×13=52 tokens in the full sequence after the Transformer, plus
mean-pool baselines (current turn and all 4 turns).

Backbone + poke_emb fully frozen. Cheater mode: full info, no POMDP masking.

Usage:
    cd /local_scratch/mattwood/projects/rl_agent/CynthAI_v2
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=1 python diag/probing.py \\
        --checkpoint checkpoints/cheater_v7/agent_001400.pt \\
        --n_envs 32 --min_steps 8192 \\
        --device cuda

Output: diag/probing.png, diag/probing.json
"""

from __future__ import annotations

import argparse
import json
import random
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

from model.agent import CynthAIAgent
from model.backbone import K_TURNS, N_SLOTS, D_MODEL, SEQ_LEN
from training.rollout import collect_rollout

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Labels for all 52 sequence tokens: T{turn}_{slot}
# Turn 0 = oldest, Turn K_TURNS-1 = current
SEQ_TOKEN_LABELS = []
for t in range(K_TURNS):
    for s in range(N_SLOTS):
        if s < 6:
            SEQ_TOKEN_LABELS.append(f"T{t}:O{s}")
        elif s < 12:
            SEQ_TOKEN_LABELS.append(f"T{t}:P{s-6}")
        else:
            SEQ_TOKEN_LABELS.append(f"T{t}:F")

assert len(SEQ_TOKEN_LABELS) == SEQ_LEN  # 52


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Token caching
# ─────────────────────────────────────────────────────────────────────────────

def cache_tokens(
    agent:      CynthAIAgent,
    buffer,
    device:     torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run poke_emb + backbone.encode(return_full_seq=True) over all transitions.

    Returns (CPU tensors):
        seq_all      : [N, 52, D_MODEL]  — all K*13 tokens after Transformer
        backbone_cls : [N, D_MODEL]      — CLS token output
        backbone_val : [N, 1]            — V(s) from backbone value head
    """
    agent.eval()
    n = len(buffer)
    all_seq = []
    all_cls = []
    all_val = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = buffer._gather(list(range(start, min(start + batch_size, n))), device)
            pt = agent.poke_emb(batch["poke_batch"])              # [B, K*12, TOKEN_DIM]
            ft = batch["field_tensor"]                             # [B, K, FIELD_DIM]
            _, _, value, _, seq, cls_out = agent.backbone.encode(pt, ft, return_full_seq=True)
            all_seq.append(seq.cpu())
            all_cls.append(cls_out.cpu())
            all_val.append(value.cpu())

    return (
        torch.cat(all_seq, dim=0),   # [N, 52, D_MODEL]
        torch.cat(all_cls, dim=0),   # [N, D_MODEL]
        torch.cat(all_val, dim=0),   # [N, 1]
    )


def cache_tokens_full(
    agent:      CynthAIAgent,
    buffer,
    device:     torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run poke_emb + backbone.encode + backbone.act(return_queries=True) over all transitions.

    Returns (all CPU tensors):
        seq_all      : [N, 52, D_MODEL]  — post-Transformer tokens (like cache_tokens)
        detr_queries : [N, 13, D_MODEL]  — attn_out from backbone.act()
        actions      : [N]               — chosen action int64
        backbone_cls : [N, D_MODEL]      — CLS token output
        backbone_val : [N, 1]            — V(s) from backbone value head
    """
    agent.eval()
    n = len(buffer)
    all_seq     = []
    all_queries = []
    all_cls     = []
    all_val     = []
    all_actions = torch.tensor(
        [buffer._transitions[i].action for i in range(n)], dtype=torch.long
    )

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = buffer._gather(list(range(start, end)), device)

            pt = agent.poke_emb(batch["poke_batch"])
            ft = batch["field_tensor"]
            pre_tokens, post_tokens, value, _, seq, cls_out = agent.backbone.encode(pt, ft, return_full_seq=True)

            action_embeds = agent.action_enc(
                active_token      = pre_tokens[:, 0, :],
                move_idx          = batch["move_idx"],
                pp_ratio          = batch["pp_ratio"],
                move_disabled     = batch["move_disabled"],
                bench_tokens      = pre_tokens[:, 1:6, :],
                mechanic_id       = batch["mechanic_id"],
                mechanic_type_idx = batch["mechanic_type_idx"],
            )

            _, _, _, attn_out = agent.backbone.act(
                action_embeds, post_tokens, batch["action_mask"], return_queries=True
            )

            all_seq.append(seq.cpu())
            all_queries.append(attn_out.cpu())
            all_cls.append(cls_out.cpu())
            all_val.append(value.cpu())

    return (
        torch.cat(all_seq,     dim=0),   # [N, 52, D_MODEL]
        torch.cat(all_queries, dim=0),   # [N, 13, D_MODEL]
        all_actions,                     # [N]
        torch.cat(all_cls,     dim=0),   # [N, D_MODEL]
        torch.cat(all_val,     dim=0),   # [N, 1]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Label extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_labels(buffer) -> dict[str, np.ndarray]:
    """
    Extract probe targets from rollout buffer transitions.

    Token tok_i in the sequence maps to:
      turn t = tok_i // N_SLOTS,  slot s = tok_i % N_SLOTS
      → Pokemon index in [K*12] arrays: t*12 + s   (only valid for s < 12)

    Returns:
        y_return   [N]          — Z-scored GAE returns
        y_win      [N]          — 0/1 win label; -1 for incomplete episodes
        y_type1    [N, K*12]    — type1 index for all 48 pokemon slots
        y_item     [N, K*12]    — item index
        y_ability  [N, K*12]    — ability index
        y_hp       [N, K*12]    — hp_ratio  (scalars[:, 1])
        y_stats    [N, K*12, 5] — 5 base stats (scalars[:, 3:8])
    """
    transitions = buffer._transitions
    N = len(transitions)

    # ── Returns ───────────────────────────────────────────────────────────────
    y_return = np.array(buffer._returns, dtype=np.float32)

    # ── Win labels (per env, episode-correct) ─────────────────────────────────
    y_win = np.full(N, -1.0, dtype=np.float32)
    ep_indices_per_env: dict[int, list[int]] = {}
    for n, tr in enumerate(transitions):
        env = tr.env_idx
        if env not in ep_indices_per_env:
            ep_indices_per_env[env] = []
        ep_indices_per_env[env].append(n)
        if tr.done:
            win = 1.0 if tr.reward > 0 else 0.0
            for k in ep_indices_per_env[env]:
                y_win[k] = win
            ep_indices_per_env[env] = []

    # ── Pokemon attributes — all K*12 slots ───────────────────────────────────
    all_type1   = torch.stack([tr.type1_idx   for tr in transitions])  # [N, K*12]
    all_item    = torch.stack([tr.item_idx    for tr in transitions])  # [N, K*12]
    all_ability = torch.stack([tr.ability_idx for tr in transitions])  # [N, K*12]
    all_scalars = torch.stack([tr.scalars     for tr in transitions])  # [N, K*12, N_SCALARS]

    return {
        "y_return":  y_return,
        "y_win":     y_win,
        "y_type1":   all_type1.numpy().astype(np.int64),           # [N, K*12]
        "y_item":    all_item.numpy().astype(np.int64),             # [N, K*12]
        "y_ability": all_ability.numpy().astype(np.int64),          # [N, K*12]
        "y_hp":      all_scalars[:, :, 1].numpy().astype(np.float32),    # [N, K*12]
        "y_stats":   all_scalars[:, :, 3:8].numpy().astype(np.float32),  # [N, K*12, 5]
    }


def extract_next_hp_labels(buffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute next-turn HP for each transition by grouping steps per env chronologically.

    For each non-terminal transition n in env e, the "next" HP is the HP observed
    in the following transition from the same env.

    HP layout in scalars [K*12, N_SCALARS]:
      Own current-turn : slots (K_TURNS-1)*12 + 0..5  → indices 36-41
      Opp current-turn : slots (K_TURNS-1)*12 + 6..11 → indices 42-47
    HP ratio is at scalar index 1.

    Returns (all float32, NaN where no next step):
        next_hp_own : [N, 6]
        next_hp_opp : [N, 6]
        next_valid  : [N]  bool  — True if a next step exists in the same episode
    """
    transitions = buffer._transitions
    N = len(transitions)

    OWN_BASE = (K_TURNS - 1) * 12       # 36
    OPP_BASE = (K_TURNS - 1) * 12 + 6  # 42

    next_hp_own = np.full((N, 6), np.nan, dtype=np.float32)
    next_hp_opp = np.full((N, 6), np.nan, dtype=np.float32)
    next_valid  = np.zeros(N, dtype=bool)

    # Group transition indices by env, in insertion order
    env_indices: dict[int, list[int]] = {}
    for n, tr in enumerate(transitions):
        env = tr.env_idx
        if env not in env_indices:
            env_indices[env] = []
        env_indices[env].append(n)

    for env_idx_list in env_indices.values():
        for k, n in enumerate(env_idx_list[:-1]):
            tr_cur  = transitions[n]
            tr_next = transitions[env_idx_list[k + 1]]
            if tr_cur.done:
                continue  # episode boundary — next step is a new episode
            sc_next = tr_next.scalars   # [K*12, N_SCALARS]
            for j in range(6):
                next_hp_own[n, j] = sc_next[OWN_BASE + j, 1].item()
                next_hp_opp[n, j] = sc_next[OPP_BASE + j, 1].item()
            next_valid[n] = True

    return next_hp_own, next_hp_opp, next_valid


# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_regression(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[float, float]:
    """Ridge regression. Returns (val_r2, val_pearson_r)."""
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_val_s)

    ss_res = float(np.sum((y_val - pred) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    if ss_tot < 1e-6:   # near-constant target → probe meaningless
        return float("nan"), 0.0
    r2 = max(-1.0, float(1.0 - ss_res / ss_tot))

    corr = (
        float(np.corrcoef(pred, y_val)[0, 1])
        if np.std(pred) > 1e-10 and np.std(y_val) > 1e-10
        else 0.0
    )
    return r2, corr


def _fit_classification(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    max_iter: int = 500,
    compute_auc: bool = False,
) -> tuple[float, float]:
    """
    Logistic regression. Returns (val_accuracy, val_auc).
    auc is 0.5 if compute_auc=False or computation fails.
    """
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    classes = np.unique(y_tr)
    if len(classes) < 2:
        pred = np.full(len(y_val), classes[0])
        return float(np.mean(pred == y_val)), 0.5

    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_val_s)
    acc  = float(np.mean(pred == y_val))

    auc = 0.5
    if compute_auc:
        try:
            all_classes = np.unique(np.concatenate([y_tr, y_val]))
            if len(all_classes) == 2:
                prob = model.predict_proba(X_val_s)[:, 1]
                auc  = float(roc_auc_score(y_val, prob))
            else:
                prob = model.predict_proba(X_val_s)
                auc  = float(roc_auc_score(
                    y_val, prob, multi_class="ovr", average="macro",
                    labels=model.classes_,
                ))
        except Exception:
            auc = 0.5

    return acc, auc


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Main probing loop
# ─────────────────────────────────────────────────────────────────────────────

def run_probes(
    seq_all:   torch.Tensor,   # [N, 52, D_MODEL]
    labels:    dict,
    train_idx: list[int],
    val_idx:   list[int],
) -> dict:
    """Run all linear probes over all 52 sequence tokens + mean pools."""
    seq_np = seq_all.numpy()   # [N, 52, 256]

    y_return  = labels["y_return"]
    y_win     = labels["y_win"]
    y_type1   = labels["y_type1"]    # [N, K*12]
    y_item    = labels["y_item"]
    y_ability = labels["y_ability"]
    y_hp      = labels["y_hp"]
    y_stats   = labels["y_stats"]    # [N, K*12, 5]

    win_mask = y_win >= 0
    win_tr   = [i for i in train_idx if win_mask[i]]
    win_val  = [i for i in val_idx   if win_mask[i]]

    results_per_token: dict[str, dict] = {}

    for tok_i in range(SEQ_LEN):    # 0..51
        t = tok_i // N_SLOTS        # turn (0=oldest, K-1=current)
        s = tok_i % N_SLOTS         # slot within turn (0-11=pokemon, 12=field)
        poke_idx = t * 12 + s       # index in [K*12] arrays (valid only for s<12)

        X = seq_np[:, tok_i, :]     # [N, 256]

        # Return
        ret_r2, ret_corr = _fit_regression(
            X[train_idx], y_return[train_idx],
            X[val_idx],   y_return[val_idx],
        )

        # Win
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = _fit_classification(
                X[win_tr], y_win[win_tr].astype(np.int64),
                X[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True,
            )
        else:
            win_acc, win_auc = 0.5, 0.5

        tok_res: dict = {
            "return_r2":   round(ret_r2,   4),
            "return_corr": round(ret_corr, 4),
            "win_acc":     round(win_acc,  4),
            "win_auc":     round(win_auc,  4),
        }

        if s < 12:  # Pokemon token
            type1_acc, _ = _fit_classification(
                X[train_idx], y_type1[train_idx, poke_idx],
                X[val_idx],   y_type1[val_idx,   poke_idx],
                max_iter=500,
            )
            item_acc, _ = _fit_classification(
                X[train_idx], y_item[train_idx, poke_idx],
                X[val_idx],   y_item[val_idx,   poke_idx],
                max_iter=1000,
            )
            ability_acc, _ = _fit_classification(
                X[train_idx], y_ability[train_idx, poke_idx],
                X[val_idx],   y_ability[val_idx,   poke_idx],
                max_iter=1000,
            )
            hp_r2, _ = _fit_regression(
                X[train_idx], y_hp[train_idx, poke_idx],
                X[val_idx],   y_hp[val_idx,   poke_idx],
            )
            stat_r2s = []
            for stat in range(5):
                r2, _ = _fit_regression(
                    X[train_idx], y_stats[train_idx, poke_idx, stat],
                    X[val_idx],   y_stats[val_idx,   poke_idx, stat],
                )
                stat_r2s.append(r2)
            basestats_r2 = float(np.mean(stat_r2s))

            tok_res.update({
                "type1_acc":    round(type1_acc,    4),
                "item_acc":     round(item_acc,     4),
                "ability_acc":  round(ability_acc,  4),
                "hp_r2":        round(hp_r2,        4),
                "basestats_r2": round(basestats_r2, 4),
            })
            print(f"  {SEQ_TOKEN_LABELS[tok_i]:8s} | "
                  f"ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}  "
                  f"type1={type1_acc:.3f}  item={item_acc:.3f}  "
                  f"ability={ability_acc:.3f}  hp={hp_r2:.3f}  bstats={basestats_r2:.3f}")
        else:  # Field token
            tok_res.update({
                "type1_acc": None, "item_acc": None, "ability_acc": None,
                "hp_r2": None, "basestats_r2": None,
            })
            print(f"  {SEQ_TOKEN_LABELS[tok_i]:8s} | "
                  f"ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")

        results_per_token[str(tok_i)] = tok_res

    # ── Mean pool probes ───────────────────────────────────────────────────────
    def _probe_pool(X_pool: np.ndarray, label: str) -> dict:
        ret_r2, ret_corr = _fit_regression(
            X_pool[train_idx], y_return[train_idx],
            X_pool[val_idx],   y_return[val_idx],
        )
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc, win_auc = _fit_classification(
                X_pool[win_tr], y_win[win_tr].astype(np.int64),
                X_pool[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True,
            )
        else:
            win_acc, win_auc = 0.5, 0.5
        print(f"  {label:20s} | ret_r2={ret_r2:+.3f}  win_auc={win_auc:.3f}")
        return {
            "return_r2":   round(ret_r2,   4),
            "return_corr": round(ret_corr, 4),
            "win_acc":     round(win_acc,  4),
            "win_auc":     round(win_auc,  4),
        }

    # Current-turn tokens = last 13 in seq_all
    X_mean_cur = seq_np[:, -N_SLOTS:, :].mean(axis=1)   # [N, 256]
    X_mean_all = seq_np.mean(axis=1)                     # [N, 256]

    print("\nMean pool probes:")
    pool_cur = _probe_pool(X_mean_cur, "mean_pool_current")
    pool_all = _probe_pool(X_mean_all, "mean_pool_all")

    # ── Per-slot cross-turn pool probes ────────────────────────────────────────
    # For each Pokemon slot s (0..11): average that slot's token across all K turns.
    # This tests whether the historical aggregate carries more attribute information
    # than any single-turn token for the same slot.
    # Label source: current turn (t=K_TURNS-1), poke_idx = (K-1)*12 + s
    SLOT_NAMES = [f"OWN{s}" if s < 6 else f"OPP{s-6}" for s in range(12)]
    per_slot_pool: dict[str, dict] = {}

    print("\nPer-slot cross-turn pool probes:")
    for s in range(12):
        indices  = [t * N_SLOTS + s for t in range(K_TURNS)]   # 4 token positions
        X_slot   = seq_np[:, indices, :].mean(axis=1)            # [N, 256]
        poke_idx = (K_TURNS - 1) * 12 + s                        # current turn slot

        type1_acc, _ = _fit_classification(
            X_slot[train_idx], y_type1[train_idx, poke_idx],
            X_slot[val_idx],   y_type1[val_idx,   poke_idx],
            max_iter=500,
        )
        item_acc, _ = _fit_classification(
            X_slot[train_idx], y_item[train_idx, poke_idx],
            X_slot[val_idx],   y_item[val_idx,   poke_idx],
            max_iter=1000,
        )
        ability_acc, _ = _fit_classification(
            X_slot[train_idx], y_ability[train_idx, poke_idx],
            X_slot[val_idx],   y_ability[val_idx,   poke_idx],
            max_iter=1000,
        )
        hp_r2, _ = _fit_regression(
            X_slot[train_idx], y_hp[train_idx, poke_idx],
            X_slot[val_idx],   y_hp[val_idx,   poke_idx],
        )
        stat_r2s = []
        for stat in range(5):
            r2, _ = _fit_regression(
                X_slot[train_idx], y_stats[train_idx, poke_idx, stat],
                X_slot[val_idx],   y_stats[val_idx,   poke_idx, stat],
            )
            stat_r2s.append(r2)
        bstats_r2 = float(np.mean(stat_r2s))

        win_acc_s, win_auc_s = 0.5, 0.5
        if len(win_tr) > 20 and len(win_val) > 10:
            win_acc_s, win_auc_s = _fit_classification(
                X_slot[win_tr], y_win[win_tr].astype(np.int64),
                X_slot[win_val], y_win[win_val].astype(np.int64),
                max_iter=500, compute_auc=True,
            )

        per_slot_pool[SLOT_NAMES[s]] = {
            "type1_acc":    round(type1_acc,   4),
            "item_acc":     round(item_acc,    4),
            "ability_acc":  round(ability_acc, 4),
            "hp_r2":        round(hp_r2,       4),
            "basestats_r2": round(bstats_r2,   4),
            "win_acc":      round(win_acc_s,   4),
            "win_auc":      round(win_auc_s,   4),
        }
        print(f"  {SLOT_NAMES[s]:6s} | "
              f"type1={type1_acc:.3f}  item={item_acc:.3f}  "
              f"ability={ability_acc:.3f}  hp={hp_r2:.3f}  "
              f"bstats={bstats_r2:.3f}  win_auc={win_auc_s:.3f}")

    return {
        "per_token":         results_per_token,
        "mean_pool_current": pool_cur,
        "mean_pool_all":     pool_all,
        "per_slot_pool":     per_slot_pool,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-token matrix
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_token_matrix(
    seq_all:   torch.Tensor,   # [N, 52, D_MODEL]
    labels:    dict,
    train_idx: list[int],
    val_idx:   list[int],
) -> dict:
    """
    For each pair (i, j) of current-turn Pokemon slots (0..11):
      probe whether token_i encodes the attribute of slot_j.

    This reveals how attention routes information between tokens:
    - Diagonal (i=j): trivially high — info is directly in the input
    - Off-diagonal: measures cross-token information sharing via attention

    Returns dict with [12, 12] matrices (as nested lists for JSON serialisation):
        hp_r2_mat, type1_acc_mat, item_acc_mat, ability_acc_mat, bstats_r2_mat
    """
    seq_np = seq_all.numpy()   # [N, 52, D_MODEL]

    cur_turn_start = (K_TURNS - 1) * N_SLOTS   # first token index of current turn
    poke_start     = (K_TURNS - 1) * 12         # first poke_idx of current turn

    y_type1   = labels["y_type1"]
    y_item    = labels["y_item"]
    y_ability = labels["y_ability"]
    y_hp      = labels["y_hp"]
    y_stats   = labels["y_stats"]

    n_slots = 12
    hp_r2_mat       = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    type1_acc_mat   = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    item_acc_mat    = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    ability_acc_mat = np.full((n_slots, n_slots), np.nan, dtype=np.float32)
    bstats_r2_mat   = np.full((n_slots, n_slots), np.nan, dtype=np.float32)

    total = n_slots * n_slots
    done  = 0
    for i in range(n_slots):
        X     = seq_np[:, cur_turn_start + i, :]   # [N, 256]
        X_tr  = X[train_idx]
        X_val = X[val_idx]
        for j in range(n_slots):
            poke_idx = poke_start + j

            hp_r2, _ = _fit_regression(
                X_tr, y_hp[train_idx, poke_idx],
                X_val, y_hp[val_idx, poke_idx],
            )
            hp_r2_mat[i, j] = hp_r2

            type1_acc, _ = _fit_classification(
                X_tr, y_type1[train_idx, poke_idx],
                X_val, y_type1[val_idx, poke_idx],
                max_iter=500,
            )
            type1_acc_mat[i, j] = type1_acc

            item_acc, _ = _fit_classification(
                X_tr, y_item[train_idx, poke_idx],
                X_val, y_item[val_idx, poke_idx],
                max_iter=1000,
            )
            item_acc_mat[i, j] = item_acc

            ability_acc, _ = _fit_classification(
                X_tr, y_ability[train_idx, poke_idx],
                X_val, y_ability[val_idx, poke_idx],
                max_iter=1000,
            )
            ability_acc_mat[i, j] = ability_acc

            stat_r2s = []
            for stat in range(5):
                r2, _ = _fit_regression(
                    X_tr, y_stats[train_idx, poke_idx, stat],
                    X_val, y_stats[val_idx, poke_idx, stat],
                )
                stat_r2s.append(r2)
            bstats_r2_mat[i, j] = float(np.nanmean(stat_r2s))

            done += 1
        print(f"  cross-token matrix: {done}/{total}  (source token OWN{i} done)"
              if i < 6 else
              f"  cross-token matrix: {done}/{total}  (source token OPP{i-6} done)")

    return {
        "hp_r2_mat":       hp_r2_mat.tolist(),
        "type1_acc_mat":   type1_acc_mat.tolist(),
        "item_acc_mat":    item_acc_mat.tolist(),
        "ability_acc_mat": ability_acc_mat.tolist(),
        "bstats_r2_mat":   bstats_r2_mat.tolist(),
    }


def save_cross_matrix_figure(cross: dict, out_path: Path) -> None:
    """
    Save a 2×3 figure with 12×12 cross-token probe matrices:
      Row 0: hp R²  |  type1 accuracy  |  item accuracy
      Row 1: ability accuracy  |  bstats R²  |  (colourbar legend)

    x-axis = target slot (OWN0-5, OPP0-5)
    y-axis = source token (OWN0-5, OPP0-5)
    Diagonal = trivially high (self-encoding).
    Off-diagonal = cross-token information routing.
    """
    SLOT_LABELS = [f"OWN{s}" if s < 6 else f"OPP{s-6}" for s in range(12)]

    def _mat(key):
        return np.array(cross[key], dtype=np.float32)   # [12, 12]

    hp_mat      = _mat("hp_r2_mat")
    type1_mat   = _mat("type1_acc_mat")
    item_mat    = _mat("item_acc_mat")
    ability_mat = _mat("ability_acc_mat")
    bstats_mat  = _mat("bstats_r2_mat")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "CynthAI_v2 — Cross-token probe matrix  (current turn, 12×12)\n"
        "Row = source token, Col = target slot attribute",
        fontsize=11,
    )

    sep = 5.5   # OWN|OPP separator

    def _draw(ax, data, title, vmin, vmax, cmap="plasma"):
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       origin="upper")
        ax.set_xticks(range(12)); ax.set_xticklabels(SLOT_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(12)); ax.set_yticklabels(SLOT_LABELS, fontsize=7)
        ax.set_xlabel("target slot attribute", fontsize=7)
        ax.set_ylabel("source token", fontsize=7)
        ax.axhline(sep, color="white", lw=1.5, alpha=0.8)
        ax.axvline(sep, color="white", lw=1.5, alpha=0.8)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw(axes[0, 0], hp_mat,      "HP ratio R²",           vmin=0.0, vmax=1.0)
    _draw(axes[0, 1], type1_mat,   "Type1 accuracy",         vmin=0.0, vmax=1.0)
    _draw(axes[0, 2], item_mat,    "Item accuracy",          vmin=0.0, vmax=1.0)
    _draw(axes[1, 0], ability_mat, "Ability accuracy",       vmin=0.0, vmax=1.0)
    _draw(axes[1, 1], bstats_mat,  "Base stats R² (mean×5)", vmin=0.0, vmax=1.0)

    # [1,2]: text annotation explaining the layout
    axes[1, 2].axis("off")
    note = (
        "Diagonal (i=j):\n"
        "  trivially high — attribute info\n"
        "  is directly in token_i's input.\n\n"
        "Off-diagonal (i≠j):\n"
        "  measures whether token_i encodes\n"
        "  slot_j's attribute via attention.\n\n"
        "OWN|OPP boundary: white lines.\n\n"
        "Cheater mode: full info, no POMDP.\n"
        "Source/target: current turn only."
    )
    axes[1, 2].text(0.05, 0.95, note, transform=axes[1, 2].transAxes,
                    fontsize=8, va="top", family="monospace",
                    bbox=dict(boxstyle="round", fc="#F5F5F5", ec="#BDBDBD"))

    fig.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"Cross-token matrix PNG: {out_path.with_suffix('.png')}")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Cross-token matrix PDF: {out_path.with_suffix('.pdf')}")


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict, N: int) -> None:
    per      = results["per_token"]
    pool_cur = results["mean_pool_current"]
    pool_all = results["mean_pool_all"]

    header = (f"{'token':9s} {'ret_r2':>7s} {'ret_r':>6s} {'win_acc':>7s} "
              f"{'win_auc':>7s} {'type1':>6s} {'item':>6s} {'ability':>7s} "
              f"{'hp_r2':>6s} {'bstats':>7s}")
    print(f"\n{'='*len(header)}")
    print(f"PROBING RESULTS — val set   N={N}")
    print(f"{'='*len(header)}")
    print(header)

    prev_turn = -1
    for i in range(SEQ_LEN):
        t = i // N_SLOTS
        if t != prev_turn:
            print("-" * len(header))
            prev_turn = t
        r = per[str(i)]
        def _f(v): return f"{v:7.3f}" if v is not None else "      — "
        print(f"{SEQ_TOKEN_LABELS[i]:9s} "
              f"{_f(r['return_r2'])} {_f(r['return_corr'])} {_f(r['win_acc'])} "
              f"{_f(r['win_auc'])} {_f(r['type1_acc'])} {_f(r['item_acc'])} "
              f"{_f(r['ability_acc'])} {_f(r['hp_r2'])} {_f(r['basestats_r2'])}")

    print("-" * len(header))
    def _fp(pool, name):
        r = pool
        print(f"{name:9s} {r['return_r2']:7.3f} {r['return_corr']:6.3f} "
              f"{r['win_acc']:7.3f} {r['win_auc']:7.3f}")
    _fp(pool_cur, "mp_cur")
    _fp(pool_all, "mp_all")
    print(f"{'='*len(header)}")

    # Per-slot cross-turn pool summary
    per_slot_pool = results.get("per_slot_pool", {})
    if per_slot_pool:
        hdr2 = (f"{'slot':7s} {'type1':>6s} {'item':>6s} {'ability':>7s} "
                f"{'hp_r2':>6s} {'bstats':>7s} {'win_auc':>7s}")
        print(f"\nCross-turn slot pool (mean T0..T{K_TURNS-1} per slot):")
        print(f"{'='*len(hdr2)}")
        print(hdr2)
        SLOT_NAMES = [f"OWN{s}" if s < 6 else f"OPP{s-6}" for s in range(12)]
        for i, name in enumerate(SLOT_NAMES):
            if i == 6:
                print("-" * len(hdr2))
            r = per_slot_pool[name]
            print(f"{name:7s} "
                  f"{r['type1_acc']:6.3f} {r['item_acc']:6.3f} "
                  f"{r['ability_acc']:7.3f} {r['hp_r2']:6.3f} "
                  f"{r['basestats_r2']:7.3f} {r['win_auc']:7.3f}")
        print(f"{'='*len(hdr2)}")


def save_json(results: dict, N: int, out_path: Path) -> None:
    data = {"n_transitions": N}
    data.update(results)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved: {out_path}")


def save_figure(results: dict, out_path: Path) -> None:
    """Save figure as both PNG and PDF (same path with different extension)."""
    per           = results["per_token"]
    pool_cur      = results["mean_pool_current"]
    pool_all      = results["mean_pool_all"]
    per_slot_pool = results["per_slot_pool"]

    x = np.arange(SEQ_LEN)   # 0..51

    def _arr(key):
        return np.array([per[str(i)][key] if per[str(i)][key] is not None else np.nan
                         for i in range(SEQ_LEN)])

    ret_r2_arr  = _arr("return_r2")
    win_auc_arr = _arr("win_auc")
    type1_arr   = _arr("type1_acc")
    item_arr    = _arr("item_acc")
    ability_arr = _arr("ability_acc")
    hp_arr      = _arr("hp_r2")
    bstats_arr  = _arr("basestats_r2")

    # Colors: OWN=blue, OPP=red, FIELD=gray, repeated per turn
    _slot_colors = ["#2196F3"] * 6 + ["#E53935"] * 6 + ["#9E9E9E"]
    bar_colors   = _slot_colors * K_TURNS

    # x-axis separators between turns (columns 12.5, 25.5, 38.5)
    turn_seps = [N_SLOTS * t - 0.5 for t in range(1, K_TURNS)]

    def _turn_vlines(ax, color="gray", lw=0.8, ls="--", alpha=0.4):
        """Draw vertical separators between turns on a bar/line plot."""
        for xs in turn_seps:
            ax.axvline(xs, color=color, lw=lw, ls=ls, alpha=alpha)

    def _heatmap(ax, data, cmap, vmin, vmax, ytick_labels, title):
        """
        Draw a heatmap (data: [n_rows, 52]) with:
        - correct aspect="auto" fill
        - vertical turn separators
        - x-axis token labels
        - colorbar
        """
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        n_rows = data.shape[0]
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(SEQ_TOKEN_LABELS, rotation=90, ha="center", fontsize=5.5)
        ax.set_title(title, fontsize=9)
        # Vertical separators between turns (white lines in the heatmap)
        for xs in turn_seps:
            ax.axvline(xs, color="white", lw=1.8, alpha=0.8)
        return im

    # Per-slot cross-turn pool arrays (12 slots)
    SLOT_NAMES = [f"OWN{s}" if s < 6 else f"OPP{s-6}" for s in range(12)]
    xs12 = np.arange(12)
    slot_colors12 = ["#2196F3"] * 6 + ["#E53935"] * 6

    def _slot_arr(key):
        return np.array([per_slot_pool[n][key] for n in SLOT_NAMES], dtype=np.float32)

    sp_type1   = _slot_arr("type1_acc")
    sp_item    = _slot_arr("item_acc")
    sp_ability = _slot_arr("ability_acc")
    sp_hp      = _slot_arr("hp_r2")
    sp_bstats  = _slot_arr("basestats_r2")
    sp_winauc  = _slot_arr("win_auc")

    # ── Figure layout: 4 rows × 2 cols ────────────────────────────────────────
    # Row 0: Win AUC heatmap        |  Return R² bar
    # Row 1: Win AUC bar            |  Type1 / Item / Ability heatmap (52 tokens)
    # Row 2: HP ratio heatmap       |  Base stats heatmap
    # Row 3: Cross-turn slot pool:  |  Cross-turn slot pool:
    #         ID accuracy (3 rows)  |   HP + bstats R²  +  win AUC
    fig = plt.figure(figsize=(22, 15))
    fig.suptitle("CynthAI_v2 — Linear Probing (cheater, full info, 52 tokens × 4 turns)",
                 fontsize=12)

    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.25)

    tick_kw = dict(rotation=90, ha="center", fontsize=5.5)

    # ── [0,0] Heatmap: win AUC (1 row × 52) ──────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    im00 = _heatmap(ax00,
                    win_auc_arr.reshape(1, -1),
                    cmap="viridis", vmin=0.5, vmax=1.0,
                    ytick_labels=["win AUC"],
                    title="Win AUC — heatmap (vmin=0.5)")
    fig.colorbar(im00, ax=ax00, fraction=0.03, pad=0.04)

    # ── [0,1] Bar: return R² (zoomed, outliers annotated) ────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    # Clip bars that fall outside the zoom window
    ylim_lo, ylim_hi = -0.12, 0.05
    clipped = ret_r2_arr < ylim_lo
    bar_vals = np.where(clipped, ylim_lo, ret_r2_arr)
    ax01.bar(x, bar_vals, color=bar_colors, edgecolor="none", width=0.8)
    # Mark clipped bars with an asterisk
    for xi, clip in zip(x, clipped):
        if clip:
            ax01.text(xi, ylim_lo + 0.005, "*", ha="center", va="bottom",
                      fontsize=5, color="black")
    ax01.axhline(0, color="black", lw=0.6, ls="-", alpha=0.4)
    ax01.axhline(pool_cur["return_r2"], color="#FF9800", ls="--", lw=1.5,
                 label=f"mp_cur={pool_cur['return_r2']:.3f}")
    ax01.axhline(pool_all["return_r2"], color="#9C27B0", ls=":", lw=1.5,
                 label=f"mp_all={pool_all['return_r2']:.3f}")
    _turn_vlines(ax01)
    ax01.set_xticks(x); ax01.set_xticklabels(SEQ_TOKEN_LABELS, **tick_kw)
    ax01.set_ylabel("R²"); ax01.set_title("Return R² per token  (* = clipped < -0.12)", fontsize=9)
    ax01.set_ylim(ylim_lo, ylim_hi)
    ax01.legend(fontsize=7)

    # ── [1,0] Bar: win AUC ────────────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.bar(x, win_auc_arr, color=bar_colors, edgecolor="none", width=0.8)
    ax10.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance=0.5")
    ax10.axhline(pool_cur["win_auc"], color="#FF9800", ls="--", lw=1.5,
                 label=f"mp_cur={pool_cur['win_auc']:.3f}")
    ax10.axhline(pool_all["win_auc"], color="#9C27B0", ls=":", lw=1.5,
                 label=f"mp_all={pool_all['win_auc']:.3f}")
    _turn_vlines(ax10)
    ax10.set_xticks(x); ax10.set_xticklabels(SEQ_TOKEN_LABELS, **tick_kw)
    ax10.set_ylabel("AUC-ROC"); ax10.set_title("Win AUC per token", fontsize=9)
    ax10.set_ylim(0.5, 1.0)
    ax10.legend(fontsize=7)

    # ── [1,1] Heatmap: type1 / item / ability (3 rows) ───────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    hmap_id = np.array([type1_arr, item_arr, ability_arr])   # [3, 52]
    # vmin=0.9 to show variation among near-perfect accuracies
    im11 = _heatmap(ax11, hmap_id,
                    cmap="plasma", vmin=0.9, vmax=1.0,
                    ytick_labels=["type1", "item", "ability"],
                    title="Type1 / Item / Ability accuracy  (vmin=0.9)")
    fig.colorbar(im11, ax=ax11, fraction=0.03, pad=0.04)

    # ── [2,0] Heatmap: hp_ratio R² ───────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    im20 = _heatmap(ax20, hp_arr.reshape(1, -1),
                    cmap="plasma", vmin=0.0, vmax=1.0,
                    ytick_labels=["hp_ratio R²"],
                    title="HP ratio R² per token")
    fig.colorbar(im20, ax=ax20, fraction=0.03, pad=0.04)

    # ── [2,1] Heatmap: basestats R² ──────────────────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    im21 = _heatmap(ax21, bstats_arr.reshape(1, -1),
                    cmap="plasma", vmin=0.0, vmax=1.0,
                    ytick_labels=["bstats R²"],
                    title="Base stats R² per token (mean over 5 stats)")
    fig.colorbar(im21, ax=ax21, fraction=0.03, pad=0.04)

    # ── [3,0] Heatmap: cross-turn slot pool — ID (type1/item/ability) ─────────
    ax30 = fig.add_subplot(gs[3, 0])
    hmap_slot_id = np.array([sp_type1, sp_item, sp_ability])  # [3, 12]
    im30 = ax30.imshow(hmap_slot_id, aspect="auto", cmap="plasma",
                       vmin=0.9, vmax=1.0, interpolation="nearest")
    ax30.set_yticks([0, 1, 2])
    ax30.set_yticklabels(["type1", "item", "ability"], fontsize=8)
    ax30.set_xticks(xs12)
    ax30.set_xticklabels(SLOT_NAMES, rotation=45, ha="right", fontsize=7)
    ax30.axvline(5.5, color="white", lw=1.8, alpha=0.8)   # OWN | OPP separator
    ax30.set_title("Cross-turn slot pool — ID accuracy (vmin=0.9)", fontsize=9)
    fig.colorbar(im30, ax=ax30, fraction=0.03, pad=0.04)

    # ── [3,1] Bar: cross-turn slot pool — hp / bstats / win AUC ─────────────
    ax31 = fig.add_subplot(gs[3, 1])
    w = 0.25
    ax31.bar(xs12 - w, sp_hp,     width=w, color="#00BCD4", label="hp R²")
    ax31.bar(xs12,     sp_bstats, width=w, color="#8BC34A", label="bstats R²")
    ax31.bar(xs12 + w, sp_winauc, width=w, color="#FF5722", label="win AUC")
    ax31.axhline(0.5, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax31.axvline(5.5, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax31.set_xticks(xs12)
    ax31.set_xticklabels(SLOT_NAMES, rotation=45, ha="right", fontsize=7)
    ax31.set_ylim(0.0, 1.05)
    ax31.set_title("Cross-turn slot pool — HP R², bstats R², win AUC", fontsize=9)
    ax31.legend(fontsize=7)

    plt.savefig(out_path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    print(f"PNG saved: {out_path.with_suffix('.png')}")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"PDF saved: {out_path.with_suffix('.pdf')}")


# ─────────────────────────────────────────────────────────────────────────────
# CLS probing — backbone CLS token analysis
# ─────────────────────────────────────────────────────────────────────────────

def effective_rank(X: np.ndarray) -> float:
    """
    Effective rank = exp(H(sigma_norm)) where sigma are singular values of
    centered X and H is Shannon entropy of the normalised spectrum.
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X_c, compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def per_dim_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between each column of X and y. Returns [D]."""
    D = X.shape[1]
    corrs = np.zeros(D, dtype=np.float64)
    if np.std(y) < 1e-10:
        return corrs
    for d in range(D):
        if np.std(X[:, d]) > 1e-10:
            corrs[d] = np.corrcoef(X[:, d], y)[0, 1]
    return corrs


def extract_aggregate_labels(labels: dict) -> dict:
    """
    Derive aggregate labels from per-pokemon labels.

    Returns:
        mean_hp_own  : [N]  mean HP ratio of own team (current turn, 6 slots)
        mean_hp_opp  : [N]  mean HP ratio of opp team
        alive_own    : [N]  count of own pokemon with HP > 0
        alive_opp    : [N]  count of opp pokemon with HP > 0
    """
    y_hp = labels["y_hp"]   # [N, K*12]
    cur_own = y_hp[:, (K_TURNS - 1) * 12:(K_TURNS - 1) * 12 + 6]   # [N, 6]
    cur_opp = y_hp[:, (K_TURNS - 1) * 12 + 6:(K_TURNS - 1) * 12 + 12]  # [N, 6]
    return {
        "mean_hp_own": cur_own.mean(axis=1).astype(np.float32),
        "mean_hp_opp": cur_opp.mean(axis=1).astype(np.float32),
        "alive_own":   (cur_own > 0).sum(axis=1).astype(np.float32),
        "alive_opp":   (cur_opp > 0).sum(axis=1).astype(np.float32),
    }


def run_cls_probes(
    backbone_cls:  np.ndarray,   # [N, D_MODEL]
    backbone_val:  np.ndarray,   # [N]
    labels:        dict,
    agg_labels:    dict,
    train_idx:     list[int],
    val_idx:       list[int],
) -> dict:
    """
    Linear probes on the backbone CLS token.

    Probes: return R2, win AUC, mean HP own/opp R2, alive own/opp R2,
    effective rank, per-dim correlation with value and win.
    """
    y_return = labels["y_return"]
    y_win    = labels["y_win"]

    win_mask = y_win >= 0
    win_tr   = [i for i in train_idx if win_mask[i]]
    win_val  = [i for i in val_idx   if win_mask[i]]

    X = backbone_cls

    print("\nBackbone CLS probes:")

    # ── Return R2 ───────────────────────────────────────────────────────────
    ret_r2, ret_corr = _fit_regression(
        X[train_idx], y_return[train_idx],
        X[val_idx],   y_return[val_idx],
    )
    print(f"  return R2:   {ret_r2:+.4f}  corr: {ret_corr:.4f}")

    # ── Win AUC ─────────────────────────────────────────────────────────────
    win_acc, win_auc = 0.5, 0.5
    if len(win_tr) > 20 and len(win_val) > 10:
        win_acc, win_auc = _fit_classification(
            X[win_tr], y_win[win_tr].astype(np.int64),
            X[win_val], y_win[win_val].astype(np.int64),
            max_iter=500, compute_auc=True,
        )
    print(f"  win AUC:     {win_auc:.4f}  acc: {win_acc:.4f}")

    # ── Aggregate HP and alive probes ───────────────────────────────────────
    agg_results = {}
    for key in ("mean_hp_own", "mean_hp_opp", "alive_own", "alive_opp"):
        y = agg_labels[key]
        r2, corr = _fit_regression(
            X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
        )
        agg_results[key + "_r2"] = round(r2, 4)
        print(f"  {key:15s} R2: {r2:+.4f}")

    # ── Effective rank ──────────────────────────────────────────────────────
    erank = effective_rank(X)
    print(f"  effective rank: {erank:.2f} / {D_MODEL}")

    # ── Per-dim correlation with value and win ──────────────────────────────
    corr_value = per_dim_correlation(X, backbone_val)
    corr_win   = np.zeros(D_MODEL)
    if win_mask.sum() > 50:
        corr_win = per_dim_correlation(X[win_mask], y_win[win_mask])

    profile_corr = 0.0
    if np.std(corr_value) > 1e-10 and np.std(corr_win) > 1e-10:
        profile_corr = float(np.corrcoef(corr_value, corr_win)[0, 1])

    print(f"  max |corr(dim, V)|:   {np.abs(corr_value).max():.4f}")
    print(f"  max |corr(dim, win)|: {np.abs(corr_win).max():.4f}")
    print(f"  profile corr(V, win): {profile_corr:.4f}")

    return {
        "return_r2":   round(ret_r2, 4),
        "return_corr": round(ret_corr, 4),
        "win_acc":     round(win_acc, 4),
        "win_auc":     round(win_auc, 4),
        **agg_results,
        "effective_rank": round(erank, 2),
        "max_abs_corr_value":  round(float(np.abs(corr_value).max()), 4),
        "max_abs_corr_win":    round(float(np.abs(corr_win).max()), 4),
        "mean_abs_corr_value": round(float(np.abs(corr_value).mean()), 4),
        "mean_abs_corr_win":   round(float(np.abs(corr_win).mean()), 4),
        "profile_corr":        round(profile_corr, 4),
        "corr_value":          corr_value.tolist(),
        "corr_win":            corr_win.tolist(),
    }


def save_cls_figure(
    backbone_cls: np.ndarray,
    backbone_val: np.ndarray,
    y_win:        np.ndarray,
    cls_results:  dict,
    out_path:     Path,
) -> dict:
    """
    Backbone CLS analysis figure (2x3):
      [0,0] PCA colored by V(s)
      [0,1] PCA colored by win/loss
      [0,2] Explained variance (top 30 PCs)
      [1,0] Per-dim corr with V(s) (sorted)
      [1,1] Per-dim corr with win (sorted)
      [1,2] Scatter corr_value vs corr_win
    """
    # PCA
    X_c = backbone_cls - backbone_cls.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n_pc = min(50, len(S))
    var = S ** 2 / (backbone_cls.shape[0] - 1)
    evr = var / var.sum()
    proj = U[:, :n_pc] * S[:n_pc]

    win_mask = y_win >= 0

    corr_value = np.array(cls_results["corr_value"])
    corr_win   = np.array(cls_results["corr_win"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Backbone CLS token -- probing & PCA", fontsize=12)

    # [0,0] PCA by value
    ax = axes[0, 0]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=backbone_val, cmap="coolwarm",
                    s=4, alpha=0.5, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="V(s)")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("CLS PCA -- colored by V(s)")

    # [0,1] PCA by win
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

    # [0,2] Explained variance
    ax = axes[0, 2]
    n_show = min(30, len(evr))
    ax.bar(range(n_show), evr[:n_show] * 100, color="#42A5F5")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Explained variance (top {n_show} PCs)")
    cumvar = np.cumsum(evr)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    ax.set_xlim(-0.5, n_show - 0.5)

    # [1,0] Per-dim corr with value
    ax = axes[1, 0]
    order_v = np.argsort(corr_value)[::-1]
    ax.bar(range(D_MODEL), corr_value[order_v], color="#1976D2", width=1.0)
    ax.set_xlabel("Dimension (sorted)")
    ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs V(s)")
    ax.axhline(0, color="black", lw=0.5)

    # [1,1] Per-dim corr with win
    ax = axes[1, 1]
    order_w = np.argsort(corr_win)[::-1]
    ax.bar(range(D_MODEL), corr_win[order_w], color="#388E3C", width=1.0)
    ax.set_xlabel("Dimension (sorted)")
    ax.set_ylabel("Pearson r")
    ax.set_title("CLS dim vs win label")
    ax.axhline(0, color="black", lw=0.5)

    # [1,2] Scatter corr_value vs corr_win
    ax = axes[1, 2]
    ax.scatter(corr_value, corr_win, s=8, alpha=0.6, color="#7B1FA2")
    ax.set_xlabel("corr(dim, V)")
    ax.set_ylabel("corr(dim, win)")
    profile = cls_results["profile_corr"]
    ax.set_title(f"Value vs Win corr per dim (profile r={profile:.3f})")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    lim = max(np.abs(corr_value).max(), np.abs(corr_win).max()) * 1.1
    if lim > 0:
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.tight_layout()
    fname = out_path.with_name(out_path.stem + "_cls")
    plt.savefig(fname.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.savefig(fname.with_suffix(".pdf"), bbox_inches="tight")
    print(f"CLS figure: {fname.with_suffix('.png')}")

    return {
        "evr_top10":    evr[:10].tolist(),
        "n_pc_90pct":   int(n90),
        "n_pc_95pct":   int(n95),
        "erank":        cls_results["effective_rank"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",      default=None,
                        help="Checkpoint .pt — required unless --cache is provided")
    parser.add_argument("--cache",           default=None,
                        help="Pre-built seq_all_cache.pt (from make_token_cache.py). "
                             "Skips rollout and backbone forward pass.")
    parser.add_argument("--n_envs",          type=int, default=32)
    parser.add_argument("--min_steps",       type=int, default=8192)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size",      type=int, default=256)
    parser.add_argument("--critic_n_layers", type=int, default=2)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    if args.cache is None and args.checkpoint is None:
        parser.error("Provide --checkpoint (to run rollout) or --cache (to load pre-built cache).")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(__file__).parent

    # ── Cache mode: load everything from disk ─────────────────────────────────
    if args.cache is not None:
        print(f"\nLoading cache: {args.cache}")
        saved = torch.load(args.cache, map_location="cpu", weights_only=False)

        required_labels = {"seq_all", "y_return", "y_win", "y_type1",
                           "y_item", "y_ability", "y_hp", "y_stats"}
        missing = required_labels - set(saved.keys())
        if missing:
            parser.error(
                f"Cache is missing keys: {missing}\n"
                "Re-run make_token_cache.py to generate a full cache with all labels."
            )

        seq_all = saved["seq_all"]
        N = int(saved.get("n_transitions", seq_all.shape[0]))
        print(f"  seq_all: {tuple(seq_all.shape)}  N={N}")

        def _np(key, dtype=None):
            v = saved[key]
            v = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
            return v.astype(dtype) if dtype else v

        labels = {
            "y_return":  _np("y_return",  np.float32),
            "y_win":     _np("y_win",     np.float32),
            "y_type1":   _np("y_type1",   np.int64),
            "y_item":    _np("y_item",    np.int64),
            "y_ability": _np("y_ability", np.int64),
            "y_hp":      _np("y_hp",      np.float32),
            "y_stats":   _np("y_stats",   np.float32),
        }

        # Backbone CLS (may be absent in older caches)
        if "backbone_cls" in saved:
            backbone_cls = _np("backbone_cls", np.float32)
            backbone_val = _np("backbone_values", np.float32).squeeze(-1)
            print(f"  backbone_cls: {backbone_cls.shape}")
        else:
            backbone_cls = None
            backbone_val = None
            print("  (no backbone_cls in cache — skipping CLS probes)")

    # ── Checkpoint mode: run rollout + backbone forward ───────────────────────
    else:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=True)
        agent = CynthAIAgent(
            use_independent_critic=True,
            critic_n_layers=args.critic_n_layers,
        ).to(device)
        missing, unexpected = agent.load_state_dict(ckpt["model"], strict=False)
        agent.eval()
        print(f"  update={ckpt.get('update', '?')}  "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        for p in agent.parameters():
            p.requires_grad_(False)

        print(f"\nCollecting rollout  n_envs={args.n_envs}  min_steps={args.min_steps} ...")
        buffer = collect_rollout(
            agent_self = agent,
            agent_opp  = agent,
            n_envs     = args.n_envs,
            min_steps  = args.min_steps,
            gamma      = 0.99,
            lam        = 0.95,
            device     = device,
        )
        N = len(buffer)
        print(f"  collected {N} transitions")

        print("\nCaching seq_all + backbone CLS ...")
        seq_all, backbone_cls_t, backbone_val_t = cache_tokens(
            agent, buffer, device, batch_size=args.batch_size
        )
        backbone_cls = backbone_cls_t.numpy()
        backbone_val = backbone_val_t.squeeze(-1).numpy()
        print(f"  seq_all:      {tuple(seq_all.shape)}")
        print(f"  backbone_cls: {backbone_cls.shape}")

        print("\nExtracting labels ...")
        labels = extract_labels(buffer)

    n_win_known = int((labels["y_win"] >= 0).sum())
    print(f"  returns: mean={labels['y_return'].mean():.3f}  std={labels['y_return'].std():.3f}")
    print(f"  win labels known: {n_win_known}/{N} ({100*n_win_known/N:.1f}%)  "
          f"win_rate={labels['y_win'][labels['y_win']>=0].mean():.3f}")

    # ── Train / val split 80/20 ───────────────────────────────────────────────
    rng       = np.random.default_rng(args.seed)
    perm      = rng.permutation(N).tolist()
    n_train   = int(0.8 * N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}")

    # ── Run probes ────────────────────────────────────────────────────────────
    print(f"\nRunning probes ({SEQ_LEN} tokens + 2 mean pools + 12 slot pools) ...")
    results = run_probes(seq_all, labels, train_idx, val_idx)

    # ── Console table ─────────────────────────────────────────────────────────
    print_table(results, N)

    # ── Backbone CLS probes ────────────────────────────────────────────────────
    cls_results = None
    if backbone_cls is not None:
        agg_labels = extract_aggregate_labels(labels)
        cls_results = run_cls_probes(
            backbone_cls, backbone_val, labels, agg_labels, train_idx, val_idx,
        )
        cls_pca = save_cls_figure(
            backbone_cls, backbone_val, labels["y_win"], cls_results,
            out_dir / "probing",
        )
        # Attach summary to results (strip large per-dim arrays)
        cls_summary = {k: v for k, v in cls_results.items()
                       if k not in ("corr_value", "corr_win")}
        cls_summary["pca"] = cls_pca
        results["backbone_cls"] = cls_summary

    # ── Cross-token matrix (12×12 current-turn) ───────────────────────────────
    print(f"\nRunning cross-token matrix (12×12 = 144 source×target pairs) ...")
    cross = run_cross_token_matrix(seq_all, labels, train_idx, val_idx)

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_json(results, N, out_dir / "probing.json")
    save_figure(results, out_dir / "probing")

    cross_out = {"n_transitions": N, "cross_token_matrix": cross}
    with open(out_dir / "probing_cross.json", "w") as f:
        json.dump(cross_out, f, indent=2)
    print(f"Cross-token JSON: {out_dir / 'probing_cross.json'}")
    save_cross_matrix_figure(cross, out_dir / "probing_cross")

    # ── Save minimal token cache (checkpoint mode only — don't overwrite full cache) ──
    if args.cache is None:
        cache_path = out_dir / "seq_all_cache.pt"
        torch.save({"seq_all": seq_all, "n_transitions": N}, cache_path)
        print(f"Token cache saved: {cache_path}")


if __name__ == "__main__":
    main()
