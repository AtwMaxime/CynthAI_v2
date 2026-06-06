"""
Shared helpers for the probing package.

- Probe helpers: regression, classification, effective rank, per-dim correlation
- Constants: SEQ_TOKEN_LABELS, SLOT_NAMES, QUERY_LABELS
- Label extraction from rollout buffers
- Token caching (backbone + critic forward pass)
- Cache loading
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model.backbone import K_TURNS, N_SLOTS, D_MODEL, SEQ_LEN

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEQ_TOKEN_LABELS: list[str] = []
for _t in range(K_TURNS):
    for _s in range(N_SLOTS):
        if _s < 6:
            SEQ_TOKEN_LABELS.append(f"T{_t}:O{_s}")
        elif _s < 12:
            SEQ_TOKEN_LABELS.append(f"T{_t}:P{_s-6}")
        else:
            SEQ_TOKEN_LABELS.append(f"T{_t}:F")

assert len(SEQ_TOKEN_LABELS) == SEQ_LEN

SLOT_NAMES = [f"OWN{s}" if s < 6 else f"OPP{s-6}" for s in range(12)]

QUERY_LABELS = (
    [f"M{i}" for i in range(4)]
    + [f"MM{i}" for i in range(4)]
    + [f"SW{i}" for i in range(5)]
)
QUERY_COLORS = ["#2196F3"] * 4 + ["#9C27B0"] * 4 + ["#4CAF50"] * 5

# Bar colors for 52-token plots: OWN=blue, OPP=red, FIELD=gray, per turn
_SLOT_COLORS = ["#2196F3"] * 6 + ["#E53935"] * 6 + ["#9E9E9E"]
BAR_COLORS_52 = _SLOT_COLORS * K_TURNS

# Turn separator positions for 52-token plots
TURN_SEPS = [N_SLOTS * t - 0.5 for t in range(1, K_TURNS)]


# ─────────────────────────────────────────────────────────────────────────────
# Probe helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_regression(
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
    if ss_tot < 1e-6:
        return float("nan"), 0.0
    r2 = max(-1.0, float(1.0 - ss_res / ss_tot))

    corr = (
        float(np.corrcoef(pred, y_val)[0, 1])
        if np.std(pred) > 1e-10 and np.std(y_val) > 1e-10
        else 0.0
    )
    return r2, corr


def fit_classification(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    max_iter: int = 500,
    compute_auc: bool = False,
) -> tuple[float, float]:
    """Logistic regression. Returns (val_accuracy, val_auc)."""
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


def fit_ridge_r2(X_tr, y_tr, X_val, y_val) -> float:
    """Ridge regression → val R² only (no correlation)."""
    r2, _ = fit_regression(X_tr, y_tr, X_val, y_val)
    return r2


def fit_logistic_auc(X_tr, y_tr, X_val, y_val, max_iter=500) -> float:
    """Binary logistic regression → AUC-ROC. Returns 0.5 on failure."""
    _, auc = fit_classification(X_tr, y_tr, X_val, y_val,
                                max_iter=max_iter, compute_auc=True)
    return auc


def fit_logistic_acc(X_tr, y_tr, X_val, y_val, max_iter=500) -> float:
    """Multi-class logistic regression → top-1 accuracy."""
    acc, _ = fit_classification(X_tr, y_tr, X_val, y_val, max_iter=max_iter)
    return acc


def effective_rank(X: np.ndarray) -> float:
    """
    Effective rank = exp(H(sigma_norm)) where sigma are singular values
    of centered X and H is Shannon entropy of the normalised spectrum.
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


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_turn_vlines(ax, color="gray", lw=0.8, ls="--", alpha=0.4):
    """Draw vertical separators between turns on a bar/line plot."""
    for xs in TURN_SEPS:
        ax.axvline(xs, color=color, lw=lw, ls=ls, alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Label extraction from rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

def extract_labels(buffer) -> dict[str, np.ndarray]:
    """
    Extract probe targets from rollout buffer transitions.

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

    y_return = np.array(buffer._returns, dtype=np.float32)

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

    all_type1   = torch.stack([tr.type1_idx   for tr in transitions])
    all_item    = torch.stack([tr.item_idx    for tr in transitions])
    all_ability = torch.stack([tr.ability_idx for tr in transitions])
    all_scalars = torch.stack([tr.scalars     for tr in transitions])

    return {
        "y_return":  y_return,
        "y_win":     y_win,
        "y_type1":   all_type1.numpy().astype(np.int64),
        "y_item":    all_item.numpy().astype(np.int64),
        "y_ability": all_ability.numpy().astype(np.int64),
        "y_hp":      all_scalars[:, :, 1].numpy().astype(np.float32),
        "y_stats":   all_scalars[:, :, 3:8].numpy().astype(np.float32),
    }


def extract_next_hp_labels(buffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute next-turn HP for each transition.

    Returns (all float32, NaN where no next step):
        next_hp_own : [N, 6]
        next_hp_opp : [N, 6]
        next_valid  : [N]  bool
    """
    transitions = buffer._transitions
    N = len(transitions)

    OWN_BASE = (K_TURNS - 1) * 12
    OPP_BASE = (K_TURNS - 1) * 12 + 6

    next_hp_own = np.full((N, 6), np.nan, dtype=np.float32)
    next_hp_opp = np.full((N, 6), np.nan, dtype=np.float32)
    next_valid  = np.zeros(N, dtype=bool)

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
                continue
            sc_next = tr_next.scalars
            for j in range(6):
                next_hp_own[n, j] = sc_next[OWN_BASE + j, 1].item()
                next_hp_opp[n, j] = sc_next[OPP_BASE + j, 1].item()
            next_valid[n] = True

    return next_hp_own, next_hp_opp, next_valid


def extract_aggregate_labels(labels: dict) -> dict:
    """
    Derive aggregate labels from per-pokemon labels.

    Returns: mean_hp_own, mean_hp_opp, alive_own, alive_opp — all [N].
    """
    y_hp = labels["y_hp"]
    cur_own = y_hp[:, (K_TURNS - 1) * 12:(K_TURNS - 1) * 12 + 6]
    cur_opp = y_hp[:, (K_TURNS - 1) * 12 + 6:(K_TURNS - 1) * 12 + 12]
    return {
        "mean_hp_own": cur_own.mean(axis=1).astype(np.float32),
        "mean_hp_opp": cur_opp.mean(axis=1).astype(np.float32),
        "alive_own":   (cur_own > 0).sum(axis=1).astype(np.float32),
        "alive_opp":   (cur_opp > 0).sum(axis=1).astype(np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Token caching (from agent + buffer)
# ─────────────────────────────────────────────────────────────────────────────

def cache_tokens(agent, buffer, device, batch_size=256):
    """
    Run poke_emb + backbone.encode(return_full_seq=True) over all transitions.

    Returns (CPU tensors):
        seq_all      : [N, 52, D_MODEL]
        backbone_cls : [N, D_MODEL]
        backbone_val : [N, 1]
    """
    agent.eval()
    n = len(buffer)
    all_seq = []
    all_cls = []
    all_val = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = buffer._gather(list(range(start, min(start + batch_size, n))), device)
            pt = agent.poke_emb(batch["poke_batch"])
            ft = batch["field_tensor"]
            _, _, value, _, seq, cls_out = agent.backbone.encode(pt, ft, return_full_seq=True)
            all_seq.append(seq.cpu())
            all_cls.append(cls_out.cpu())
            all_val.append(value.cpu())

    return (
        torch.cat(all_seq, dim=0),
        torch.cat(all_cls, dim=0),
        torch.cat(all_val, dim=0),
    )


def cache_tokens_full(agent, buffer, device, batch_size=256):
    """
    Run poke_emb + backbone.encode + backbone.act(return_queries=True).

    Returns (all CPU tensors):
        seq_all      : [N, 52, D_MODEL]
        detr_queries : [N, 13, D_MODEL]
        actions      : [N]
        backbone_cls : [N, D_MODEL]
        backbone_val : [N, 1]
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
        torch.cat(all_seq,     dim=0),
        torch.cat(all_queries, dim=0),
        all_actions,
        torch.cat(all_cls,     dim=0),
        torch.cat(all_val,     dim=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cache loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cache(path: str | Path) -> dict:
    """Load a token cache and convert tensors to numpy where appropriate."""
    path = Path(path)
    print(f"Loading cache: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    N = int(data.get("n_transitions", data["seq_all"].shape[0]))
    print(f"  N={N}  keys={sorted(data.keys())}")
    return data


def numpy_from_cache(data: dict, key: str, dtype=None) -> np.ndarray | None:
    """Extract a key from cache as numpy, or None if missing."""
    if key not in data:
        return None
    v = data[key]
    v = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
    return v.astype(dtype) if dtype else v


def get_labels_from_cache(data: dict) -> dict[str, np.ndarray]:
    """Extract label arrays from cache dict."""
    return {
        "y_return":  numpy_from_cache(data, "y_return",  np.float32),
        "y_win":     numpy_from_cache(data, "y_win",     np.float32),
        "y_type1":   numpy_from_cache(data, "y_type1",   np.int64),
        "y_item":    numpy_from_cache(data, "y_item",    np.int64),
        "y_ability": numpy_from_cache(data, "y_ability", np.int64),
        "y_hp":      numpy_from_cache(data, "y_hp",      np.float32),
        "y_stats":   numpy_from_cache(data, "y_stats",   np.float32),
    }


def check_labels(labels: dict) -> bool:
    """Return True if all required label keys are present and non-None."""
    required = ("y_return", "y_win", "y_type1", "y_item", "y_ability", "y_hp", "y_stats")
    return all(labels.get(k) is not None for k in required)
