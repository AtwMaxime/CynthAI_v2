"""
offline_critic_fit.py — Diagnostic : le critic peut-il apprendre les returns GAE offline ?

Protocole :
  1. Charger un checkpoint.
  2. Collecter un rollout (self-play symétrique, ~50% WR).
  3. Cacher poke_tokens + field_tensor (no_grad, une seule passe).
  4. Split train/val 80/20 (indices fixes, reproductible).
  5. Stats initiales (EV, corr, r²) sur train et val.
  6. Pour chaque LR ∈ [1e-4, 1e-3, 5e-3] :
       - Réinitialiser le critic aux poids initiaux
       - AdamW, N=300 epochs, batch_size=256
       - Logger EV / corr / r² / MSE après chaque epoch (train)
  7. Stats finales (train + val) pour chaque LR.
  8. Résumé console + figure 2×2.

Usage :
    cd /local_scratch/mattwood/projects/rl_agent/CynthAI_v2
    python diag/offline_critic_fit.py \\
        --checkpoint checkpoints/cheater_v7/agent_001300.pt \\
        --n_envs 32 --min_steps 8192 \\
        --n_epochs 300 --batch_size 256 \\
        --device cuda
    Output : diag/offline_critic_fit.png
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.agent import CynthAIAgent
from training.rollout import collect_rollout


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ev(pred: np.ndarray, target: np.ndarray) -> float:
    """Explained variance."""
    var_res = float(np.var(target - pred))
    var_tgt = float(np.var(target))
    if var_tgt < 1e-10:
        return 0.0
    return float(1.0 - var_res / var_tgt)


def _corr(pred: np.ndarray, target: np.ndarray) -> float:
    if np.std(pred) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def eval_critic(
    critic: nn.Module,
    poke_tokens: torch.Tensor,
    field_tensors: torch.Tensor,
    norm_returns: np.ndarray,
    indices: list[int],
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, float, float, float, float]:
    """Run critic on a subset of cached data, return (preds, EV, corr, r2, MSE)."""
    critic.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            pt = poke_tokens[idx].to(device)
            ft = field_tensors[idx].to(device)
            v  = critic(pt, ft).squeeze(-1).cpu().numpy()
            preds.append(v)
    preds = np.concatenate(preds)
    tgt   = norm_returns[indices]
    return preds, _ev(preds, tgt), _corr(preds, tgt), _r2(preds, tgt), _mse(preds, tgt)


# ─────────────────────────────────────────────────────────────────────────────
# Cache poke_tokens + field_tensor from buffer transitions
# ─────────────────────────────────────────────────────────────────────────────

def cache_embeddings(
    agent: CynthAIAgent,
    buffer,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run poke_emb in no_grad batches over all transitions.
    Returns:
        poke_tokens  : [N, K*12, TOKEN_DIM]  (CPU)
        field_tensors: [N, K, FIELD_DIM]      (CPU)
    """
    from model.backbone import K_TURNS
    from model.embeddings import FIELD_DIM, collate_features, collate_field_features

    agent.eval()
    n = len(buffer)
    all_poke, all_field = [], []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = buffer._gather(list(range(start, min(start + batch_size, n))), device)
            pt = agent.poke_emb(batch["poke_batch"])        # [B, K*12, TOKEN_DIM]
            ft = batch["field_tensor"]                       # [B, K, FIELD_DIM]
            all_poke.append(pt.cpu())
            all_field.append(ft.cpu())

    return torch.cat(all_poke, dim=0), torch.cat(all_field, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Single training run for one LR
# ─────────────────────────────────────────────────────────────────────────────

def train_critic(
    critic: nn.Module,
    poke_tokens: torch.Tensor,
    field_tensors: torch.Tensor,
    norm_returns: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    lr: float,
    n_epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Train critic offline; return per-epoch metrics on train AND val sets."""
    opt = torch.optim.AdamW(critic.parameters(), lr=lr)
    norm_ret_t = torch.tensor(norm_returns, dtype=torch.float32)

    logs: dict[str, list[float]] = {
        "ev": [], "corr": [], "r2": [], "mse": [],
        "ev_val": [], "corr_val": [], "r2_val": [], "mse_val": [],
    }

    for epoch in range(n_epochs):
        critic.train()
        idxs = train_idx.copy()
        random.shuffle(idxs)

        for start in range(0, len(idxs) - batch_size + 1, batch_size):
            b_idx = idxs[start : start + batch_size]
            pt  = poke_tokens[b_idx].to(device)
            ft  = field_tensors[b_idx].to(device)
            ret = norm_ret_t[b_idx].to(device)

            v   = critic(pt, ft).squeeze(-1)
            loss = nn.functional.mse_loss(v, ret)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()

        # Log on train + val after each epoch
        _, ev, corr, r2, mse = eval_critic(
            critic, poke_tokens, field_tensors, norm_returns, train_idx,
            batch_size=512, device=device,
        )
        _, ev_v, corr_v, r2_v, mse_v = eval_critic(
            critic, poke_tokens, field_tensors, norm_returns, val_idx,
            batch_size=512, device=device,
        )
        logs["ev"].append(ev);         logs["corr"].append(corr)
        logs["r2"].append(r2);         logs["mse"].append(mse)
        logs["ev_val"].append(ev_v);   logs["corr_val"].append(corr_v)
        logs["r2_val"].append(r2_v);   logs["mse_val"].append(mse_v)

        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{n_epochs}  "
                  f"EV={ev:.3f}  EV_val={ev_v:.3f}  corr_val={corr_v:.3f}  MSE={mse:.4f}")

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--n_envs",      type=int,   default=32)
    parser.add_argument("--min_steps",   type=int,   default=8192)
    parser.add_argument("--n_epochs",    type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--critic_n_layers", type=int, default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── 1. Load checkpoint ────────────────────────────────────────────────────
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

    # ── 2. Collect rollout ────────────────────────────────────────────────────
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

    # ── 3. Extract returns ────────────────────────────────────────────────────
    raw_returns  = np.array(buffer._returns, dtype=np.float32)
    norm_returns = raw_returns   # returns are no longer Z-scored

    print(f"\nReturns       : mean={raw_returns.mean():.3f}  std={raw_returns.std():.3f}"
          f"  min={raw_returns.min():.3f}  max={raw_returns.max():.3f}")

    # ── 4. Cache embeddings ───────────────────────────────────────────────────
    print("\nCaching poke_tokens + field_tensors ...")
    poke_tokens, field_tensors = cache_embeddings(agent, buffer, device, batch_size=256)
    print(f"  poke_tokens:   {tuple(poke_tokens.shape)}")
    print(f"  field_tensors: {tuple(field_tensors.shape)}")

    # ── 5. Train / val split ──────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    all_idx  = list(range(N))
    perm     = rng.permutation(N).tolist()
    n_train  = int(0.8 * N)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    print(f"\nSplit: train={len(train_idx)}  val={len(val_idx)}")

    # ── 6. Initial stats ──────────────────────────────────────────────────────
    critic_init = agent.independent_critic
    init_weights = copy.deepcopy(critic_init.state_dict())

    _, ev_init_tr, corr_init_tr, r2_init_tr, mse_init_tr = eval_critic(
        critic_init, poke_tokens, field_tensors, norm_returns, train_idx,
        batch_size=512, device=device,
    )
    _, ev_init_val, corr_init_val, r2_init_val, mse_init_val = eval_critic(
        critic_init, poke_tokens, field_tensors, norm_returns, val_idx,
        batch_size=512, device=device,
    )
    mse_baseline = float(np.var(norm_returns))  # ≈ 1.0 by Z-score definition

    print(f"\n{'':20s}  {'TRAIN':^30s}  {'VAL':^30s}")
    print(f"{'':20s}  {'EV':>7s}  {'corr':>7s}  {'r²':>7s}  {'MSE':>7s}    "
          f"{'EV':>7s}  {'corr':>7s}  {'r²':>7s}  {'MSE':>7s}")
    print(f"{'Baseline':20s}  {'0.000':>7s}  {'—':>7s}  {'0.000':>7s}  {mse_baseline:>7.4f}    "
          f"{'0.000':>7s}  {'—':>7s}  {'0.000':>7s}  {mse_baseline:>7.4f}")
    print(f"{'Init':20s}  {ev_init_tr:>7.3f}  {corr_init_tr:>7.3f}  {r2_init_tr:>7.3f}  {mse_init_tr:>7.4f}    "
          f"{ev_init_val:>7.3f}  {corr_init_val:>7.3f}  {r2_init_val:>7.3f}  {mse_init_val:>7.4f}")

    # ── 7. Freeze all except independent_critic ───────────────────────────────
    for p in agent.parameters():
        p.requires_grad = False
    for p in critic_init.parameters():
        p.requires_grad = True

    # ── 8. Train for each LR ──────────────────────────────────────────────────
    lr_list    = [1e-4, 1e-3, 5e-3]
    all_logs   = {}
    final_rows = {}

    for lr in lr_list:
        label = f"LR={lr:.0e}"
        print(f"\n{'─'*60}")
        print(f"Training  {label}  ({args.n_epochs} epochs, batch={args.batch_size})")

        # Reset critic to initial weights
        critic_init.load_state_dict(copy.deepcopy(init_weights))

        logs = train_critic(
            critic_init, poke_tokens, field_tensors, norm_returns,
            train_idx, val_idx,
            lr=lr, n_epochs=args.n_epochs, batch_size=args.batch_size, device=device,
        )
        all_logs[label] = logs

        # Final stats (train + val)
        _, ev_tr, corr_tr, r2_tr, mse_tr = eval_critic(
            critic_init, poke_tokens, field_tensors, norm_returns, train_idx,
            batch_size=512, device=device,
        )
        _, ev_val, corr_val, r2_val, mse_val = eval_critic(
            critic_init, poke_tokens, field_tensors, norm_returns, val_idx,
            batch_size=512, device=device,
        )
        final_rows[label] = dict(
            ev_tr=ev_tr, corr_tr=corr_tr, r2_tr=r2_tr, mse_tr=mse_tr,
            ev_val=ev_val, corr_val=corr_val, r2_val=r2_val, mse_val=mse_val,
        )
        print(f"  Final →  TRAIN EV={ev_tr:.3f} corr={corr_tr:.3f} r²={r2_tr:.3f} MSE={mse_tr:.4f}"
              f"   VAL EV={ev_val:.3f} corr={corr_val:.3f} r²={r2_val:.3f} MSE={mse_val:.4f}")

    # ── 9. Summary table ──────────────────────────────────────────────────────
    best_ev_val = max(v["ev_val"] for v in final_rows.values())
    best_mse_tr = min(v["mse_tr"] for v in final_rows.values())

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Raw returns   : mean={raw_returns.mean():.2f}  std={raw_returns.std():.2f}"
          f"  min={raw_returns.min():.2f}  max={raw_returns.max():.2f}")
    print(f"Returns       : mean≈{norm_returns.mean():.2f}  std≈{norm_returns.std():.2f}  (raw scale)")
    print()
    print(f"{'':20s}  {'TRAIN':^38s}  {'VAL':^38s}")
    print(f"{'':20s}  {'EV':>7s}  {'corr':>7s}  {'r²':>7s}  {'MSE':>7s}    "
          f"{'EV':>7s}  {'corr':>7s}  {'r²':>7s}  {'MSE':>7s}")
    print(f"{'Baseline':20s}  {'0.000':>7s}  {'—':>7s}  {'0.000':>7s}  {mse_baseline:>7.4f}    "
          f"{'0.000':>7s}  {'—':>7s}  {'0.000':>7s}  {mse_baseline:>7.4f}")
    print(f"{'Init':20s}  {ev_init_tr:>7.3f}  {corr_init_tr:>7.3f}  {r2_init_tr:>7.3f}  {mse_init_tr:>7.4f}    "
          f"{ev_init_val:>7.3f}  {corr_init_val:>7.3f}  {r2_init_val:>7.3f}  {mse_init_val:>7.4f}")
    for label, row in final_rows.items():
        print(f"{label:20s}  {row['ev_tr']:>7.3f}  {row['corr_tr']:>7.3f}  "
              f"{row['r2_tr']:>7.3f}  {row['mse_tr']:>7.4f}    "
              f"{row['ev_val']:>7.3f}  {row['corr_val']:>7.3f}  "
              f"{row['r2_val']:>7.3f}  {row['mse_val']:>7.4f}")
    print()
    print(f"MSE baseline         : {mse_baseline:.4f}")
    print(f"MSE init             : {mse_init_tr:.4f}")
    print(f"MSE final (best LR)  : {best_mse_tr:.4f}")
    print(f"Best EV val          : {best_ev_val:.3f}")

    # Best val EV across all LRs and all epochs (not just final)
    best_ev_val_any = max(
        v for logs in all_logs.values() for v in logs["ev_val"]
    )
    best_ev_val_early = max(
        v for logs in all_logs.values() for v in logs["ev_val"][:10]
    )
    print(f"Best EV val (any epoch)   : {best_ev_val_any:.3f}")
    print(f"Best EV val (epochs 1-10) : {best_ev_val_early:.3f}  ← PPO-relevant")

    if best_ev_val_early > 0.1:
        print("\n→ EV_val peaks > 0.1 in first 10 epochs : critic learns online")
        print("  Overfitting dominates at 100+ epochs — expected with 1.25M params / 6k samples")
        print("  Recommendation: tune critic_updates_per_ppo + weight_decay in v8")
    elif best_ev_val_any > 0.1:
        print("\n→ EV_val > 0.1 but only after many epochs : critic too slow to learn online")
        print("  Recommendation: higher LR or critic_n_layers=4 in v8")
    else:
        print("\n→ EV_val < 0.1 everywhere : intrinsic return variance too high")
        print("  Recommendation: wait for MC estimate or increase rollout size")

    # ── 10. Plots ─────────────────────────────────────────────────────────────
    colors = {"LR=1e-04": "#2196F3", "LR=1e-03": "#FF9800", "LR=5e-03": "#E91E63"}
    epochs  = list(range(1, args.n_epochs + 1))
    zoom_n  = min(50, args.n_epochs)   # zoom window for early epochs

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Offline critic fit — {Path(args.checkpoint).stem}  "
        f"(N={N}, train={len(train_idx)}, val={len(val_idx)})",
        fontsize=12,
    )

    # [0,0] EV train vs epoch (full)
    ax = axes[0, 0]
    ax.set_title("EV vs epoch — TRAIN")
    ax.axhline(0, color="gray", ls="--", lw=0.8, label="Baseline")
    ax.axhline(ev_init_tr, color="black", ls=":", lw=1.2, label=f"Init={ev_init_tr:.3f}")
    for label, logs in all_logs.items():
        ax.plot(epochs, logs["ev"], color=colors.get(label, None), label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("EV")
    ax.legend(fontsize=8)

    # [0,1] EV val vs epoch (full)
    ax = axes[0, 1]
    ax.set_title("EV vs epoch — VAL")
    ax.axhline(0, color="gray", ls="--", lw=0.8, label="Baseline")
    ax.axhline(ev_init_val, color="black", ls=":", lw=1.2, label=f"Init={ev_init_val:.3f}")
    for label, logs in all_logs.items():
        ax.plot(epochs, logs["ev_val"], color=colors.get(label, None), label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("EV")
    ax.legend(fontsize=8)

    # [1,0] EV val zoomed — first 50 epochs (the PPO-relevant window)
    ax = axes[1, 0]
    ax.set_title(f"EV val — zoomed (first {zoom_n} epochs)  ← PPO-relevant")
    ax.axhline(0, color="gray", ls="--", lw=0.8, label="Baseline")
    ax.axhline(ev_init_val, color="black", ls=":", lw=1.2, label=f"Init={ev_init_val:.3f}")
    for label, logs in all_logs.items():
        ax.plot(epochs[:zoom_n], logs["ev_val"][:zoom_n],
                color=colors.get(label, None), label=label, lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("EV val")
    ax.legend(fontsize=8)

    # [1,1] corr val vs epoch
    ax = axes[1, 1]
    ax.set_title("corr (Pearson) vs epoch — VAL")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(corr_init_val, color="black", ls=":", lw=1.2, label=f"Init={corr_init_val:.3f}")
    for label, logs in all_logs.items():
        ax.plot(epochs, logs["corr_val"], color=colors.get(label, None), label=label)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Pearson r")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = Path(__file__).parent / "offline_critic_fit.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nPlot saved: {out_png}")

    # ── 11. Save JSON results ──────────────────────────────────────────────────
    import json
    results = {
        "checkpoint": args.checkpoint,
        "n_transitions": N,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "returns": {
            "raw_mean": float(raw_returns.mean()),
            "raw_std":  float(raw_returns.std()),
            "raw_min":  float(raw_returns.min()),
            "raw_max":  float(raw_returns.max()),
            "ret_mean": float(raw_returns.mean()),
            "ret_std":  float(raw_returns.std()),
        },
        "init": {
            "train": {"ev": ev_init_tr, "corr": corr_init_tr, "r2": r2_init_tr, "mse": mse_init_tr},
            "val":   {"ev": ev_init_val, "corr": corr_init_val, "r2": r2_init_val, "mse": mse_init_val},
        },
        "mse_baseline": mse_baseline,
        "runs": {
            label: {
                "lr": float(label.split("=")[1]),
                "final": final_rows[label],
                "per_epoch": {
                    "train": {k: all_logs[label][k] for k in ("ev", "corr", "r2", "mse")},
                    "val":   {k: all_logs[label][k] for k in ("ev_val", "corr_val", "r2_val", "mse_val")},
                },
            }
            for label in all_logs
        },
        "best_ev_val_any_epoch":   best_ev_val_any,
        "best_ev_val_early_10ep":  best_ev_val_early,
    }
    out_json = Path(__file__).parent / "offline_critic_fit.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()
