"""
Diagnostic complet : similarité cosinus des 5 matrices clés.

Mesure :
  A @ A^T  — tokens AVANT transformer (current turn)
  B @ B^T  — tokens APRÈS transformer (= keys pour cross-attention)
  C @ C^T  — action queries
  B @ C^T  — keys vs queries (diagnostic direct du cross-attention)
  A @ C^T  — pre-transformer vs queries

Usage:
    python diag_cosine.py --checkpoint NONE --n-battles 64
    python diag_cosine.py --checkpoint checkpoints/cheater/agent_001600.pt --n-battles 64
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import numpy as np

from model.agent import CynthAIAgent
from training.rollout import encode_state, build_action_mask
from model.embeddings import collate_features, collate_field_features, FieldBatch, FIELD_DIM

# ── Labels ─────────────────────────────────────────────────────────────────────
TOKEN_LABELS  = ["O0","O1","O2","O3","O4","O5",
                 "P0","P1","P2","P3","P4","P5","FL"]
ACTION_LABELS = ["M1","M2","M3","M4",
                 "T1","T2","T3","T4",
                 "S0","S1","S2","S3","S4"]

# ── Similarity helpers ─────────────────────────────────────────────────────────

def cos_sim_self(x: torch.Tensor) -> torch.Tensor:
    """x: [B, N, D]  ->  [N, N] mean self cosine similarity."""
    xn = F.normalize(x, dim=-1)
    return (xn @ xn.transpose(-1, -2)).mean(dim=0)

def cos_sim_cross(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: [B, N, D], y: [B, M, D]  ->  [N, M] mean cross cosine similarity."""
    xn = F.normalize(x, dim=-1)
    yn = F.normalize(y, dim=-1)
    return (xn @ yn.transpose(-1, -2)).mean(dim=0)

# ── Stats ──────────────────────────────────────────────────────────────────────

def off_diag_stats(sim: torch.Tensor) -> dict:
    """Off-diagonal summary for a square [N, N] matrix."""
    N = sim.shape[0]
    off = sim.clone()
    off.fill_diagonal_(float("nan"))
    vals = off[~torch.isnan(off)]
    return {
        "mean": vals.mean().item(),
        "std":  vals.std().item(),
        "min":  vals.min().item(),
        "max":  vals.max().item(),
        "diag": sim.diag().mean().item(),
    }

def print_stats(sim: torch.Tensor, label: str):
    s = off_diag_stats(sim)
    print(f"  {label}: off-diag mean={s['mean']:.4f} std={s['std']:.4f} "
          f"min={s['min']:.4f} max={s['max']:.4f} diag={s['diag']:.4f}")

# ── Pretty-print ───────────────────────────────────────────────────────────────

def print_matrix(sim: torch.Tensor, row_labels: list[str],
                 col_labels: list[str] | None = None, title: str = ""):
    if col_labels is None:
        col_labels = row_labels
    print(f"\n  {title}")
    header = "       " + "".join(f"{l:>6}" for l in col_labels)
    print(f"  {header}")
    for i, row in enumerate(sim):
        row_str = "".join(f"{v:>6.3f}" for v in row)
        print(f"  {row_labels[i]:>4s}: {row_str}")

    if row_labels == col_labels:
        # Square matrix: per-group off-diag means
        idx_map = {
            "Own": (0, 6), "Opp": (6, 12), "Field": (12, 13),
            "Move": (0, 4), "Mech": (4, 8), "Switch": (8, 13),
        }
        print(f"  -- per-group within --")
        for gname, (gi, gj) in idx_map.items():
            if gj > sim.shape[0]:
                continue
            g = sim[gi:gj, gi:gj]
            if g.numel() <= 1:
                continue
            off = g.clone()
            off.fill_diagonal_(float("nan"))
            vals = off[~torch.isnan(off)]
            if vals.numel() > 0:
                print(f"    {gname:>7s}: {vals.mean().item():.4f}")
    else:
        # Rectangular: per-row argmax + per-group block means
        print(f"  -- row argmax (key index) --")
        for i, row in enumerate(sim):
            topk = row.argsort(descending=True)[:3]
            top_str = ",".join(f"{col_labels[t]}({row[t]:.3f})" for t in topk)
            print(f"    {row_labels[i]:>4s} -> {top_str}")

        # Block means
        blocks = [
            ("Moves->Own", 0, 4, 0, 6),
            ("Moves->Opp", 0, 4, 6, 12),
            ("Moves->Field", 0, 4, 12, 13),
            ("Mech->Own", 4, 8, 0, 6),
            ("Mech->Opp", 4, 8, 6, 12),
            ("Switch->Own", 8, 13, 0, 6),
            ("Switch->Opp", 8, 13, 6, 12),
            ("Switch->Field", 8, 13, 12, 13),
        ]
        print(f"  -- block means (query->key) --")
        for bname, qi, qj, ki, kj in blocks:
            block = sim[qi:qj, ki:kj]
            if block.numel() > 0:
                print(f"    {bname:>14s}: {block.mean().item():.4f}")

# ── Heatmap save ───────────────────────────────────────────────────────────────

def save_heatmap(sim: torch.Tensor, row_labels: list[str],
                 col_labels: list[str] | None = None,
                 filename: str = "heatmap.png", title: str = ""):
    """Save a cosine similarity matrix as a PNG heatmap via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if col_labels is None:
        col_labels = row_labels

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.5),
                                    max(5, len(row_labels) * 0.5)))
    im = ax.imshow(sim.cpu().numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, fontsize=7)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = sim[i, j].item()
            color = "white" if abs(val) > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("cosine similarity", fontsize=8)
    ax.set_title(title, fontsize=10)

    out_dir = Path("diag")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="NONE")
    parser.add_argument("--n-battles", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  Checkpoint: {args.checkpoint}")

    # ── Model ──────────────────────────────────────────────────────────────────
    agent = CynthAIAgent().to(device)
    agent.eval()
    if args.checkpoint != "NONE":
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        agent.load_state_dict(sd)
        print(f"Loaded: {sum(p.numel() for p in agent.parameters()):,} params")
    else:
        print(f"Fresh model: {sum(p.numel() for p in agent.parameters()):,} params")

    # ── Collect states ─────────────────────────────────────────────────────────
    from simulator import PyBattle
    fmt = "gen9randombattle"

    poke_feats_list = []
    field_feats_list = []
    move_idx_list = []
    pp_ratio_list = []
    move_disabled_list = []
    mech_id_list = []
    mech_type_list = []

    n_envs = min(args.n_battles, 64)
    seeds = [hash(f"d_{i}") % (2**31) for i in range(n_envs)]
    envs = [PyBattle(fmt, seed=s) for s in seeds]
    done = [False] * n_envs
    collected = 0
    steps = 0

    while collected < args.n_battles and steps < 500:
        steps += 1
        for i in range(n_envs):
            if done[i]:
                continue
            try:
                state = envs[i].get_state()
            except Exception:
                done[i] = True; continue

            pf, ff = encode_state(state, 0)
            poke_feats_list.append(pf)
            field_feats_list.append(ff)

            msk = build_action_mask(state, 0)
            # don't need mask for the diagnostic, just move info
            act = pf[0]
            move_idx_list.append(act.move_indices[:4])
            pp_ratio_list.append(act.move_pp[:4])
            move_disabled_list.append(act.move_disabled[:4])
            mech_id_list.append(0)
            mech_type_list.append(0)
            collected += 1
            if collected >= args.n_battles:
                break

            try:
                legal = state["sides"][0].get("request_state", "")
                if legal in ("Move", "Switch"):
                    envs[i].step(torch.randint(0, 13, (1,)).item())
                else:
                    done[i] = True
            except Exception:
                done[i] = True

    print(f"Collected {collected} states in {steps} env steps")

    # ── Build batch (K-turn history = repeat each state K times) ───────────────
    K = 4
    poke_batches = []
    field_flat = []
    for pf, ff in zip(poke_feats_list, field_feats_list):
        turn = []
        for _ in range(K):
            turn.extend(pf)
            field_flat.append(ff)
        poke_batches.append(turn)

    poke_batch  = collate_features(poke_batches)
    fflat       = collate_field_features(field_flat)
    field_tensor = fflat.field.view(-1, K, FIELD_DIM)

    move_idx   = torch.tensor(move_idx_list[:collected], dtype=torch.long)
    pp_ratio   = torch.tensor(pp_ratio_list[:collected], dtype=torch.float32)
    mv_dis     = torch.tensor(move_disabled_list[:collected], dtype=torch.float32)
    mech_id    = torch.tensor(mech_id_list[:collected], dtype=torch.long)
    mech_type  = torch.tensor(mech_type_list[:collected], dtype=torch.long)

    B = poke_batch.species_idx.shape[0]
    print(f"Batch size: {B}")

    # Move to device
    poke_batch  = poke_batch.to(device)
    field_tensor = field_tensor.to(device)
    move_idx    = move_idx.to(device)
    pp_ratio    = pp_ratio.to(device)
    mv_dis      = mv_dis.to(device)
    mech_id     = mech_id.to(device)
    mech_type   = mech_type.to(device)

    # ── Forward ────────────────────────────────────────────────────────────────
    pokemon_tokens = agent.poke_emb(poke_batch)      # [B, 48, TOKEN_DIM]
    pre_tokens, post_tokens, value = agent.backbone.encode(pokemon_tokens, field_tensor)
    # pre_tokens, post_tokens: [B, 13, D_MODEL]

    action_embeds = agent.action_enc(
        active_token      = pre_tokens[:, 0, :],
        move_idx          = move_idx,
        pp_ratio          = pp_ratio,
        move_disabled     = mv_dis,
        bench_tokens      = pre_tokens[:, 1:6, :],
        mechanic_id       = mech_id,
        mechanic_type_idx = mech_type,
    )   # [B, 13, D_MODEL]

    # ── 5 matrices ─────────────────────────────────────────────────────────────
    A = pre_tokens       # pre-transformer (current turn)
    B = post_tokens      # post-transformer (= keys for cross-attention)
    C = action_embeds    # action queries

    AA = cos_sim_self(A)
    BB = cos_sim_self(B)
    CC = cos_sim_self(C)
    BC = cos_sim_cross(B, C)
    AC = cos_sim_cross(A, C)

    # ── Print all ──────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("MATRICE 1 : A @ A^T    tokens AVANT transformer (current turn)")
    print("="*65)
    print_stats(AA, "A@A^T")
    print_matrix(AA, TOKEN_LABELS, title="Pre-transformer self-similarity")

    print("\n" + "="*65)
    print("MATRICE 2 : B @ B^T    tokens APRÈS transformer (= keys pour cross-attn)")
    print("="*65)
    print_stats(BB, "B@B^T")
    print_matrix(BB, TOKEN_LABELS, title="Post-transformer self-similarity")

    print("\n" + "="*65)
    print("MATRICE 3 : C @ C^T    action queries")
    print("="*65)
    print_stats(CC, "C@C^T")
    print_matrix(CC, ACTION_LABELS, title="Action query self-similarity")

    print("\n" + "="*65)
    print("MATRICE 4 : B @ C^T    keys vs queries  <- DIAGNOSTIC DIRECT")
    print("="*65)
    print_stats(BC, "B@C^T (flattened)")
    print_matrix(BC, ACTION_LABELS, TOKEN_LABELS,
                 title="Keys (B rows) vs Queries (C cols) — attention proxy")

    print("\n" + "="*65)
    print("MATRICE 5 : A @ C^T    pre-transformer vs queries")
    print("="*65)
    print_stats(AC, "A@C^T (flattened)")
    print_matrix(AC, ACTION_LABELS, TOKEN_LABELS,
                 title="Pre-transformer (A) vs Queries (C)")

    # ── Save heatmaps ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SAVING HEATMAPS")
    print("="*65)
    save_heatmap(AA, TOKEN_LABELS, filename="01_A_AT_pre.png",
                 title="A @ A^T  — Pre-transformer self-similarity")
    save_heatmap(BB, TOKEN_LABELS, filename="02_B_BT_post.png",
                 title="B @ B^T  — Post-transformer self-similarity (keys)")
    save_heatmap(CC, ACTION_LABELS, filename="03_C_CT_queries.png",
                 title="C @ C^T  — Action query self-similarity")
    save_heatmap(BC, ACTION_LABELS, TOKEN_LABELS, filename="04_B_CT_keys_vs_queries.png",
                 title="B @ C^T  — Keys vs Queries (cross-attention proxy)")
    save_heatmap(AC, ACTION_LABELS, TOKEN_LABELS, filename="05_A_CT_pre_vs_queries.png",
                 title="A @ C^T  — Pre-transformer vs Queries")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("SUMMARY")
    print("="*65)
    a_off = off_diag_stats(AA)["mean"]
    b_off = off_diag_stats(BB)["mean"]
    c_off = off_diag_stats(CC)["mean"]

    print(f"  A@A^T  (pre-tokens)        : {a_off:.4f}")
    print(f"  B@B^T  (post-tokens / keys) : {b_off:.4f}  Δ={b_off - a_off:+.4f} vs pre")
    print(f"  C@C^T  (action queries)      : {c_off:.4f}")

    if b_off - a_off > 0.05:
        print(f"  ⚠ Le transformer AUGMENTE la similarité (+{b_off - a_off:.3f}) → léger oversmoothing")
    elif b_off - a_off < -0.05:
        print(f"  ✓ Le transformer DIFFÉRENCIE les tokens ({b_off - a_off:.3f}) → healthy")
    else:
        print(f"  ~ Le transformer ne change presque pas la similarité")

    if c_off > 0.95:
        print(f"  ✗ COLLAPSE QUERIES : C@C^T = {c_off:.4f} > 0.95")
    elif c_off > 0.8:
        print(f"  ⚠ Queries modérément distinctes ({c_off:.4f})")
    else:
        print(f"  ✓ Queries bien distinctes ({c_off:.4f})")

    # B@C^T: check if all queries attend to the same key
    bc_argmax = BC.argmax(dim=1)
    unique_keys = bc_argmax.unique()
    if len(unique_keys) <= 2:
        print(f"  ✗ CROSS-ATTENTION COLLAPSE : toutes les queries max sur {len(unique_keys)} key(s)")
        for qidx in range(BC.shape[0]):
            print(f"      {ACTION_LABELS[qidx]} -> max on {TOKEN_LABELS[BC[qidx].argmax().item()]}")
    else:
        print(f"  ✓ Queries dispersées sur {len(unique_keys)} keys différentes")
        for qidx in range(BC.shape[0]):
            print(f"      {ACTION_LABELS[qidx]} -> max on {TOKEN_LABELS[BC[qidx].argmax().item()]}")

    print("\nHeatmaps saved to diag/")
    print("Done.")


if __name__ == "__main__":
    main()