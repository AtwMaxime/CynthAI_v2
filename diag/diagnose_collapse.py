"""
Diagnose representation collapse — measure effective rank at each pipeline stage.

Pipeline stages:
  1. PokemonEmbeddings output     [B, N, 439]  — pre-projection token
  2. pokemon_proj(tokens)         [B, N, 256]  — after linear projection
  3. + positional embeddings      [B, N, 256]  — after adding slot/temporal emb
  4. After each Transformer layer [B, N, 256]  — layer 0, 1, 2
  5. Final output (current turn)  [B, 13, 256] — post-backbone

For each stage, compute:
  - Per-token SVD across batch: effective rank of the token space
  - Per-state SVD: how diverse are the N tokens within a single state?
  - Variance explained by top-k components

Usage:
    python diag/diagnose_collapse.py [--checkpoint path] [--n-states 500]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import torch
import numpy as np
from collections import OrderedDict

from model.embeddings import PokemonEmbeddings, PokemonBatch, TOKEN_DIM
from model.backbone import BattleBackbone, D_MODEL, K_TURNS, N_SLOTS
from env.state_encoder import FIELD_DIM


def effective_rank(S: np.ndarray) -> float:
    """Effective rank from singular values via Shannon entropy."""
    p = S / S.sum()
    p = p[p > 1e-10]
    return float(np.exp(-np.sum(p * np.log(p))))


def variance_explained(S: np.ndarray, k: int) -> float:
    """Fraction of variance explained by top-k singular values."""
    total = (S ** 2).sum()
    if total < 1e-12:
        return 1.0
    return float((S[:k] ** 2).sum() / total)


def components_for_threshold(S: np.ndarray, threshold: float) -> int:
    """Number of components needed to reach threshold of variance."""
    total = (S ** 2).sum()
    if total < 1e-12:
        return 1
    cumsum = np.cumsum(S ** 2) / total
    return int(np.searchsorted(cumsum, threshold) + 1)


def analyze_stage(name: str, tensor: torch.Tensor, results: dict):
    """Analyze a [B, N, D] tensor at a pipeline stage."""
    B, N, D = tensor.shape
    mat = tensor.detach().float().cpu()

    # 1. Per-token SVD: flatten [B, N, D] → [B*N, D], compute SVD
    flat = mat.reshape(-1, D).numpy()
    flat_centered = flat - flat.mean(axis=0)
    S_token = np.linalg.svd(flat_centered, compute_uv=False)

    # 2. Per-state SVD: for each batch, SVD on [N, D] tokens
    eff_ranks = []
    top1_energies = []
    for b in range(B):
        state = mat[b].numpy()
        state_centered = state - state.mean(axis=0)
        S_state = np.linalg.svd(state_centered, compute_uv=False)
        eff_ranks.append(effective_rank(S_state))
        total_e = (S_state ** 2).sum()
        if total_e > 1e-12:
            top1_energies.append(float(S_state[0] ** 2 / total_e))
        else:
            top1_energies.append(1.0)

    results[name] = {
        "shape": [B, N, D],
        "token_eff_rank": round(effective_rank(S_token), 2),
        "token_var_top1": round(variance_explained(S_token, 1), 4),
        "token_var_top5": round(variance_explained(S_token, 5), 4),
        "token_var_top10": round(variance_explained(S_token, 10), 4),
        "token_n_for_90pct": components_for_threshold(S_token, 0.90),
        "token_n_for_95pct": components_for_threshold(S_token, 0.95),
        "token_n_for_99pct": components_for_threshold(S_token, 0.99),
        "state_eff_rank_mean": round(float(np.mean(eff_ranks)), 2),
        "state_eff_rank_std": round(float(np.std(eff_ranks)), 2),
        "state_top1_energy_mean": round(float(np.mean(top1_energies)), 4),
    }

    r = results[name]
    print(f"\n  [{name}]  shape={B}x{N}x{D}")
    print(f"    Token-level:  eff_rank={r['token_eff_rank']:<6}  "
          f"var_top1={r['token_var_top1']:.3f}  top5={r['token_var_top5']:.3f}  "
          f"top10={r['token_var_top10']:.3f}  "
          f"n_90%={r['token_n_for_90pct']}  n_95%={r['token_n_for_95pct']}  n_99%={r['token_n_for_99pct']}")
    print(f"    State-level:  eff_rank={r['state_eff_rank_mean']:.2f}±{r['state_eff_rank_std']:.2f}  "
          f"top1_energy={r['state_top1_energy_mean']:.3f}")


def generate_random_batch(B: int) -> tuple:
    """Generate random batch for analysis (or load from cache)."""
    pb = PokemonBatch(
        species_idx=torch.randint(0, 100, (B, K_TURNS * 12)),
        type1_idx=torch.randint(0, 19, (B, K_TURNS * 12)),
        type2_idx=torch.randint(0, 19, (B, K_TURNS * 12)),
        tera_idx=torch.randint(0, 19, (B, K_TURNS * 12)),
        item_idx=torch.randint(0, 50, (B, K_TURNS * 12)),
        ability_idx=torch.randint(0, 50, (B, K_TURNS * 12)),
        move_idx=torch.randint(0, 100, (B, K_TURNS * 12, 4)),
        scalars=torch.randn(B, K_TURNS * 12, 223),
    )
    field_t = torch.randn(B, K_TURNS, FIELD_DIM)
    return pb, field_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--n-states", type=int, default=256,
                        help="Number of random states to analyze")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    B = args.n_states

    # Build model
    poke_emb = PokemonEmbeddings()
    backbone = BattleBackbone()

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("agent_state_dict", ckpt)
        # Extract sub-dicts
        poke_keys = {k.replace("poke_emb.", ""): v for k, v in state_dict.items() if k.startswith("poke_emb.")}
        backbone_keys = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        if poke_keys:
            poke_emb.load_state_dict(poke_keys, strict=False)
        if backbone_keys:
            backbone.load_state_dict(backbone_keys, strict=False)
        print("  Loaded weights.")

    poke_emb.eval()
    backbone.eval()

    pb, field_t = generate_random_batch(B)

    results = OrderedDict()
    hooks = []
    intermediates = {}

    print(f"\n{'='*60}")
    print(f"Collapse diagnosis — {B} states")
    print(f"{'='*60}")

    with torch.no_grad():
        # Stage 1: PokemonEmbeddings output [B, N, 439]
        poke_tokens = poke_emb(pb)
        analyze_stage("1_poke_emb_output", poke_tokens, results)

        # Stage 2: pokemon_proj [B, N, 256] (no positional yet)
        N = poke_tokens.shape[1]  # K*12 = 48
        pokemon_part = poke_tokens[:, :, :]
        # Project pokemon tokens
        proj_pokemon = backbone.pokemon_proj(pokemon_part[:, :K_TURNS*12, :])
        # Project field tokens
        proj_field = backbone.field_proj(field_t)
        # Interleave into sequence
        seq = torch.zeros(B, K_TURNS * N_SLOTS, D_MODEL)
        for t in range(K_TURNS):
            seq[:, t*N_SLOTS:t*N_SLOTS+12, :] = proj_pokemon[:, t*12:(t+1)*12, :]
            seq[:, t*N_SLOTS+12, :] = proj_field[:, t, :]
        analyze_stage("2_after_proj", seq, results)

        # Stage 3: + positional embeddings
        temporal_ids = torch.arange(K_TURNS).unsqueeze(1).expand(-1, N_SLOTS).reshape(-1)
        slot_ids = torch.arange(N_SLOTS).unsqueeze(0).expand(K_TURNS, -1).reshape(-1)
        seq_pos = seq + backbone.temporal_emb(temporal_ids) + backbone.slot_emb(slot_ids)
        analyze_stage("3_after_pos_emb", seq_pos, results)

        # Stage 4: After each Transformer layer
        x = seq_pos.clone()
        # Prepend CLS token
        cls = backbone.cls_token.expand(B, -1, -1)
        x_with_cls = torch.cat([cls, x], dim=1)  # [B, 1+52, 256]

        for i, layer in enumerate(backbone.transformer.layers):
            x_with_cls = layer(x_with_cls)
            # Analyze without CLS (tokens only)
            analyze_stage(f"4_after_layer_{i}", x_with_cls[:, 1:, :], results)

        # Stage 5: Current turn only (last 13 tokens)
        current = x_with_cls[:, 1:, :][:, -N_SLOTS:, :]  # [B, 13, 256]
        analyze_stage("5_current_turn", current, results)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY — Where does collapse happen?")
    print(f"{'='*60}")
    print(f"{'Stage':<25} {'Shape':>12} {'TokenEffRank':>13} {'Top1Var':>8} {'n_95%':>6} {'StateEffRank':>13}")
    print("-" * 80)
    for name, r in results.items():
        s = r['shape']
        print(f"{name:<25} {s[1]:>4}x{s[2]:<5} {r['token_eff_rank']:>13.1f} "
              f"{r['token_var_top1']:>8.3f} {r['token_n_for_95pct']:>6} "
              f"{r['state_eff_rank_mean']:>8.2f}±{r['state_eff_rank_std']:.2f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
