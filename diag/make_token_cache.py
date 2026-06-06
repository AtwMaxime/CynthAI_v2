"""
make_token_cache.py — Rollout + backbone forward pass → seq_all_cache.pt

Minimal script: no sklearn, no probes. Only collects transitions and runs the
backbone to produce the full post-Transformer token sequence.

Usage:
    python diag/make_token_cache.py \\
        --checkpoint checkpoints/cheater_v7/agent_000300.pt \\
        --n_envs 32 --min_steps 8192 --device cpu

Output: diag/seq_all_cache.pt  (or --output path)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.agent import CynthAIAgent
from training.rollout import collect_rollout
from diag.probing._common import cache_tokens_full, extract_next_hp_labels, extract_labels


def main():
    parser = argparse.ArgumentParser(
        description="Collect rollout and cache backbone token representations."
    )
    parser.add_argument("--checkpoint",      required=True,
                        help="Path to agent checkpoint (.pt)")
    parser.add_argument("--n_envs",          type=int, default=32)
    parser.add_argument("--min_steps",       type=int, default=8192)
    parser.add_argument("--device",          default="cpu")
    parser.add_argument("--output",          default="diag/seq_all_cache.pt",
                        help="Output path for the cache file")
    parser.add_argument("--batch_size",      type=int, default=256)
    parser.add_argument("--critic_n_layers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
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

    # ── Collect rollout ───────────────────────────────────────────────────────
    print(f"\nCollecting rollout  n_envs={args.n_envs}  min_steps={args.min_steps} ...")
    buffer = collect_rollout(
        agent_self=agent,
        agent_opp=agent,
        n_envs=args.n_envs,
        min_steps=args.min_steps,
        gamma=0.99,
        lam=0.95,
        device=device,
    )
    N = len(buffer)
    print(f"  collected {N} transitions")

    # ── Cache tokens + DETR queries + backbone CLS ──────────────────────────
    print("\nCaching seq_all, detr_queries, actions, backbone_cls, backbone_values ...")
    seq_all, detr_queries, actions, backbone_cls, backbone_values = cache_tokens_full(
        agent, buffer, device, batch_size=args.batch_size
    )
    print(f"  seq_all:          {tuple(seq_all.shape)}")           # [N, 52, 256]
    print(f"  detr_queries:     {tuple(detr_queries.shape)}")      # [N, 13, 256]
    print(f"  backbone_cls:     {tuple(backbone_cls.shape)}")      # [N, 256]
    print(f"  backbone_values:  {tuple(backbone_values.shape)}")   # [N, 1]

    # ── Extract current + next-HP labels ─────────────────────────────────────
    import numpy as np
    from model.backbone import K_TURNS

    print("\nExtracting HP labels ...")
    OWN_BASE = (K_TURNS - 1) * 12       # 36
    OPP_BASE = (K_TURNS - 1) * 12 + 6  # 42
    transitions = buffer._transitions
    cur_hp_own = np.array(
        [[tr.scalars[OWN_BASE + j, 1].item() for j in range(6)] for tr in transitions],
        dtype=np.float32,
    )   # [N, 6]
    cur_hp_opp = np.array(
        [[tr.scalars[OPP_BASE + j, 1].item() for j in range(6)] for tr in transitions],
        dtype=np.float32,
    )   # [N, 6]

    next_hp_own, next_hp_opp, next_valid = extract_next_hp_labels(buffer)
    n_valid = int(next_valid.sum())
    print(f"  next_valid: {n_valid}/{N} ({100*n_valid/N:.1f}%)")

    # no_switch_valid : own didn't switch (action < 8) AND has a next step.
    # Bench slot order resets on every switch, so dHP is only meaningful for the
    # active mon (slot 0) when we didn't switch ourselves.
    # For the opponent we additionally exclude turns where opp likely switched:
    # if next_hp_opp[0] > cur_hp_opp[0] + 0.05, a new (healthier) mon came in.
    actions_np = actions.numpy()
    no_switch_valid = next_valid & (actions_np < 8)
    opp_no_switch_valid = no_switch_valid & (
        next_hp_opp[:, 0] <= cur_hp_opp[:, 0] + 0.05
    )
    n_ns  = int(no_switch_valid.sum())
    n_ons = int(opp_no_switch_valid.sum())
    print(f"  no_switch_valid (own):     {n_ns}/{N} ({100*n_ns/N:.1f}%)")
    print(f"  opp_no_switch_valid:       {n_ons}/{N} ({100*n_ons/N:.1f}%)")

    # ── Extract all labels ────────────────────────────────────────────────────
    print("\nExtracting labels ...")
    labels = extract_labels(buffer)
    y_win = labels["y_win"]
    n_win_known = int((y_win >= 0).sum())
    win_rate = float(y_win[y_win >= 0].mean()) if n_win_known > 0 else float("nan")
    print(f"  win labels known: {n_win_known}/{N} ({100*n_win_known/N:.1f}%)  win_rate={win_rate:.3f}")

    # ── Cache critic representations ──────────────────────────────────────────
    critic_cls = None
    critic_seq = None
    critic_values = None
    if hasattr(agent, 'independent_critic'):
        print("\nCaching critic representations ...")
        all_critic_cls = []
        all_critic_seq = []
        all_critic_val = []
        with torch.no_grad():
            for start in range(0, N, args.batch_size):
                end   = min(start + args.batch_size, N)
                batch = buffer._gather(list(range(start, end)), device)
                pt = agent.poke_emb(batch["poke_batch"])
                ft = batch["field_tensor"]
                v, _, cls_out, seq_52 = agent.independent_critic(
                    pt, ft, return_repr=True,
                )
                all_critic_cls.append(cls_out.cpu())
                all_critic_seq.append(seq_52.cpu())
                all_critic_val.append(v.cpu())
        critic_cls    = torch.cat(all_critic_cls, dim=0)   # [N, D_MODEL]
        critic_seq    = torch.cat(all_critic_seq, dim=0)   # [N, 52, D_MODEL]
        critic_values = torch.cat(all_critic_val, dim=0)   # [N, 1]
        print(f"  critic_cls:    {tuple(critic_cls.shape)}")
        print(f"  critic_seq:    {tuple(critic_seq.shape)}")
        print(f"  critic_values: {tuple(critic_values.shape)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "seq_all":          seq_all,
        "detr_queries":     detr_queries,
        "actions":          actions,
        "backbone_cls":     backbone_cls,
        "backbone_values":  backbone_values,
        "cur_hp_own":    cur_hp_own,
        "cur_hp_opp":    cur_hp_opp,
        "next_hp_own":   next_hp_own,
        "next_hp_opp":   next_hp_opp,
        "next_valid":          next_valid,
        "no_switch_valid":     no_switch_valid,
        "opp_no_switch_valid": opp_no_switch_valid,
        "y_win":               y_win,
        "y_return":            labels["y_return"],
        "y_type1":             labels["y_type1"],
        "y_item":              labels["y_item"],
        "y_ability":           labels["y_ability"],
        "y_hp":                labels["y_hp"],
        "y_stats":             labels["y_stats"],
        "n_transitions":       N,
    }
    if critic_cls is not None:
        save_dict["critic_cls"]    = critic_cls
        save_dict["critic_seq"]    = critic_seq
        save_dict["critic_values"] = critic_values
    torch.save(save_dict, out_path)
    print(f"Token cache saved: {out_path}")


if __name__ == "__main__":
    main()
