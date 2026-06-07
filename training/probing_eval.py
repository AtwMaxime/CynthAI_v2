"""
probing_eval — Run light probing analyses during training evaluation.

Orchestrates: rollout -> cache -> run_light() on each probe module -> scalars.
Called from self_play.py every probe_freq evals.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from model.backbone import K_TURNS
from training.rollout import collect_rollout
from diag.probing._common import (
    cache_tokens_full, extract_labels, extract_next_hp_labels,
    numpy_from_cache, check_labels,
)
from diag.probing import svd_probes, actor_probes, critic_probes, detr_probes


def run_probing_eval(
    agent,
    device: torch.device,
    cfg,
    update: int,
    probe_min_steps: int = 2048,
) -> dict[str, float]:
    """
    Run light probing analyses on a fresh rollout.

    Returns dict of scalar metrics prefixed with 'probe/'.
    """
    t0 = time.perf_counter()
    print(f"\n  [PROBE] Running probing eval (min_steps={probe_min_steps}) ...")

    agent.eval()

    # 1. Collect rollout (self-play)
    buffer = collect_rollout(
        agent_self=agent,
        agent_opp=agent,
        n_envs=min(cfg.n_envs, 32),
        min_steps=probe_min_steps,
        gamma=cfg.gamma,
        lam=cfg.lam,
        device=device,
    )
    N = len(buffer)
    print(f"  [PROBE] collected {N} transitions")

    # 2. Cache backbone tokens + DETR queries
    seq_all, detr_queries, actions, backbone_cls, backbone_val = cache_tokens_full(
        agent, buffer, device, batch_size=256
    )

    # 3. Extract labels
    labels = extract_labels(buffer)

    # 4. Extract next-HP labels
    next_hp_own, next_hp_opp, next_valid = extract_next_hp_labels(buffer)

    # 5. Current HP + validity masks
    OWN_BASE = (K_TURNS - 1) * 12
    OPP_BASE = (K_TURNS - 1) * 12 + 6
    transitions = buffer._transitions
    cur_hp_own = np.array(
        [[tr.scalars[OWN_BASE + j, 1].item() for j in range(6)] for tr in transitions],
        dtype=np.float32,
    )
    cur_hp_opp = np.array(
        [[tr.scalars[OPP_BASE + j, 1].item() for j in range(6)] for tr in transitions],
        dtype=np.float32,
    )

    actions_np = actions.numpy()
    no_switch_valid = next_valid & (actions_np < 8)
    opp_no_switch_valid = no_switch_valid & (
        next_hp_opp[:, 0] <= cur_hp_opp[:, 0] + 0.05
    )

    # 6. Critic representations (when value_head has its own Transformer)
    critic_cls = None
    critic_seq = None
    critic_values = None
    if agent.value_head.n_layers > 0:
        all_critic_cls, all_critic_seq, all_critic_val = [], [], []
        with torch.no_grad():
            for start in range(0, N, 256):
                end = min(start + 256, N)
                batch = buffer._gather(list(range(start, end)), device)
                pt = agent.poke_emb(batch["poke_batch"])
                ft = batch["field_tensor"]
                _, _, full_seq_b, padding_mask_b, _ = agent.backbone.encode(pt, ft)
                # Run value head transformer manually to get CLS + seq
                vh = agent.value_head
                B_b = full_seq_b.shape[0]
                dev = full_seq_b.device
                cls_tok = vh.cls_token.expand(B_b, 1, full_seq_b.shape[-1])
                seq_in = torch.cat([cls_tok, full_seq_b.detach()], dim=1)
                mask_in = torch.cat([
                    torch.zeros(B_b, 1, dtype=torch.bool, device=dev),
                    padding_mask_b,
                ], dim=1)
                seq_out = vh.transformer(seq_in, src_key_padding_mask=mask_in)
                cls_out = seq_out[:, 0, :]
                seq_52 = seq_out[:, 1:, :]
                v = vh.value_head(cls_out)
                all_critic_cls.append(cls_out.cpu())
                all_critic_seq.append(seq_52.cpu())
                all_critic_val.append(v.cpu())
        critic_cls = torch.cat(all_critic_cls, dim=0)
        critic_seq = torch.cat(all_critic_seq, dim=0)
        critic_values = torch.cat(all_critic_val, dim=0)

    # 7. Assemble cache dict
    cache: dict = {
        "seq_all": seq_all,
        "detr_queries": detr_queries,
        "backbone_cls": backbone_cls,
        "backbone_values": backbone_val,
        "actions": actions_np,
        "cur_hp_own": cur_hp_own,
        "cur_hp_opp": cur_hp_opp,
        "next_hp_own": next_hp_own,
        "next_hp_opp": next_hp_opp,
        "next_valid": next_valid,
        "no_switch_valid": no_switch_valid,
        "opp_no_switch_valid": opp_no_switch_valid,
        "n_transitions": N,
    }
    if check_labels(labels):
        cache.update(labels)
    if critic_cls is not None:
        cache["critic_cls"] = critic_cls
        cache["critic_seq"] = critic_seq
        cache["critic_values"] = critic_values

    # 8. Train/val split
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_val = max(1, int(N * 0.2))
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())

    # 9. Output directory
    out_dir = Path(cfg.checkpoint_dir) / "probes" / f"update{update:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 10. Run each module's run_light()
    all_results: dict = {}
    modules = [
        ("svd", svd_probes),
        ("actor", actor_probes),
        ("critic", critic_probes),
        ("detr", detr_probes),
    ]
    for name, module in modules:
        if module.can_run(cache):
            try:
                result = module.run_light(cache, train_idx, val_idx, out_dir)
                all_results[name] = result
            except Exception as e:
                print(f"  [PROBE] WARNING: {name} failed: {e}")

    # 11. Extract scalar metrics
    metrics: dict[str, float] = {}

    # SVD
    svd = all_results.get("svd", {})
    pca = svd.get("pca", {})
    if "token_emb" in pca:
        metrics["probe/svd_pca_token_n90"] = pca["token_emb"].get("n90", 0)
        metrics["probe/svd_pca_token_n95"] = pca["token_emb"].get("n95", 0)
    svd_cur = svd.get("svd_current_turn", {})
    svd_all = svd.get("svd_all_turns", {})
    if svd_cur:
        metrics["probe/svd_eff_rank_cur"] = svd_cur.get("effective_rank_mean", 0)
        metrics["probe/svd_top1_energy_cur"] = svd_cur.get("top1_mean", 0)
    if svd_all:
        metrics["probe/svd_eff_rank_all"] = svd_all.get("effective_rank_mean", 0)

    # Actor
    actor = all_results.get("actor", {})
    mp = actor.get("mean_pool_current", {})
    if mp:
        metrics["probe/actor_mean_return_r2"] = mp.get("return_r2", 0)
        metrics["probe/actor_mean_win_auc"] = mp.get("win_auc", 0)
        metrics["probe/actor_mean_hp_r2"] = mp.get("hp_r2", 0)
        metrics["probe/actor_mean_type1_acc"] = mp.get("type1_acc", 0)
        metrics["probe/actor_mean_item_acc"] = mp.get("item_acc", 0)
    cls = actor.get("backbone_cls", {})
    if cls:
        metrics["probe/actor_cls_return_r2"] = cls.get("return_r2", 0)
        metrics["probe/actor_cls_win_auc"] = cls.get("win_auc", 0)
        metrics["probe/actor_cls_eff_rank"] = cls.get("effective_rank", 0)

    # Critic
    crit = all_results.get("critic", {})
    crit_cls = crit.get("probes", {}).get("cls", {})
    if crit_cls:
        metrics["probe/critic_cls_return_r2"] = crit_cls.get("return_r2", 0)
        metrics["probe/critic_cls_win_auc"] = crit_cls.get("win_auc", 0)
    crit_er = crit.get("effective_rank", {})
    if crit_er:
        metrics["probe/critic_eff_rank"] = crit_er.get("cls", 0)

    # DETR
    detr = all_results.get("detr", {})
    pa = detr.get("probe_action", {})
    if pa:
        metrics["probe/detr_action_acc"] = pa.get("mean_pool_top1_acc", 0)
    pw = detr.get("probe_win", {})
    if pw:
        metrics["probe/detr_win_auc"] = pw.get("mean_pool_auc", 0)
    pdh = detr.get("probe_delta_hp", {})
    if pdh:
        metrics["probe/detr_dhp_own_r2"] = pdh.get("mean_pool_own_r2", 0)
        metrics["probe/detr_dhp_opp_r2"] = pdh.get("mean_pool_opp_r2", 0)
    pko = detr.get("probe_ko", {})
    if pko:
        metrics["probe/detr_ko_own_auc"] = pko.get("mean_pool_own_auc", 0)
        metrics["probe/detr_ko_opp_auc"] = pko.get("mean_pool_opp_auc", 0)

    elapsed = time.perf_counter() - t0
    print(f"  [PROBE] done [{elapsed:.1f}s] — {len(metrics)} scalars")

    return metrics
