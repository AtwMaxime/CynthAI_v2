"""
CynthAI_v2 Evaluation — run a checkpoint against an opponent and report win rate.

Usage:
    python -m training.evaluate --checkpoint checkpoints/agent_000600.pt

Options:
    --checkpoint   path to .pt checkpoint (default: checkpoints/agent_best.pt)
    --n-battles    number of battles to play (default: 200)
    --n-envs       parallel environments (default: 16)
    --opponent     "random" or path to opponent checkpoint (default: random)
    --device       cpu / cuda (default: cpu)
    --format       battle format (default: gen9randombattle)

Outputs:
    win_rate, wins, losses, ties, total, ci_low, ci_high (Wilson 95% CI)
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model.agent import CynthAIAgent
from training.rollout import collect_rollout, RandomPolicy, encode_state, build_action_mask
from env.bots import FullOffensePolicy
from model.embeddings import collate_features, collate_field_features, FieldBatch, FIELD_DIM


TOKEN_LABELS  = ["O0","O1","O2","O3","O4","O5",
                 "P0","P1","P2","P3","P4","P5","FL"]
ACTION_LABELS = ["M1","M2","M3","M4",
                 "T1","T2","T3","T4",
                 "S0","S1","S2","S3","S4"]


# ── Cosine similarity diagnostics ─────────────────────────────────────────────

@torch.no_grad()
def compute_cos_sim_metrics(
    agent:   torch.nn.Module,
    device:  torch.device,
    n_battles: int = 64,
) -> dict:
    """
    Compute the 5 cosine similarity matrices from a batch of eval states.

    Returns dict with:
      matrices: { "A_AT": [13,13], "B_BT": [13,13], "C_CT": [13,13],
                  "B_CT": [13,13], "A_CT": [13,13] }
      scalars:  { "cos_pre", "cos_post", "cos_query",
                  "cos_keys_queries_mean", "cos_pre_queries_mean",
                  "cos_n_unique_keys" }
    """
    from simulator import PyBattle

    # ── Collect states ─────────────────────────────────────────────────────
    poke_feats, field_feats = [], []
    move_idx, pp_ratio, mv_dis = [], [], []
    mech_id, mech_type = [], []

    n_envs = min(n_battles, 64)
    seeds = [hash(f"cos_eval_{i}") % (2**31) for i in range(n_envs)]
    envs = [PyBattle("gen9randombattle", seed=s) for s in seeds]
    finished = [False] * n_envs
    collected, steps = 0, 0

    while collected < n_battles and steps < 500:
        steps += 1
        for i in range(n_envs):
            if finished[i]:
                continue
            try:
                state = envs[i].get_state()
            except Exception:
                finished[i] = True; continue
            pf, ff = encode_state(state, 0)
            poke_feats.append(pf)
            field_feats.append(ff)
            act = pf[0]
            move_idx.append(act.move_indices[:4])
            pp_ratio.append(act.move_pp[:4])
            mv_dis.append(act.move_disabled[:4])
            mech_id.append(0)
            mech_type.append(0)
            collected += 1
            if collected >= n_battles:
                break
            try:
                req = state["sides"][0].get("request_state", "")
                if req in ("Move", "Switch"):
                    envs[i].step(torch.randint(0, 13, (1,)).item())
                else:
                    finished[i] = True
            except Exception:
                finished[i] = True

    # ── Build batch ────────────────────────────────────────────────────────
    K = 4
    poke_batches, field_flat = [], []
    for pf, ff in zip(poke_feats, field_feats):
        turn = []
        for _ in range(K):
            turn.extend(pf)
            field_flat.append(ff)
        poke_batches.append(turn)

    poke_batch = collate_features(poke_batches).to(device)
    fflat      = collate_field_features(field_flat)
    field_tensor = fflat.field.view(-1, K, FIELD_DIM).to(device)

    move_idx   = torch.tensor(move_idx, dtype=torch.long).to(device)
    pp_ratio   = torch.tensor(pp_ratio, dtype=torch.float32).to(device)
    mv_dis     = torch.tensor(mv_dis, dtype=torch.float32).to(device)
    mech_id    = torch.tensor(mech_id, dtype=torch.long).to(device)
    mech_type  = torch.tensor(mech_type, dtype=torch.long).to(device)

    # ── Forward ────────────────────────────────────────────────────────────
    pokemon_tokens = agent.poke_emb(poke_batch)
    pre_tokens, post_tokens, _ = agent.backbone.encode(pokemon_tokens, field_tensor)

    action_embeds = agent.action_enc(
        active_token      = pre_tokens[:, 0, :],
        move_idx          = move_idx,
        pp_ratio          = pp_ratio,
        move_disabled     = mv_dis,
        bench_tokens      = pre_tokens[:, 1:6, :],
        mechanic_id       = mech_id,
        mechanic_type_idx = mech_type,
    )

    # ── Compute 5 matrices ─────────────────────────────────────────────────
    def _cos_self(x):
        xn = F.normalize(x, dim=-1)
        return (xn @ xn.transpose(-1, -2)).mean(dim=0).cpu()

    def _cos_cross(x, y):
        xn = F.normalize(x, dim=-1)
        yn = F.normalize(y, dim=-1)
        return (xn @ yn.transpose(-1, -2)).mean(dim=0).cpu()

    AA = _cos_self(pre_tokens)
    BB = _cos_self(post_tokens)
    CC = _cos_self(action_embeds)
    BC = _cos_cross(post_tokens, action_embeds)   # keys vs queries
    AC = _cos_cross(pre_tokens, action_embeds)     # pre vs queries

    # ── Scalar summaries ───────────────────────────────────────────────────
    def _off_diag_mean(m):
        N = m.shape[0]
        off = m.clone().fill_diagonal_(float("nan"))
        return off[~torch.isnan(off)].mean().item()

    bc_argmax = BC.argmax(dim=1)
    n_unique = bc_argmax.unique().numel()

    scalars = {
        "cos_pre_offdiag":        _off_diag_mean(AA),
        "cos_post_offdiag":       _off_diag_mean(BB),
        "cos_query_offdiag":      _off_diag_mean(CC),
        "cos_keys_queries_mean":  BC.mean().item(),
        "cos_pre_queries_mean":   AC.mean().item(),
        "cos_n_unique_keys":      n_unique,
    }

    return {
        "matrices": {"A_AT": AA, "B_BT": BB, "C_CT": CC, "B_CT": BC, "A_CT": AC},
        "scalars":  scalars,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def run_eval(
    agent:             torch.nn.Module,
    opponent:          torch.nn.Module | None = None,
    n_games:           int   = 500,
    n_envs:            int   = 16,
    format_id:         str   = "gen9randombattle",
    device:            torch.device = torch.device("cpu"),
    opponent_sampler:  callable | None = None,
    mask_ratio:        float = 0.0,   # P1: match training masking
    capture_cross_attn: bool = False, # P13e: capture cross-attention weights
    compute_cos_sim:   bool = False,  # P22: cosine similarity diagnostics
) -> dict:
    """
    Evaluate an agent against an opponent policy.

    agent and opponent are callable policy objects (nn.Module, RandomPolicy,
    FullOffensePolicy, etc.). Runs n_games total using parallel environments.

    If opponent_sampler is provided, it is called each batch to get a fresh
    opponent (e.g. sampling from an OpponentPool for diversity). When using
    opponent_sampler, the `opponent` parameter is ignored.

    When capture_cross_attn=True, stores cross-attention weights from the
    actor head and returns cross_attn_stats in the result dict.

    When compute_cos_sim=True, computes 5 cosine similarity matrices
    (A@A^T, B@B^T, C@C^T, B@C^T, A@C^T) from a sample of eval states
    and returns cos_sim_scalars + cos_sim_matrices in the result dict.

    Returns dict with keys: win_rate, wins, losses, ties, total, ci_low, ci_high.
    """
    wins   = 0
    losses = 0
    ties   = 0
    total  = 0

    # Diagnostic collection
    battle_lengths: list[int] = []
    action_histogram = [0] * 13
    reward_decomp_sum: dict[str, float] = {"ko_own": 0.0, "ko_opp": 0.0, "hp_adv": 0.0, "count_adv": 0.0, "status": 0.0, "hazard": 0.0, "hazard_remove": 0.0, "terminal": 0.0}
    reward_decomp_n = 0
    value_preds: list[float] = []
    value_returns: list[float] = []

    # Cross-attention aggregation
    cross_attn_sums: torch.Tensor | None = None
    cross_attn_n = 0

    if capture_cross_attn:
        agent.backbone._store_cross_attn = True

    t0 = time.perf_counter()

    while total < n_games:
        remaining = n_games - total
        envs      = min(n_envs, remaining)
        min_steps = envs * 50  # enough for multiple battles to finish

        # Sample a fresh opponent each batch when using a sampler
        current_opp = opponent_sampler() if opponent_sampler is not None else opponent

        buffer = collect_rollout(
            agent_self = agent,
            agent_opp  = current_opp,
            n_envs     = envs,
            min_steps  = min_steps,
            format_id  = format_id,
            gamma      = 0.99,
            lam        = 0.95,
            device     = device,
            mask_ratio = mask_ratio,
        )

        # --- Collect cross-attention stats ---
        if capture_cross_attn:
            stats = agent.backbone.get_cross_attention_stats()
            if stats is not None:
                if cross_attn_sums is None:
                    cross_attn_sums = stats["mean"] * stats["n"]  # [H, 13, 13]
                else:
                    cross_attn_sums += stats["mean"] * stats["n"]
                cross_attn_n += stats["n"]

        # --- Collect diagnostics from this batch ---
        ep_len = 0
        for t_idx, t in enumerate(buffer._transitions):
            # Action histogram
            if 0 <= t.action < 13:
                action_histogram[t.action] += 1

            # Value calibration
            if t_idx < len(buffer._returns):
                value_preds.append(t.value_old)
                value_returns.append(buffer._returns[t_idx])

            # Reward decomposition
            if t.reward_components:
                for k in reward_decomp_sum:
                    reward_decomp_sum[k] += t.reward_components.get(k, 0.0)
                reward_decomp_n += 1

            # Battle length and win/loss tracking
            ep_len += 1
            if t.done:
                total += 1
                battle_lengths.append(ep_len)
                ep_len = 0
                if t.reward > 0.0:
                    wins += 1
                elif t.reward < 0.0:
                    losses += 1
                else:
                    ties += 1

    # Clean up flag
    if capture_cross_attn:
        agent.backbone._store_cross_attn = False

    elapsed  = time.perf_counter() - t0
    win_rate = wins / total if total > 0 else 0.0

    # Wilson 95% confidence interval
    z      = 1.96
    n      = total
    p      = win_rate
    denom  = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    ci_low  = max(0.0, centre - margin)
    ci_high = min(1.0, centre + margin)

    reward_decomp_avg = {}
    if reward_decomp_n > 0:
        reward_decomp_avg = {k: v / reward_decomp_n for k, v in reward_decomp_sum.items()}

    result = {
        "win_rate": win_rate,
        "wins":     wins,
        "losses":   losses,
        "ties":     ties,
        "total":    total,
        "ci_low":   ci_low,
        "ci_high":  ci_high,

        # Diagnostics
        "battle_lengths":    battle_lengths,
        "action_histogram":  action_histogram,
        "reward_decomp_avg": reward_decomp_avg,
        "value_preds":       value_preds,
        "value_returns":     value_returns,
    }

    if capture_cross_attn and cross_attn_sums is not None and cross_attn_n > 0:
        result["cross_attn_stats"] = {
            "mean": cross_attn_sums / cross_attn_n,
            "n": cross_attn_n,
        }

    if compute_cos_sim:
        try:
            cos_data = compute_cos_sim_metrics(agent, device, n_battles=64)
            result["cos_sim_scalars"] = cos_data["scalars"]
            result["cos_sim_matrices"] = cos_data["matrices"]
        except Exception as e:
            print(f"  WARNING: cos_sim failed: {e}")

    return result


def evaluate(
    checkpoint: str,
    n_envs:     int   = 16,
    n_battles:  int   = 200,
    format_id:  str   = "gen9randombattle",
    opponent:   str   = "random",
    device:     str   = "cpu",
) -> dict:
    """Load a checkpoint and evaluate against an opponent (CLI entry point)."""
    dev = torch.device(device)

    agent = CynthAIAgent().to(dev)
    ckpt  = torch.load(checkpoint, map_location=dev, weights_only=True)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    if opponent == "random":
        opp_policy = RandomPolicy()
    else:
        opp_policy = CynthAIAgent().to(dev)
        opp_ckpt   = torch.load(opponent, map_location=dev, weights_only=True)
        opp_policy.load_state_dict(opp_ckpt["model"])
        opp_policy.eval()
        for p in opp_policy.parameters():
            p.requires_grad_(False)

    print(f"Evaluating {checkpoint}  vs  {opponent}  ...")
    t0 = time.perf_counter()

    result = run_eval(
        agent    = agent,
        opponent = opp_policy,
        n_games  = n_battles,
        n_envs   = n_envs,
        format_id= format_id,
        device   = dev,
    )

    print(
        f"  win_rate={result['win_rate']*100:.1f}%  "
        f"(95% CI: [{result['ci_low']*100:.1f}%, {result['ci_high']*100:.1f}%])  "
        f"W={result['wins']}  L={result['losses']}  T={result['ties']}  "
        f"[{time.perf_counter() - t0:.1f}s]"
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/agent_best.pt")
    parser.add_argument("--n-battles",  type=int, default=200)
    parser.add_argument("--n-envs",     type=int, default=16)
    parser.add_argument("--opponent",   default="random")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--format",     default="gen9randombattle")
    args = parser.parse_args()

    evaluate(
        checkpoint = args.checkpoint,
        n_envs     = args.n_envs,
        n_battles  = args.n_battles,
        format_id  = args.format,
        opponent   = args.opponent,
        device     = args.device,
    )
