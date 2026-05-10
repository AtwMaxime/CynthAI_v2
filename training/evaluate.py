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

from model.agent import CynthAIAgent
from training.rollout import collect_rollout, RandomPolicy
from env.bots import FullOffensePolicy


def run_eval(
    agent:             torch.nn.Module,
    opponent:          torch.nn.Module | None = None,
    n_games:           int   = 500,
    n_envs:            int   = 16,
    format_id:         str   = "gen9randombattle",
    device:            torch.device = torch.device("cpu"),
    opponent_sampler:  callable | None = None,
    mask_ratio:        float = 0.0,   # P1: match training masking
) -> dict:
    """
    Evaluate an agent against an opponent policy.

    agent and opponent are callable policy objects (nn.Module, RandomPolicy,
    FullOffensePolicy, etc.). Runs n_games total using parallel environments.

    If opponent_sampler is provided, it is called each batch to get a fresh
    opponent (e.g. sampling from an OpponentPool for diversity). When using
    opponent_sampler, the `opponent` parameter is ignored.

    Returns dict with keys: win_rate, wins, losses, ties, total, ci_low, ci_high.
    """
    wins   = 0
    losses = 0
    ties   = 0
    total  = 0

    # Diagnostic collection
    battle_lengths: list[int] = []
    action_histogram = [0] * 13
    reward_decomp_sum: dict[str, float] = {"ko_own": 0.0, "ko_opp": 0.0, "hp_adv": 0.0, "terminal": 0.0}
    reward_decomp_n = 0
    value_preds: list[float] = []
    value_returns: list[float] = []

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

    return {
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
