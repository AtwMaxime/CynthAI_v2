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
from training.rollout import collect_rollout


def evaluate(
    checkpoint: str,
    n_envs:     int   = 16,
    n_battles:  int   = 200,
    format_id:  str   = "gen9randombattle",
    opponent:   str   = "random",
    device:     str   = "cpu",
) -> dict:
    """
    Evaluate a checkpoint against an opponent.

    Returns a dict with keys:
        win_rate, wins, losses, ties, total, ci_low, ci_high
    """
    dev = torch.device(device)

    agent = CynthAIAgent().to(dev)
    ckpt  = torch.load(checkpoint, map_location=dev, weights_only=True)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    if opponent == "random":
        agent_opp = None
    else:
        agent_opp = CynthAIAgent().to(dev)
        opp_ckpt  = torch.load(opponent, map_location=dev, weights_only=True)
        agent_opp.load_state_dict(opp_ckpt["model"])
        agent_opp.eval()
        for p in agent_opp.parameters():
            p.requires_grad_(False)

    wins   = 0
    losses = 0
    ties   = 0
    total  = 0

    print(f"Evaluating {checkpoint}  vs  {opponent}  ...")
    t0 = time.perf_counter()

    while total < n_battles:
        remaining = n_battles - total
        envs      = min(n_envs, remaining)

        buffer = collect_rollout(
            agent_self = agent,
            agent_opp  = agent_opp,
            n_envs     = envs,
            min_steps  = 1,
            format_id  = format_id,
            gamma      = 0.99,
            lam        = 0.95,
            device     = dev,
        )

        for t in buffer._transitions:
            if t.done:
                total += 1
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

    result = {
        "win_rate": win_rate,
        "wins":     wins,
        "losses":   losses,
        "ties":     ties,
        "total":    total,
        "ci_low":   ci_low,
        "ci_high":  ci_high,
    }

    print(
        f"  win_rate={win_rate*100:.1f}%  "
        f"(95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%])  "
        f"W={wins}  L={losses}  T={ties}  "
        f"[{elapsed:.1f}s]"
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
