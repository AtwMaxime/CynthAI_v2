"""
CynthAI_v2 Self-Play Training — PPO update loop.

Outer loop (each update):
  1. Sample opponent from pool (or live agent if pool empty → pure self-play)
  2. collect_rollout → complete episodes, GAE pre-computed
  3. n_epochs × minibatch PPO updates
       - forward agent (with gradients)
       - pred auxiliary loss (PredictionHeads on opponent tokens)
       - PPO losses (policy + value + entropy)
       - clip_grad_norm 0.5 + Adam step
  4. Step linear LR scheduler
  5. Log metrics to stdout
  6. Checkpoint .pt + snapshot agent into opponent pool every N updates

Opponent pool:
  Starts empty — agent plays itself (simplest, identical distributions).
  A deep-copy snapshot (requires_grad=False) is added every pool_update_freq
  updates. Pool keeps at most pool_size snapshots (oldest dropped).

Logging columns (every log_every updates):
  update | win% | policy | value | entropy | pred | total | lr | time | eps
"""

from __future__ import annotations

import copy
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from model.agent import CynthAIAgent
from model.embeddings import PokemonBatch
from model.prediction_heads import PredictionHeads
from model.backbone import K_TURNS
from training.rollout import collect_rollout
from training.losses import compute_losses


# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Optimiser
    lr:            float = 3e-4
    lr_min:        float = 1e-5
    total_updates: int   = 2000

    # Rollout
    n_envs:    int = 16
    min_steps: int = 512
    format_id: str = "gen9randombattle"

    # PPO
    n_epochs:      int   = 4
    batch_size:    int   = 128
    gamma:         float = 0.99
    lam:           float = 0.95
    clip_eps:      float = 0.2
    c_value:       float = 0.5
    c_entropy:     float = 0.01
    c_pred:        float = 0.5
    max_grad_norm: float = 0.5

    # Opponent pool
    pool_size:        int = 5
    pool_update_freq: int = 10   # snapshot every N updates

    # Checkpointing / logging
    checkpoint_dir:  str = "checkpoints"
    checkpoint_freq: int = 50
    log_every:       int = 1
    win_rate_window: int = 100

    # Device
    device: str = "cpu"

    # Resume
    resume: str = ""   # path to checkpoint .pt to resume from


# ── Opponent pool ─────────────────────────────────────────────────────────────

class OpponentPool:
    """
    Fixed-size pool of past agent snapshots for self-play diversity.

    When empty, sample() returns the live agent so training begins as pure
    self-play (both sides are the same object).
    """

    def __init__(self, pool_size: int = 5):
        self._pool: deque[CynthAIAgent] = deque(maxlen=pool_size)

    def add(self, agent: CynthAIAgent) -> None:
        """Deep-copy the agent (detached, eval mode) and push to pool."""
        snapshot = copy.deepcopy(agent)
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad_(False)
        self._pool.append(snapshot)

    def sample(self, current_agent: CynthAIAgent) -> CynthAIAgent:
        """Return a random past snapshot, or current_agent if pool is empty."""
        if not self._pool:
            return current_agent
        return random.choice(list(self._pool))

    def __len__(self) -> int:
        return len(self._pool)


# ── Opponent token slice helper ───────────────────────────────────────────────

# Within the K*12 flattened poke_batch, the current turn's opponent tokens
# occupy the last 6 positions: [(K-1)*12+6 : K*12]
_OPP_CUR = slice((K_TURNS - 1) * 12 + 6, K_TURNS * 12)   # [42:48] for K=4


def _slice_opp_batch(poke_batch: PokemonBatch) -> PokemonBatch:
    """Return a PokemonBatch with only the 6 current-turn opponent slots."""
    return PokemonBatch(
        species_idx = poke_batch.species_idx[:, _OPP_CUR],
        type1_idx   = poke_batch.type1_idx[:, _OPP_CUR],
        type2_idx   = poke_batch.type2_idx[:, _OPP_CUR],
        tera_idx    = poke_batch.tera_idx[:, _OPP_CUR],
        item_idx    = poke_batch.item_idx[:, _OPP_CUR],
        ability_idx = poke_batch.ability_idx[:, _OPP_CUR],
        move_idx    = poke_batch.move_idx[:, _OPP_CUR, :],
        scalars     = poke_batch.scalars[:, _OPP_CUR, :],
    )


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: TrainingConfig = TrainingConfig()) -> None:
    """PPO self-play loop. Blocks until cfg.total_updates complete."""
    device = torch.device(cfg.device)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    agent     = CynthAIAgent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1.0,
        end_factor   = cfg.lr_min / cfg.lr,
        total_iters  = cfg.total_updates,
    )

    start_update = 1
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=True)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_update = ckpt["update"] + 1
        print(f"Resumed from {cfg.resume}  (update {ckpt['update']})")

    pool        = OpponentPool(pool_size=cfg.pool_size)
    win_history : deque[int] = deque(maxlen=cfg.win_rate_window)
    total_eps   = 0

    n_params = sum(p.numel() for p in agent.parameters())
    print(
        f"CynthAI_v2 self-play  |  {n_params:,} params  "
        f"|  device={cfg.device}  |  {cfg.total_updates} updates"
    )
    print(
        f"{'update':>7}  {'win%':>6}  "
        f"{'policy':>8}  {'value':>8}  {'entropy':>8}  "
        f"{'pred':>8}  {'total':>8}  {'lr':>8}  info"
    )

    for update in range(start_update, cfg.total_updates + 1):
        t0 = time.perf_counter()

        # ── 1. Rollout collection ─────────────────────────────────────────────
        agent.eval()
        opponent = pool.sample(agent)

        buffer = collect_rollout(
            agent_self = agent,
            agent_opp  = opponent,
            n_envs     = cfg.n_envs,
            min_steps  = cfg.min_steps,
            format_id  = cfg.format_id,
            gamma      = cfg.gamma,
            lam        = cfg.lam,
            device     = device,
        )

        for t in buffer._transitions:
            if t.done:
                total_eps += 1
                win_history.append(1 if t.reward > 0.0 else 0)

        win_rate = sum(win_history) / len(win_history) if win_history else 0.5

        # ── 2. PPO update epochs ──────────────────────────────────────────────
        agent.train()
        loss_acc: defaultdict[str, float] = defaultdict(float)
        n_steps = 0

        for _epoch in range(cfg.n_epochs):
            for batch in buffer.minibatches(cfg.batch_size, device):
                optimizer.zero_grad()

                out = agent(
                    batch["poke_batch"],
                    batch["field_tensor"],
                    batch["move_idx"],
                    batch["pp_ratio"],
                    batch["move_disabled"],
                    batch["mechanic_id"],
                    batch["mechanic_type_idx"],
                    batch["action_mask"],
                )

                opp_batch = _slice_opp_batch(batch["poke_batch"])
                targets   = PredictionHeads.build_targets(opp_batch)
                pred_loss = PredictionHeads.compute_loss(out.pred_logits, *targets)["total"]

                losses = compute_losses(
                    logits_new   = out.action_logits,
                    log_prob_old = batch["log_prob_old"],
                    actions      = batch["actions"],
                    advantages   = batch["advantages"],
                    returns      = batch["returns"],
                    values       = out.value,
                    action_mask  = batch["action_mask"],
                    pred_loss    = pred_loss,
                    clip_eps     = cfg.clip_eps,
                    c_value      = cfg.c_value,
                    c_entropy    = cfg.c_entropy,
                    c_pred       = cfg.c_pred,
                )

                losses["total"].backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                for k, v in losses.items():
                    loss_acc[k] += v.item()
                n_steps += 1

        scheduler.step()

        # ── 3. Logging ────────────────────────────────────────────────────────
        if update % cfg.log_every == 0:
            ns         = max(n_steps, 1)
            elapsed    = time.perf_counter() - t0
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"{update:>7}  {win_rate*100:>5.1f}%  "
                f"{loss_acc['policy']/ns:>8.4f}  "
                f"{loss_acc['value']/ns:>8.4f}  "
                f"{loss_acc['entropy']/ns:>8.4f}  "
                f"{loss_acc['pred']/ns:>8.4f}  "
                f"{loss_acc['total']/ns:>8.4f}  "
                f"{current_lr:.2e}  "
                f"[{elapsed:.1f}s  eps={total_eps}  pool={len(pool)}]"
            )

        # ── 4. Checkpoint ─────────────────────────────────────────────────────
        if update % cfg.checkpoint_freq == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"agent_{update:06d}.pt"
            torch.save({
                "update":    update,
                "model":     agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

        # ── 5. Update opponent pool ───────────────────────────────────────────
        if update % cfg.pool_update_freq == 0:
            pool.add(agent)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",        default="",      help="checkpoint .pt to resume from")
    parser.add_argument("--total_updates", type=int, default=2000)
    parser.add_argument("--n_envs",        type=int, default=16)
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        device        = args.device,
        n_envs        = args.n_envs,
        min_steps     = 512,
        total_updates = args.total_updates,
        resume        = args.resume,
    )
    train(cfg)
