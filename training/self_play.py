"""
CynthAI_v2 Self-Play Training — PPO update loop.

Outer loop (each update):
  1. Sample opponent via mixing: 80% pool / 10% Random / 10% FullOffense
  2. collect_rollout -> complete episodes, GAE pre-computed
  3. n_epochs x minibatch PPO updates
       - forward agent (with gradients)
       - pred auxiliary loss (PredictionHeads on opponent tokens)
       - PPO losses (policy + value + entropy)
       - clip_grad_norm 0.5 + AdamW step
  4. Update LR (warmup + cosine decay)
  5. Log metrics to stdout
  6. Every eval_freq updates: evaluate vs Random / FullOffense / pool (500 games each)
       - Snapshot agent into opponent pool when pool eval WR > threshold
  7. Checkpoint .pt every checkpoint_freq updates

Opponent pool:
  Starts empty - agent plays itself (simplest, identical distributions).
  A deep-copy snapshot (requires_grad=False) is added when win_rate exceeds
  pool_snapshot_threshold (default 0.55), with a cooldown between snapshots.
  Pool keeps at most pool_size snapshots (oldest dropped).

Logging columns (every log_every updates):
  update | win% | policy | value | entropy | pred | total | lr | time | eps |
  grad_norm | clip_frac | explained_variance | opp
Plus eval lines every eval_freq updates showing WR vs random, fulloff, pool.
"""

from __future__ import annotations

import copy
import math
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
from training.rollout import collect_rollout, RandomPolicy
from training.losses import compute_losses
from training.evaluate import run_eval
from env.bots import FullOffensePolicy


# -- Training configuration -------------------------------------------------------

@dataclass
class TrainingConfig:
    # Optimiser
    lr:            float = 2.5e-4   # P6: lowered from 3e-4
    lr_min:        float = 1e-5
    total_updates: int   = 2000
    warmup_steps:  int   = 20       # P6: linear warmup before cosine decay

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
    c_value:       float = 1.0      # P5: increased from 0.5
    c_entropy:     float = 0.01
    c_pred:        float = 0.5
    max_grad_norm: float = 0.5
    weight_decay:  float = 1e-4     # P4: L2 regularisation

    # Opponent pool
    pool_size:        int   = 20        # P3: increased from 5 for more diversity
    pool_snapshot_threshold: float = 0.55  # P3: snapshot agent when win_rate exceeds this
    pool_cooldown:    int   = 5         # P3: min updates between snapshots

    # Checkpointing / logging
    checkpoint_dir:  str = "checkpoints"
    checkpoint_freq: int = 50
    log_every:       int = 1
    win_rate_window: int = 100
    eval_freq:       int = 20      # P3: run evaluation every N updates
    eval_n_games:    int = 500     # P3: games per opponent in evaluation

    # Device
    device: str = "cpu"

    # Resume
    resume: str = ""   # path to checkpoint .pt to resume from


# -- Opponent pool ----------------------------------------------------------------

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


# -- Opponent token slice helper --------------------------------------------------

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


# -- Main training loop -----------------------------------------------------------

def train(cfg: TrainingConfig = TrainingConfig()) -> None:
    """PPO self-play loop. Blocks until cfg.total_updates complete."""
    device = torch.device(cfg.device)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    agent     = CynthAIAgent().to(device)
    optimizer = torch.optim.AdamW(
        agent.parameters(),
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
        eps          = 1e-5,
    )

    start_update = 1
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=True)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"] + 1
        print(f"Resumed from {cfg.resume}  (update {ckpt['update']})")

    pool        = OpponentPool(pool_size=cfg.pool_size)
    win_history : deque[int] = deque(maxlen=cfg.win_rate_window)
    total_eps   = 0
    last_snapshot_update = 0  # P3: cooldown counter for WR-based snapshots

    n_params = sum(p.numel() for p in agent.parameters())
    print(
        f"CynthAI_v2 self-play  |  {n_params:,} params  "
        f"|  device={cfg.device}  |  {cfg.total_updates} updates"
    )
    print(
        f"{'update':>7}  {'win%':>6}  "
        f"{'policy':>8}  {'value':>8}  {'entropy':>8}  "
        f"{'pred':>8}  {'total':>8}  {'lr':>8}  info  opp"
    )

    for update in range(start_update, cfg.total_updates + 1):
        t0 = time.perf_counter()

        # -- 1a. Opponent selection (mixing) -----------------------------------------
        # P3: 80% pool / 10% RandomPolicy / 10% FullOffense
        roll = random.random()
        if roll < 0.10:
            opponent = RandomPolicy()
            opp_label = "rand"
        elif roll < 0.20:
            opponent = FullOffensePolicy()
            opp_label = "fo"
        else:
            opponent = pool.sample(agent)
            opp_label = f"pool({len(pool)})"

        # -- 1b. Rollout collection ----------------------------------------------------
        agent.eval()

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

        # -- 2. PPO update epochs -----------------------------------------------------
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
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                for k, v in losses.items():
                    loss_acc[k] += v.item()
                loss_acc["grad_norm"] += grad_norm.item()
                n_steps += 1

        # -- 3. LR schedule (warmup + cosine) -----------------------------------------
        # P6: manual LR scheduling for clean warmup -> cosine decay
        if update <= cfg.warmup_steps:
            lr = cfg.lr * update / cfg.warmup_steps
        else:
            progress = (update - cfg.warmup_steps) / (cfg.total_updates - cfg.warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr       = max(cfg.lr_min, cfg.lr * cosine)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # -- 4. Logging ---------------------------------------------------------------
        if update % cfg.log_every == 0:
            ns      = max(n_steps, 1)
            elapsed = time.perf_counter() - t0
            print(
                f"{update:>7}  {win_rate*100:>5.1f}%  "
                f"{loss_acc['policy']/ns:>8.4f}  "
                f"{loss_acc['value']/ns:>8.4f}  "
                f"{loss_acc['entropy']/ns:>8.4f}  "
                f"{loss_acc['pred']/ns:>8.4f}  "
                f"{loss_acc['total']/ns:>8.4f}  "
                f"{lr:.2e}  "
                f"[{elapsed:.1f}s  eps={total_eps}  pool={len(pool)}  "
                f"gn={loss_acc['grad_norm']/ns:.3f}  "
                f"cf={loss_acc['clip_frac']/ns:.3f}  "
                f"ev={loss_acc['explained_variance']/ns:.3f}  "
                f"opp={opp_label}]"
            )

        # -- 5. Periodic evaluation ---------------------------------------------------
        # P3: every eval_freq updates, run 500 games vs each opponent type
        if update % cfg.eval_freq == 0:
            agent.eval()
            eval_t0 = time.perf_counter()

            # Fixed opponents: Random and FullOffense (deterministic, no need to sample)
            eval_opponents = {
                "random":  RandomPolicy(),
                "fulloff": FullOffensePolicy(),
            }
            eval_results = {}
            for label, opp in eval_opponents.items():
                res = run_eval(
                    agent    = agent,
                    opponent = opp,
                    n_games  = cfg.eval_n_games,
                    n_envs   = cfg.n_envs,
                    format_id= cfg.format_id,
                    device   = device,
                )
                eval_results[label] = res
                print(f"  eval {label}: WR={res['win_rate']*100:.1f}%  "
                      f"W={res['wins']} L={res['losses']}  "
                      f"({res['total']} games)")

            # Pool eval: sample a fresh opponent each batch for representative WR
            eval_results["pool"] = run_eval(
                agent             = agent,
                n_games           = cfg.eval_n_games,
                n_envs            = cfg.n_envs,
                format_id         = cfg.format_id,
                device            = device,
                opponent_sampler  = lambda: pool.sample(agent),
            )
            pool_wr = eval_results["pool"]["win_rate"]
            print(f"  eval pool: WR={pool_wr*100:.1f}%  "
                  f"W={eval_results['pool']['wins']} L={eval_results['pool']['losses']}  "
                  f"({eval_results['pool']['total']} games)")

            # Snapshot agent when pool eval WR exceeds threshold
            if (pool_wr > cfg.pool_snapshot_threshold
                and update - last_snapshot_update >= cfg.pool_cooldown):
                pool.add(agent)
                last_snapshot_update = update
                print(f"  -> snapshot added (pool eval WR={pool_wr*100:.1f}%)")

            eval_elapsed = time.perf_counter() - eval_t0
            print(f"  eval finished [{eval_elapsed:.1f}s]")

        # -- 7. Checkpoint (standalone; snapshot is inside eval block) -----------------
        if update % cfg.checkpoint_freq == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"agent_{update:06d}.pt"
            torch.save({
                "update":    update,
                "model":     agent.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")


# -- Entry point --------------------------------------------------------------------

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