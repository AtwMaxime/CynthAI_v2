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
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

from model.agent import CynthAIAgent
from model.embeddings import PokemonBatch, apply_reveal_mask
from model.prediction_heads import PredictionHeads
from model.backbone import K_TURNS
from training.rollout import collect_rollout, RandomPolicy
from training.losses import compute_losses
from training.evaluate import run_eval
from training.monitor import save_eval_plots
from env.bots import FullOffensePolicy
import gc
from tqdm import tqdm


def _save_fig(fig, path_stem: str):
    """Save figure as both PNG and PDF."""
    for ext in [".png", ".pdf"]:
        fig.savefig(f"{path_stem}{ext}", dpi=150, bbox_inches="tight")


def save_attention_maps(
    agent: "CynthAIAgent",
    save_dir: str,
    tag: str = "",
    device: torch.device = torch.device("cpu"),
) -> None:
    """Run a forward pass on a real battle state and save attention heatmap PNGs.

    Saves one grid image (all layers × heads) to {save_dir}/attn_{tag}.png
    and one per layer to {save_dir}/attn_{tag}_layer{L}.png
    """
    try:
        from simulator import PyBattle
        from training.rollout import BattleWindow, encode_state
        from model.embeddings import collate_features, collate_field_features
        from model.backbone import K_TURNS, N_SLOTS, SEQ_LEN
    except ImportError:
        return  # missing dependencies during early development

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    agent.eval()
    b = PyBattle("gen9randombattle", 42)
    state = b.get_state()
    window = BattleWindow()
    for _ in range(K_TURNS):
        poke_feats, field_feat = encode_state(state, side_idx=0)
        window.push(poke_feats, field_feat)

    poke_turns, field_turns = window.as_padded()
    flat_poke = [p for turn in poke_turns for p in turn]
    poke_batch = collate_features([flat_poke]).to(device)
    field_tensor = collate_field_features(field_turns).field.unsqueeze(0).to(device)

    with torch.no_grad():
        pokemon_tokens = agent.poke_emb(poke_batch)
        result = agent.backbone.get_attention_maps(pokemon_tokens, field_tensor)

    attn_maps = result["attention_maps"]
    if not attn_maps:
        print(f"  WARNING: no attention maps captured, skipping attention save")
        return
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[1]
    labels = result["token_labels"]

    def _plot_single(ax, mat, title_str, show_labels=True, colorbar=False):
        """Plot one attention heatmap with proper token labels and optional colorbar."""
        im = ax.imshow(mat, cmap="viridis", aspect="equal", vmin=0.0, vmax=mat.max())
        ax.set_title(title_str, fontsize=9)
        ax.set_xlabel("Key token", fontsize=6)
        ax.set_ylabel("Query token", fontsize=6)
        ax.set_xticks(range(SEQ_LEN))
        ax.set_yticks(range(SEQ_LEN))
        ax.set_xticklabels(labels if show_labels else [], fontsize=4, rotation=90)
        ax.set_yticklabels(labels if show_labels else [], fontsize=4)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 1. Grid: all layers × heads
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5 * n_heads, 5 * n_layers))
    fig.suptitle(f"Attention Maps — {tag}", fontsize=14, fontweight="bold")
    if n_layers == 1:
        axes = [axes]
    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li][hi] if n_layers > 1 else axes[hi]
            _plot_single(ax, attn_maps[li][0, hi].numpy(),
                         f"Layer {li}, Head {hi}",
                         show_labels=(li == n_layers - 1),
                         colorbar=(hi == n_heads - 1))
    fig.tight_layout()
    attn_dir = Path(save_dir) / "attn"
    attn_dir.mkdir(parents=True, exist_ok=True)
    _save_fig(fig, attn_dir / f"{tag}_grid")
    plt.close(fig)

    # 2. Individual per-head plots with colorbar + top-5 analysis
    for li in range(n_layers):
        for hi in range(n_heads):
            mat = attn_maps[li][0, hi].numpy()
            fig, ax = plt.subplots(figsize=(7, 6))
            _plot_single(ax, mat, f"Layer {li}, Head {hi}", show_labels=True, colorbar=True)

            # Top-5 attended keys for current-turn query tokens (last 13)
            cur_start = SEQ_LEN - 13
            topk = 5
            txt_lines = ["Top-5 attended keys (current turn queries):"]
            for query_offset in range(13):
                query_idx = cur_start + query_offset
                q_label = labels[query_idx]
                row = mat[query_idx]
                top_indices = np.argsort(row)[-topk:][::-1]
                short_q = q_label.split("_", 1)[1] if "_" in q_label else q_label
                keys_str = " ".join(
                    lbl.split("_", 1)[1] if "_" in lbl else lbl
                    for lbl in [labels[i] for i in top_indices]
                )
                txt_lines.append(f"{short_q:>10} -> {keys_str}")
            ax.text(1.02, 0.98, "\n".join(txt_lines), transform=ax.transAxes,
                    fontsize=5.5, family="monospace", verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

            fig.tight_layout()
            _save_fig(fig, attn_dir / f"{tag}_L{li}_H{hi}")
            plt.close(fig)

    print(f"  -> attention maps saved to {attn_dir}/  [{tag}]")


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
    c_value:       float = 2.0      # P11b: increased from 1.0 for critic stability
    c_entropy:     float = 0.01
    c_pred:        float = 0.5
    c_attn_entropy: float = 0.001  # P14: cross-attention entropy regularisation
    c_attn_rank:    float = 0.001  # P18: cross-attention rank regularisation (von Neumann entropy)
    max_grad_norm: float = 0.5
    weight_decay:  float = 1e-4     # P4: L2 regularisation

    # Opponent pool
    pool_size:        int   = 10        # P10c: reduced from 20; periodic snapshots now
    pool_snapshot_threshold: float = 0.55  # P3: snapshot agent when win_rate exceeds this
    pool_cooldown:    int   = 5         # P3: min updates between snapshots

    # P10b: EMA opponent
    ema_decay:        float = 0.995  # smoothing factor for EMA weights
    ema_warmup:       int   = 5      # updates before EMA is used as opponent (pure self-play before)

    # P10c: periodic pool snapshots (more reliable than WR-based)
    pool_snapshot_freq: int = 100    # snapshot agent into pool every N updates

    # P1: POMDP mask curriculum
    mask_schedule:      str   = "linear"  # "linear", "exp", "step", "phase"
    mask_warmup:        int   = 200       # updates before masking begins
    mask_max_ratio:     float = 1.0       # final masking probability
    mask_exp_k:         float = 3.0       # exponential ramp rate
    mask_step_update:   int   = 500       # plateau length for step schedule
    mask_phase_breakpoints: tuple = ()    # update breakpoints for "phase" mode
    mask_phase_values:      tuple = ()    # mask ratios at each phase

    # P2: Reward curriculum
    dense_schedule:         str   = "linear"  # "linear", "exp", "step", "phase"
    dense_warmup:           int   = 0         # updates before decay begins
    dense_min_scale:        float = 0.25      # floor for dense_scale
    dense_exp_k:            float = 3.0       # exponential decay rate
    dense_step_update:      int   = 500       # plateau length for step schedule
    dense_phase_breakpoints: tuple = ()       # update breakpoints for "phase" mode
    dense_phase_values:      tuple = ()       # dense_scale at each phase

    # Checkpointing / logging
    # Run metadata
    run_name:      str = ""         # auto-generated if empty: curriculum_max_YYYYMMDD_HHMM
    checkpoint_dir: str = ""         # auto-generated from run_name if empty
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


# -- P10b: EMA Opponent -------------------------------------------------------------

class EMAOpponent:
    """
    Exponential Moving Average of agent weights.

    Provides a stable, lagging self-play opponent that smoothly tracks the
    online agent. Unlike OpponentPool (discrete snapshots), the EMA updates
    every training step via polyak averaging::

        ema_params = decay * ema_params + (1 - decay) * online_params

    Always available — no empty pool problem, no WR threshold to meet.
    """

    def __init__(self, agent: CynthAIAgent, decay: float = 0.995):
        self.ema = copy.deepcopy(agent)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, agent: CynthAIAgent) -> None:
        """Polyak-average the online weights into the EMA copy."""
        with torch.no_grad():
            for ema_p, online_p in zip(self.ema.parameters(), agent.parameters()):
                ema_p.data.mul_(self.decay).add_(online_p.data, alpha=1 - self.decay)

    def sample(self, current_agent: CynthAIAgent) -> CynthAIAgent:
        return self.ema


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


# -- Mask schedule (P1 POMDP curriculum) ----------------------------------------

def compute_mask_ratio(
    update: int,
    total:  int,
    schedule: str = "linear",
    warmup: int = 200,
    max_ratio: float = 1.0,
    exp_k: float = 3.0,
    step_update: int = 500,
    phase_breakpoints: tuple = (),
    phase_values: tuple = (),
) -> float:
    """Return masking probability at the given update.

    Schedules:
      linear : linear ramp 0.0 -> max_ratio over [warmup, total]
      exp    : exp ramp 0.0 -> max_ratio (faster initial rise)
      step   : 0.0 until step_update, then max_ratio
      phase  : manual phases defined by (breakpoints, values) tuples.
               e.g. breakpoints=(600, 2500), values=(0.0, 0.5, 1.0) means
               [0, 600] → 0.0, (600, 2500] → 0.5, (2500, ∞) → 1.0
               When using "phase", warmup is ignored (breakpoints are absolute).
    """
    if schedule == "phase":
        if not phase_breakpoints:
            return 0.0
        for i, bp in enumerate(phase_breakpoints):
            if update <= bp:
                return phase_values[i]
        return phase_values[-1]

    if update <= warmup:
        return 0.0

    progress = (update - warmup) / (total - warmup)

    if schedule == "linear":
        return min(max_ratio, progress * max_ratio)
    elif schedule == "exp":
        return max_ratio * (1.0 - math.exp(-exp_k * progress))
    elif schedule == "step":
        return max_ratio if update - warmup >= step_update else 0.0
    else:
        raise ValueError(f"Unknown mask_schedule: {schedule}")


# -- Dense scale schedule (P2 reward curriculum) ---------------------------------

def compute_dense_scale(
    update: int,
    total:  int,
    schedule: str = "linear",
    warmup: int = 0,
    min_scale: float = 0.25,
    exp_k: float = 3.0,
    step_update: int = 500,
    phase_breakpoints: tuple = (),
    phase_values: tuple = (),
) -> float:
    """Return dense_scale for non-terminal rewards at the given update.

    Schedules:
      linear : linear decay 1.0 -> min_scale over [warmup, total]
      exp    : exp decay from 1.0 towards min_scale
      step   : 1.0 until step_update, then min_scale
      phase  : manual phases defined by (breakpoints, values) tuples.
               e.g. breakpoints=(600, 2500), values=(1.0, 0.5, 0.1) means
               [0, 600] → 1.0, (600, 2500] → 0.5, (2500, ∞) → 0.1
               When using "phase", warmup is ignored (breakpoints are absolute).
    """
    if schedule == "phase":
        if not phase_breakpoints:
            return 1.0
        for i, bp in enumerate(phase_breakpoints):
            if update <= bp:
                return phase_values[i]
        return phase_values[-1]

    if update <= warmup:
        return 1.0

    progress = (update - warmup) / (total - warmup)

    if schedule == "linear":
        return max(min_scale, 1.0 - progress * (1.0 - min_scale))
    elif schedule == "exp":
        return max(min_scale, (1.0 - min_scale) * math.exp(-exp_k * progress) + min_scale)
    elif schedule == "step":
        return min_scale if update - warmup >= step_update else 1.0
    else:
        raise ValueError(f"Unknown dense_schedule: {schedule}")


# -- Main training loop -----------------------------------------------------------

def train(cfg: TrainingConfig = TrainingConfig()) -> None:
    """PPO self-play loop. Blocks until cfg.total_updates complete."""
    device = torch.device(cfg.device)

    # Auto-generate run name and checkpoint dir
    if cfg.resume:
        # When resuming, reuse the existing checkpoint directory
        resume_path = Path(cfg.resume)
        cfg.checkpoint_dir = str(resume_path.parent)
        cfg.run_name = Path(cfg.checkpoint_dir).name
    else:
        if not cfg.run_name:
            cfg.run_name = f"curriculum_max_{datetime.now():%Y%m%d_%H%M}"
        if not cfg.checkpoint_dir:
            cfg.checkpoint_dir = str(Path("checkpoints") / cfg.run_name)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # CSV loggers — don't re-write headers on resume
    metrics_path = Path(cfg.checkpoint_dir) / "metrics.csv"
    eval_path    = Path(cfg.checkpoint_dir) / "eval.csv"
    _first_metrics = not metrics_path.exists()
    _first_eval    = not eval_path.exists()

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
        # Restore saved config (except resume path — keep the new one)
        if "config" in ckpt:
            saved = dict(ckpt["config"])
            saved["resume"] = cfg.resume
            cfg = TrainingConfig(**saved)
        print(f"Resumed from {cfg.resume}  (update {ckpt['update']})")
        print(f"  run_name={cfg.run_name}  total_updates={cfg.total_updates}")

    pool        = OpponentPool(pool_size=cfg.pool_size)
    ema_opponent = EMAOpponent(agent, decay=cfg.ema_decay)  # P10b
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

    pbar = tqdm(total=cfg.total_updates - start_update + 1,
                desc="Training", unit="updates", position=0, leave=True)
    for update in range(start_update, cfg.total_updates + 1):
        t0 = time.perf_counter()

        # -- 1a. Opponent selection (P10b-P10c: EMA + pool + fixed policies) -----------
        # Mixing: 10% Random / 10% FullOffense / 60% EMA (or self-play before warmup)
        #         / 20% pool (or EMA if pool empty)
        roll = random.random()
        if roll < 0.10:
            opponent = RandomPolicy()
            opp_label = "rand"
        elif roll < 0.20:
            opponent = FullOffensePolicy()
            opp_label = "fo"
        elif roll < 0.80:
            # 70% — EMA opponent (stable lagging self-play)
            if update > cfg.ema_warmup:
                opponent = ema_opponent.sample(agent)
                opp_label = "ema"
            else:
                opponent = agent
                opp_label = "self"
        else:
            # 20% — pool (diversity) or EMA fallback
            if len(pool) > 0:
                opponent = pool.sample(agent)
                opp_label = f"pool({len(pool)})"
            else:
                opponent = ema_opponent.sample(agent) if update > cfg.ema_warmup else agent
                opp_label = "ema" if update > cfg.ema_warmup else "self"

        # -- 1b. Rollout collection ----------------------------------------------------
        agent.eval()
        mask_ratio = compute_mask_ratio(
            update, cfg.total_updates,
            schedule         = cfg.mask_schedule,
            warmup           = cfg.mask_warmup,
            max_ratio        = cfg.mask_max_ratio,
            exp_k            = cfg.mask_exp_k,
            step_update      = cfg.mask_step_update,
            phase_breakpoints = cfg.mask_phase_breakpoints,
            phase_values     = cfg.mask_phase_values,
        )
        dense_scale = compute_dense_scale(
            update, cfg.total_updates,
            schedule          = cfg.dense_schedule,
            warmup            = cfg.dense_warmup,
            min_scale         = cfg.dense_min_scale,
            exp_k             = cfg.dense_exp_k,
            step_update       = cfg.dense_step_update,
            phase_breakpoints = cfg.dense_phase_breakpoints,
            phase_values      = cfg.dense_phase_values,
        )

        buffer = collect_rollout(
            agent_self = agent,
            agent_opp  = opponent,
            n_envs     = cfg.n_envs,
            min_steps  = cfg.min_steps,
            format_id  = cfg.format_id,
            gamma      = cfg.gamma,
            lam        = cfg.lam,
            device     = device,
            mask_ratio = mask_ratio,
            dense_scale = dense_scale,
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

                # POMDP masking: apply reveal mask before forward pass
                # build targets from GROUND TRUTH (unmasked batch)
                if "reveal_species" in batch:
                    masked_poke = apply_reveal_mask(
                        batch["poke_batch"],
                        reveal_species = batch["reveal_species"],
                        reveal_item    = batch["reveal_item"],
                        reveal_ability = batch["reveal_ability"],
                        reveal_tera    = batch["reveal_tera"],
                        reveal_moves   = batch["reveal_moves"],
                        mask_ratio     = mask_ratio,
                    )
                else:
                    masked_poke = batch["poke_batch"]  # backward compat

                out = agent(
                    masked_poke,
                    batch["field_tensor"],
                    batch["move_idx"],
                    batch["pp_ratio"],
                    batch["move_disabled"],
                    batch["mechanic_id"],
                    batch["mechanic_type_idx"],
                    batch["action_mask"],
                )

                opp_batch = _slice_opp_batch(batch["poke_batch"])  # ground truth
                targets   = PredictionHeads.build_targets(opp_batch)

                # Override prediction masks with RevealedTracker si POMDP actif
                # (masks ground truth = toujours True, sauf si masking overridé)
                if mask_ratio > 0 and "reveal_species" in batch:
                    it_t, ab_t, te_t, mv_t, it_m, ab_m, te_m, mv_m, st_t, st_m = targets
                    it_m = batch["reveal_item"]                              # [B, 6] bool
                    ab_m = batch["reveal_ability"]                           # [B, 6] bool
                    te_m = batch["reveal_tera"]                              # [B, 6] bool
                    mv_m = batch["reveal_moves"].any(dim=-1)                 # [B, 6, 4] → [B, 6] bool
                    targets = (it_t, ab_t, te_t, mv_t, it_m, ab_m, te_m, mv_m, st_t, st_m)

                pred_loss = PredictionHeads.compute_loss(out.pred_logits, *targets)["total"]
                acc = PredictionHeads.compute_accuracy(out.pred_logits, *targets)

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

                # P14: cross-attention entropy regularisation (maximise = spread per query)
                attn_entropy_val = out.attn_entropy
                losses["attn_entropy"] = attn_entropy_val.detach()
                losses["total"] = losses["total"] - cfg.c_attn_entropy * attn_entropy_val

                # P18: cross-attention rank regularisation (maximise von Neumann entropy)
                attn_rank_val = out.attn_rank
                losses["attn_rank"] = attn_rank_val.detach()
                # Scale rank deficiency: ln(13) - vn_entropy, so adding it penalises low rank
                max_vn = math.log(13.0)
                rank_loss = cfg.c_attn_rank * (max_vn - attn_rank_val)
                losses["rank_reg"] = rank_loss.detach()
                losses["total"] = losses["total"] + rank_loss

                losses["total"].backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        loss_acc[k] += v.item()
                    else:
                        loss_acc[k] += v
                loss_acc["grad_norm"] += grad_norm.item()
                for k, v in acc.items():
                    loss_acc[k] += v
                n_steps += 1

        # -- 2b. P10b: EMA weight update -----------------------------------------------
        ema_opponent.update(agent)

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
            pbar.write(
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
                f"am={loss_acc.get('adv_mean',0)/ns:.3f}  "
                f"as={loss_acc.get('adv_std',0)/ns:.3f}  "
                f"rd={loss_acc.get('ratio_dev',0)/ns:.4f}  "
                f"ia={loss_acc.get('item_acc',0)/ns:.2f} "
                f"aa={loss_acc.get('ability_acc',0)/ns:.2f} "
                f"ma={loss_acc.get('move_recall',0)/ns:.2f}  "
                f"ae={loss_acc.get('attn_entropy',0)/ns:.4f}  "
                f"ar={loss_acc.get('attn_rank',0)/ns:.4f}  "
                f"rr={loss_acc.get('rank_reg',0)/ns:.4f}  "
                f"opp={opp_label}]"
            )
            pbar.set_postfix_str(f"WR={win_rate*100:.1f}%  "
                                 f"loss={loss_acc['total']/ns:.3f}  "
                                 f"lr={lr:.2e}  "
                                 f"opp={opp_label}  "
                                 f"eps={total_eps}")

            # CSV log
            with open(metrics_path, "a", newline="") as f:
                import csv
                row = {
                    "update": update, "win_rate": win_rate,
                    "policy": loss_acc["policy"]/ns, "value": loss_acc["value"]/ns,
                    "entropy": loss_acc["entropy"]/ns, "pred": loss_acc["pred"]/ns,
                    "total": loss_acc["total"]/ns, "lr": lr,
                    "grad_norm": loss_acc["grad_norm"]/ns,
                    "clip_frac": loss_acc["clip_frac"]/ns,
                    "explained_variance": loss_acc["explained_variance"]/ns,
                    "adv_mean": loss_acc.get("adv_mean", 0)/ns,
                    "adv_std": loss_acc.get("adv_std", 0)/ns,
                    "ratio_dev": loss_acc.get("ratio_dev", 0)/ns,
                    "eps": total_eps, "pool": len(pool),
                    "mask_ratio": mask_ratio, "dense_scale": dense_scale,
                    "item_acc": loss_acc.get("item_acc", 0) / ns,
                    "ability_acc": loss_acc.get("ability_acc", 0) / ns,
                    "tera_acc": loss_acc.get("tera_acc", 0) / ns,
                    "move_recall": loss_acc.get("move_recall", 0) / ns,
                    "attn_entropy": loss_acc.get("attn_entropy", 0) / ns,
                    "attn_rank": loss_acc.get("attn_rank", 0) / ns,
                    "rank_reg": loss_acc.get("rank_reg", 0) / ns,
                    "opp": opp_label,
                }
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if _first_metrics:
                    w.writeheader()
                    _first_metrics = False
                w.writerow(row)

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
                    mask_ratio = mask_ratio,
                )
                eval_results[label] = res
                print(f"  eval {label}: WR={res['win_rate']*100:.1f}%  "
                      f"W={res['wins']} L={res['losses']}  "
                      f"({res['total']} games)")

            # EMA eval (P10b): measure progress against the lagging target
            # Also captures cross-attention weights for diagnostic visualization
            eval_results["ema"] = run_eval(
                agent             = agent,
                n_games           = cfg.eval_n_games,
                n_envs            = cfg.n_envs,
                format_id         = cfg.format_id,
                device            = device,
                mask_ratio        = mask_ratio,
                opponent_sampler  = lambda: ema_opponent.sample(agent),
                capture_cross_attn = True,
                compute_cos_sim    = True,
            )
            ema_wr = eval_results["ema"]["win_rate"]
            print(f"  eval ema: WR={ema_wr*100:.1f}%  "
                  f"W={eval_results['ema']['wins']} L={eval_results['ema']['losses']}  "
                  f"({eval_results['ema']['total']} games)")

            # Pool eval: sample a fresh opponent each batch for representative WR
            eval_results["pool"] = run_eval(
                agent             = agent,
                n_games           = cfg.eval_n_games,
                n_envs            = cfg.n_envs,
                format_id         = cfg.format_id,
                device            = device,
                mask_ratio        = mask_ratio,
                opponent_sampler  = lambda: pool.sample(agent),
            )
            pool_wr = eval_results["pool"]["win_rate"]
            print(f"  eval pool: WR={pool_wr*100:.1f}%  "
                  f"W={eval_results['pool']['wins']} L={eval_results['pool']['losses']}  "
                  f"({eval_results['pool']['total']} games)")

            # CSV eval log
            with open(eval_path, "a", newline="") as f:
                import csv
                row = {"update": update}
                for label in ("random", "fulloff", "ema", "pool"):
                    r = eval_results[label]
                    row[f"{label}_wr"] = r["win_rate"]
                    row[f"{label}_w"]  = r["wins"]
                    row[f"{label}_l"]  = r["losses"]
                    row[f"{label}_n"]  = r["total"]
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if _first_eval:
                    w.writeheader()
                    _first_eval = False
                w.writerow(row)

            # Save evaluation diagnostic plots
            try:
                save_eval_plots(eval_results, cfg.checkpoint_dir,
                                tag=f"update{update:06d}")
            except Exception as e:
                print(f"  WARNING: eval plots failed: {e}")

            # P10c: periodic pool snapshot (replaces WR-based snapping)
            if (update % cfg.pool_snapshot_freq == 0
                and update > 0
                and update - last_snapshot_update >= cfg.pool_cooldown):
                pool.add(agent)
                last_snapshot_update = update
                print(f"  -> pool snapshot added (size={len(pool)})")

            eval_elapsed = time.perf_counter() - eval_t0
            print(f"  eval finished [{eval_elapsed:.1f}s]")

        # -- 7. Checkpoint (standalone; snapshot is inside eval block) -----------------
        if update % cfg.checkpoint_freq == 0:
            ckpt_path = Path(cfg.checkpoint_dir) / f"agent_{update:06d}.pt"
            torch.save({
                "update":    update,
                "config":    asdict(cfg),
                "model":     agent.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

            # Save attention maps alongside checkpoint
            save_attention_maps(agent, cfg.checkpoint_dir,
                                tag=f"update{update:06d}", device=device)

        # Collect garbage to prevent Rust PyBattle objects from accumulating
        gc.collect()

        pbar.update(1)

    pbar.close()


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