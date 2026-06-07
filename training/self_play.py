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
import torch.nn.functional as F

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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


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

    n_toks = len(labels)   # 53 (CLS + 52 battle tokens)

    def _plot_single(ax, mat, title_str, show_labels=True, colorbar=False):
        """Plot one attention heatmap with proper token labels and optional colorbar."""
        im = ax.imshow(mat, cmap="viridis", aspect="equal", vmin=0.0, vmax=mat.max())
        ax.set_title(title_str, fontsize=9)
        ax.set_xlabel("Key token", fontsize=6)
        ax.set_ylabel("Query token", fontsize=6)
        ax.set_xticks(range(n_toks))
        ax.set_yticks(range(n_toks))
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
            cur_start = n_toks - 13
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
    probe_freq:      int = 5      # run probes every N evals (0 = disabled)
    probe_min_steps: int = 2048   # rollout transitions for probing

    # Independent critic (optional — separate Transformer, no shared weights with actor)
    use_independent_critic: bool  = False
    critic_n_layers:        int   = 2      # Transformer depth (actor uses 3)
    critic_lr:              float = 5e-4   # independent learning rate
    critic_wd:              float = 1e-4   # independent weight decay
    critic_grad_norm:       float = 0.5    # independent gradient clip

    # Action-aware critic (cross-attention on action embeddings)
    critic_action_aware:   bool = False  # enable cross-attn on action embeds
    critic_mask_actions:   bool = True   # mask illegal actions in cross-attn
    critic_n_cross_layers: int  = 1      # number of cross-attn layers

    # Critic from backbone: feed backbone's enriched tokens to critic instead of raw inputs
    critic_from_backbone:  bool = False

    # Critic-stability diagnostics / switches
    value_dump_threshold: float = 0.0  # log warning when |vp_max| > threshold (0 = disabled)
    critic_value_bound:   float = 0.0  # Switch A: tanh squash ±bound on critic output (0 = off)
    value_target_clip:    float = 0.0  # Switch B: clip return targets to ±clip (0 = off)

    # Victory head — auxiliary BCE loss on the CLS token of IndependentCritic (or backbone)
    use_victory_head: bool  = False
    c_victory:        float = 0.1   # weight for victory BCE loss
    # Gradient control for CLS token (shared backbone only, ignored when use_independent_critic=True)
    # cls_backbone_grad=False uses 2 transformer passes to isolate cls_token from actor gradient.
    cls_value_grad:    bool = True
    cls_victory_grad:  bool = True
    cls_backbone_grad: bool = True

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

    def state_dicts(self) -> list[dict]:
        return [snap.state_dict() for snap in self._pool]

    def load_state_dicts(self, agent: CynthAIAgent, dicts: list[dict]) -> None:
        self._pool.clear()
        for sd in dicts:
            snap = copy.deepcopy(agent)
            snap.load_state_dict(sd)
            snap.eval()
            for p in snap.parameters():
                p.requires_grad_(False)
            self._pool.append(snap)


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

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.ema.load_state_dict(sd)


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

    # ── Resume: load checkpoint config BEFORE constructing agent ─────────────
    # Architecture fields must come from the saved config so the agent matches
    # the checkpoint weights.  Training hyperparams (lr, schedules, etc.) come
    # from the launcher config so they can be changed on resume.
    _ARCH_FIELDS = {
        "use_independent_critic", "critic_n_layers", "critic_value_bound",
        "use_victory_head", "cls_value_grad", "cls_victory_grad", "cls_backbone_grad",
        "critic_action_aware", "critic_n_cross_layers", "critic_mask_actions",
    }

    start_update = 1
    ckpt = None
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=True)
        start_update = ckpt["update"] + 1
        if "config" in ckpt:
            saved = dict(ckpt["config"])
            # Override architecture fields from checkpoint
            current = asdict(cfg)
            for key in _ARCH_FIELDS:
                if key in saved:
                    current[key] = saved[key]
            current["resume"] = cfg.resume
            cfg = TrainingConfig(**current)
            print(f"Architecture config restored from checkpoint:")
            print(f"  {', '.join(f'{k}={getattr(cfg, k)}' for k in sorted(_ARCH_FIELDS))}")

    # W&B init
    if _WANDB_AVAILABLE:
        wandb.init(
            project = "cynthai",
            name    = cfg.run_name,
            config  = asdict(cfg),
            resume  = "allow",
            id      = cfg.run_name,
        )

    # CSV loggers — don't re-write headers on resume
    metrics_path = Path(cfg.checkpoint_dir) / "metrics.csv"
    eval_path    = Path(cfg.checkpoint_dir) / "eval.csv"
    _first_metrics = not metrics_path.exists()
    _first_eval    = not eval_path.exists()

    agent = CynthAIAgent(
        use_independent_critic = cfg.use_independent_critic,
        critic_n_layers        = cfg.critic_n_layers,
        critic_value_bound     = cfg.critic_value_bound,
        use_victory_head       = cfg.use_victory_head,
        cls_value_grad         = cfg.cls_value_grad,
        cls_victory_grad       = cfg.cls_victory_grad,
        cls_backbone_grad      = cfg.cls_backbone_grad,
        critic_action_aware    = cfg.critic_action_aware,
        critic_n_cross_layers  = cfg.critic_n_cross_layers,
        critic_mask_actions    = cfg.critic_mask_actions,
        critic_from_backbone   = cfg.critic_from_backbone,
    ).to(device)

    if cfg.use_independent_critic:
        critic_params = list(agent.independent_critic.parameters())
        critic_ids    = {id(p) for p in critic_params}
        actor_params  = [p for p in agent.parameters() if id(p) not in critic_ids]
        optimizer = torch.optim.AdamW([
            {"params": actor_params,  "lr": cfg.lr,        "weight_decay": cfg.weight_decay},
            {"params": critic_params, "lr": cfg.critic_lr, "weight_decay": cfg.critic_wd},
        ], eps=1e-5)
    else:
        optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr           = cfg.lr,
            weight_decay = cfg.weight_decay,
            eps          = 1e-5,
        )

    if ckpt is not None:
        missing, unexpected = agent.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params (random init): {missing}")
        if unexpected:
            print(f"  Dropped params: {unexpected}")
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except ValueError:
            print("  Optimizer state incompatible (architecture change) — starting optimizer fresh.")
        print(f"Resumed from {cfg.resume}  (update {ckpt['update']})")
        print(f"  run_name={cfg.run_name}  total_updates={cfg.total_updates}")

    pool         = OpponentPool(pool_size=cfg.pool_size)
    ema_opponent = EMAOpponent(agent, decay=cfg.ema_decay)  # P10b

    # Restore pool and EMA from pool_ema_latest.pt (separate lightweight file)
    if cfg.resume:
        pool_ema_path = Path(cfg.resume).parent / "pool_ema_latest.pt"
        if pool_ema_path.exists():
            pe = torch.load(pool_ema_path, map_location=device, weights_only=True)
            try:
                ema_opponent.load_state_dict(pe["ema"])
                print(f"  EMA opponent restored (from update {pe.get('update', '?')}).")
            except Exception as e:
                print(f"  EMA restore failed ({e}) — starting fresh EMA.")
            try:
                pool.load_state_dicts(agent, pe["pool"])
                print(f"  Opponent pool restored: {len(pool)} snapshots.")
            except Exception as e:
                print(f"  Pool restore failed ({e}) — starting with empty pool.")
        else:
            print("  No pool_ema_latest.pt found — starting with empty pool and fresh EMA.")
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
        # Mixing: 0% Random / 25% FullOffense / 30% EMA / 45% pool
        #         Pool fallback: EMA (plus stable que FullOffense une fois l'EMA chaude)
        roll = random.random()
        if roll < 0.25:
            opponent = FullOffensePolicy()
            opp_label = "fo"
        elif roll < 0.55:
            # 30% — EMA opponent (stable lagging self-play)
            if update > cfg.ema_warmup:
                opponent = ema_opponent.sample(agent)
                opp_label = "ema"
            else:
                opponent = agent
                opp_label = "self"
        else:
            # 45% — pool (diversity) or EMA fallback if pool empty
            if len(pool) > 0:
                opponent = pool.sample(agent)
                opp_label = f"pool({len(pool)})"
            else:
                opponent = ema_opponent.sample(agent) if update > cfg.ema_warmup else agent
                opp_label = "ema"

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
        nan_steps = 0

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

                returns_in = batch["returns"]
                if cfg.value_target_clip > 0:
                    returns_in = returns_in.clamp(-cfg.value_target_clip, cfg.value_target_clip)

                losses = compute_losses(
                    logits_new   = out.action_logits,
                    log_prob_old = batch["log_prob_old"],
                    actions      = batch["actions"],
                    advantages   = batch["advantages"],
                    returns      = returns_in,
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

                if cfg.value_dump_threshold > 0:
                    vp_max_batch = losses.get("vp_max", 0)
                    if isinstance(vp_max_batch, torch.Tensor):
                        vp_max_batch = vp_max_batch.item()
                    if abs(vp_max_batch) > cfg.value_dump_threshold:
                        print(f"[WARN] vp_max={vp_max_batch:.1f} exceeds dump threshold={cfg.value_dump_threshold:.1f}", flush=True)

                # Victory head auxiliary loss (BCE on episode outcome)
                if cfg.use_victory_head and out.win_logit is not None:
                    win_labels  = batch["win_labels"]
                    known_mask  = win_labels != 0.5
                    if known_mask.any():
                        victory_loss = F.binary_cross_entropy_with_logits(
                            out.win_logit.squeeze(-1)[known_mask],
                            win_labels[known_mask],
                        )
                        with torch.no_grad():
                            win_prob    = torch.sigmoid(out.win_logit.squeeze(-1)[known_mask])
                            victory_acc = ((win_prob > 0.5) == win_labels[known_mask].bool()).float().mean()
                        losses["victory"]     = victory_loss.detach()
                        losses["victory_acc"] = victory_acc
                        losses["total"]       = losses["total"] + cfg.c_victory * victory_loss

                # Log scaled (weighted) loss components for diagnostics
                with torch.no_grad():
                    losses["scaled_loss/policy"]       = losses["policy"]
                    losses["scaled_loss/value"]        = cfg.c_value * losses["value"]
                    losses["scaled_loss/entropy"]      = cfg.c_entropy * losses["entropy"]
                    losses["scaled_loss/pred"]         = cfg.c_pred * pred_loss.detach()
                    losses["scaled_loss/attn_entropy"] = -cfg.c_attn_entropy * attn_entropy_val.detach()
                    losses["scaled_loss/rank_reg"]     = rank_loss.detach()
                    if "victory" in losses:
                        losses["scaled_loss/victory"]  = cfg.c_victory * losses["victory"]

                if torch.isnan(losses["total"]):
                    optimizer.zero_grad()
                    print(f"[WARN] NaN loss at update {update}, epoch {_epoch} -- skipping step", flush=True)
                    nan_steps += 1
                    continue
                losses["total"].backward()
                if cfg.use_independent_critic:
                    critic_params = list(agent.independent_critic.parameters())
                    critic_ids    = {id(p) for p in critic_params}
                    actor_params  = [p for p in agent.parameters() if id(p) not in critic_ids]
                    actor_grad_norm  = nn.utils.clip_grad_norm_(actor_params,  cfg.max_grad_norm)
                    critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, cfg.critic_grad_norm)
                    grad_norm = actor_grad_norm  # backward compat: "grad_norm" = actor
                else:
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    critic_grad_norm = None
                optimizer.step()

                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        loss_acc[k] += v.item()
                    else:
                        loss_acc[k] += v
                loss_acc["grad_norm"] += grad_norm.item()
                if critic_grad_norm is not None:
                    loss_acc["critic_grad_norm"] = loss_acc.get("critic_grad_norm", 0) + critic_grad_norm.item()
                for k, v in acc.items():
                    loss_acc[k] += v
                # Critic cross-attention diagnostics
                if out.critic_action_attn_entropy is not None:
                    loss_acc["critic_action_attn_entropy"] = (
                        loss_acc.get("critic_action_attn_entropy", 0)
                        + out.critic_action_attn_entropy.item()
                    )
                    loss_acc["critic_action_attn_max"] = (
                        loss_acc.get("critic_action_attn_max", 0)
                        + out.critic_action_attn_max.item()
                    )
                n_steps += 1

        # -- 2b. P10b: EMA weight update -----------------------------------------------
        ema_opponent.update(agent)

        # -- 3. LR schedule (warmup + cosine) -----------------------------------------
        # P6: manual LR scheduling for clean warmup -> cosine decay
        if update <= cfg.warmup_steps:
            actor_lr = cfg.lr * update / cfg.warmup_steps
            critic_lr_now = cfg.critic_lr * update / cfg.warmup_steps
        else:
            progress = (update - cfg.warmup_steps) / (cfg.total_updates - cfg.warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            actor_lr = max(cfg.lr_min, cfg.lr * cosine)
            critic_lr_now = max(cfg.lr_min, cfg.critic_lr * cosine)

        if cfg.use_independent_critic:
            optimizer.param_groups[0]["lr"] = actor_lr   # actor
            optimizer.param_groups[1]["lr"] = critic_lr_now  # critic
            lr = actor_lr  # for logging
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = actor_lr
            lr = actor_lr

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
                    "critic_lr": critic_lr_now if cfg.use_independent_critic else lr,
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
                    "victory": loss_acc.get("victory", 0) / ns,
                    "victory_acc": loss_acc.get("victory_acc", 0) / ns,
                    "critic_grad_norm": loss_acc.get("critic_grad_norm", 0) / ns,
                    "vp_mean": loss_acc.get("vp_mean", 0) / ns,
                    "vp_std": loss_acc.get("vp_std", 0) / ns,
                    "vp_min": loss_acc.get("vp_min", 0) / ns,
                    "vp_max": loss_acc.get("vp_max", 0) / ns,
                    "abs_err_max": loss_acc.get("abs_err_max", 0) / ns,
                    "corr_v_ret": loss_acc.get("corr_v_ret", 0) / ns,
                    "opp": opp_label,
                }
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if _first_metrics:
                    w.writeheader()
                    _first_metrics = False
                w.writerow(row)

            if _WANDB_AVAILABLE:
                wandb.log({
                    "train/win_rate":          win_rate,
                    "train/update_time_s":     elapsed,
                    "train/episodes":          total_eps,
                    "train/pool_size":         len(pool),
                    "loss/policy":             loss_acc["policy"] / ns,
                    "loss/value":              loss_acc["value"] / ns,
                    "loss/entropy":            loss_acc["entropy"] / ns,
                    "loss/pred":               loss_acc["pred"] / ns,
                    "loss/total":              loss_acc["total"] / ns,
                    "optim/lr":                lr,
                    "optim/critic_lr":         critic_lr_now if cfg.use_independent_critic else lr,
                    "optim/grad_norm":         loss_acc["grad_norm"] / ns,
                    "optim/clip_frac":         loss_acc["clip_frac"] / ns,
                    "optim/explained_variance": loss_acc["explained_variance"] / ns,
                    "optim/adv_mean":          loss_acc.get("adv_mean", 0) / ns,
                    "optim/adv_std":           loss_acc.get("adv_std", 0) / ns,
                    "pred/item_acc":           loss_acc.get("item_acc", 0) / ns,
                    "pred/ability_acc":        loss_acc.get("ability_acc", 0) / ns,
                    "pred/tera_acc":           loss_acc.get("tera_acc", 0) / ns,
                    "pred/move_recall":        loss_acc.get("move_recall", 0) / ns,
                    "attn/entropy":            loss_acc.get("attn_entropy", 0) / ns,
                    "attn/rank":               loss_acc.get("attn_rank", 0) / ns,
                    "schedule/mask_ratio":     mask_ratio,
                    "schedule/dense_scale":    dense_scale,
                    "critic/victory_loss":     loss_acc.get("victory", 0) / ns,
                    "critic/victory_acc":      loss_acc.get("victory_acc", 0) / ns,
                    "critic/grad_norm":        loss_acc.get("critic_grad_norm", 0) / ns,
                    "critic/vp_mean":          loss_acc.get("vp_mean", 0) / ns,
                    "critic/vp_std":           loss_acc.get("vp_std", 0) / ns,
                    "critic/vp_min":           loss_acc.get("vp_min", 0) / ns,
                    "critic/vp_max":           loss_acc.get("vp_max", 0) / ns,
                    "critic/ret_mean":         loss_acc.get("ret_mean", 0) / ns,
                    "critic/ret_std":          loss_acc.get("ret_std", 0) / ns,
                    "critic/abs_err_max":      loss_acc.get("abs_err_max", 0) / ns,
                    "critic/corr_v_ret":       loss_acc.get("corr_v_ret", 0) / ns,
                    "critic/action_attn_entropy": loss_acc.get("critic_action_attn_entropy", 0) / ns,
                    "critic/action_attn_max":     loss_acc.get("critic_action_attn_max", 0) / ns,
                    "scaled_loss/policy":         loss_acc.get("scaled_loss/policy", 0) / ns,
                    "scaled_loss/value":          loss_acc.get("scaled_loss/value", 0) / ns,
                    "scaled_loss/entropy":        loss_acc.get("scaled_loss/entropy", 0) / ns,
                    "scaled_loss/pred":           loss_acc.get("scaled_loss/pred", 0) / ns,
                    "scaled_loss/attn_entropy":   loss_acc.get("scaled_loss/attn_entropy", 0) / ns,
                    "scaled_loss/rank_reg":       loss_acc.get("scaled_loss/rank_reg", 0) / ns,
                    "scaled_loss/victory":        loss_acc.get("scaled_loss/victory", 0) / ns,
                }, step=update)

        # -- 5. Periodic evaluation ---------------------------------------------------
        # P3: every eval_freq updates, run 500 games vs each opponent type
        if update % cfg.eval_freq == 0 or update == 100:
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

            if _WANDB_AVAILABLE:
                eval_log = {}
                for label in ("random", "fulloff", "ema", "pool"):
                    r = eval_results[label]
                    eval_log[f"eval/{label}_wr"] = r["win_rate"]
                    for comp, val in r.get("reward_decomp_avg", {}).items():
                        eval_log[f"reward/{label}/{comp}"] = val
                    if "corr_v_win" in r:
                        eval_log[f"critic/{label}_corr_v_win"] = r["corr_v_win"]
                        eval_log[f"critic/{label}_auc_v_win"]  = r["auc_v_win"]
                wandb.log(eval_log, step=update)

            # Save evaluation diagnostic plots + attention maps, then push all to wandb
            try:
                save_eval_plots(eval_results, cfg.checkpoint_dir,
                                tag=f"update{update:06d}")
                save_attention_maps(agent, cfg.checkpoint_dir,
                                    tag=f"update{update:06d}", device=device)
                if _WANDB_AVAILABLE:
                    plot_images = {}
                    for png in sorted(Path(cfg.checkpoint_dir).rglob(f"*update{update:06d}*.png")):
                        key = f"plots/{png.parent.name}/{png.stem}"
                        plot_images[key] = wandb.Image(str(png))
                    if plot_images:
                        wandb.log(plot_images, step=update)
            except Exception as e:
                print(f"  WARNING: eval plots failed: {e}")

            # Probing analyses (every probe_freq evals)
            if cfg.probe_freq > 0 and (update // cfg.eval_freq) % cfg.probe_freq == 0:
                try:
                    from training.probing_eval import run_probing_eval
                    probe_metrics = run_probing_eval(
                        agent, device, cfg, update,
                        probe_min_steps=cfg.probe_min_steps,
                    )
                    if _WANDB_AVAILABLE:
                        # Images
                        probe_dir = Path(cfg.checkpoint_dir) / "probes" / f"update{update:06d}"
                        probe_images = {}
                        for png in sorted(probe_dir.glob("*.png")):
                            probe_images[f"probes/{png.stem}"] = wandb.Image(str(png))
                        if probe_images:
                            wandb.log(probe_images, step=update)
                        # Scalars
                        if probe_metrics:
                            wandb.log(probe_metrics, step=update)
                except Exception as e:
                    print(f"  WARNING: probing failed: {e}")

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
        if update % cfg.checkpoint_freq == 0 or update == 100:
            ckpt_path = Path(cfg.checkpoint_dir) / f"agent_{update:06d}.pt"
            torch.save({
                "update":    update,
                "config":    asdict(cfg),
                "model":     agent.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  -> checkpoint saved: {ckpt_path}")

        # -- 7b. Pool + EMA — overwrite a single file every update --------------------
        pool_ema_path = Path(cfg.checkpoint_dir) / "pool_ema_latest.pt"
        torch.save({
            "update": update,
            "ema":    ema_opponent.state_dict(),
            "pool":   pool.state_dicts(),
        }, pool_ema_path)

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