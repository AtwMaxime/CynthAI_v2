"""
CynthAI_v2 Prediction Heads — Auxiliary supervised heads for opponent hidden state.

Takes the enriched opponent tokens from the backbone and predicts:
  item    : held item            (N_ITEMS     = 250 classes)
  ability : ability              (N_ABILITIES = 311 classes)
  tera    : Tera type            (N_TYPES     =  19 classes)
  moves   : all 4 move slots     (N_MOVES     = 686 classes each)

Training signal: cross-entropy, masked to slots where the truth is known.
Loss is separate from the PPO loss — these heads do not influence the policy
gradient directly, only the backbone representations via auxiliary supervision.

Move head design: Linear(D_MODEL, N_MOVES * 4) predicts all 4 slots jointly
from one Pokémon token. Different subspaces of D_MODEL serve different slots.
Shared weights are correct here: move slot order in PS is arbitrary per player.

Inference (v2 belief injection): see BELIEF_STATE.md.
  logits → argmax + softmax top-1 confidence → stored in belief_buffer (detached)
  → reinjected into state_encoder at next turn as item_idx + item_confidence scalar

Inputs:
  opp_tokens : [B, 6, D_MODEL]  — current_tokens[:, 6:12, :] from backbone

build_targets() expects the FULL opponent PokemonBatch (simulator ground truth,
not the agent-visible masked version). A slot is "revealed" when its index != UNK.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from model.backbone import D_MODEL
from env.state_encoder import N_ITEMS, N_ABILITIES, N_TYPES, N_MOVES, UNK

N_OPP = 6   # opponent slots in current_tokens (indices 6-11)


@dataclass
class PredictionLogits:
    """Raw logits (pre-softmax) output by PredictionHeads.forward()."""
    item:    torch.Tensor   # [B, 6, N_ITEMS]
    ability: torch.Tensor   # [B, 6, N_ABILITIES]
    tera:    torch.Tensor   # [B, 6, N_TYPES]
    moves:   torch.Tensor   # [B, 6, N_MOVES]  multi-label logits (BCE, top-4 à l'inférence)
    stats:   torch.Tensor   # [B, 6, 6]  raw HP + atk, def, spa, spd, spe


class PredictionHeads(nn.Module):
    """
    Four independent Linear heads on the opponent Pokémon tokens.

    Linear heads are enough — the Transformer tokens already encode rich
    contextual information. Deep MLPs would overfit on the sparse reveal signal.
    """

    def __init__(self):
        super().__init__()
        self.item_head    = nn.Linear(D_MODEL, N_ITEMS)
        self.ability_head = nn.Linear(D_MODEL, N_ABILITIES)
        self.tera_head    = nn.Linear(D_MODEL, N_TYPES)
        # Move head: BCE multi-label sur N_MOVES (top-4 recall à l'inférence).
        # Plus de bottleneck — le passage de 4 CE à 1 BCE réduit déjà la sortie
        # de 2744 à 686 dims.
        self.move_head = nn.Linear(D_MODEL, N_MOVES)
        self.stats_head = nn.Linear(D_MODEL, 6)   # P15c: raw HP + atk, def, spa, spd, spe

    def forward(self, opp_tokens: torch.Tensor) -> PredictionLogits:
        """
        Args:
            opp_tokens: [B, 6, D_MODEL]  (current_tokens[:, 6:12, :])
        Returns:
            PredictionLogits — raw logits, not probabilities.
        """
        B, P, _ = opp_tokens.shape
        item    = self.item_head(opp_tokens)                          # [B, 6, N_ITEMS]
        ability = self.ability_head(opp_tokens)                       # [B, 6, N_ABILITIES]
        tera    = self.tera_head(opp_tokens)                          # [B, 6, N_TYPES]
        moves   = self.move_head(opp_tokens)                          # [B, 6, N_MOVES]
        stats   = self.stats_head(opp_tokens)                         # [B, 6, 6]
        return PredictionLogits(item=item, ability=ability, tera=tera, moves=moves, stats=stats)

    @staticmethod
    def compute_accuracy(
        logits:          PredictionLogits,
        item_targets:    torch.Tensor,   # [B, 6]       int64
        ability_targets: torch.Tensor,   # [B, 6]       int64
        tera_targets:    torch.Tensor,   # [B, 6]       int64
        move_targets:    torch.Tensor,   # [B, 6, 686]  float32  (multi-label BCE)
        item_mask:       torch.Tensor,   # [B, 6]       bool
        ability_mask:    torch.Tensor,   # [B, 6]       bool
        tera_mask:       torch.Tensor,   # [B, 6]       bool
        move_mask:       torch.Tensor,   # [B, 6]       bool    (per-Pokémon)
        stats_targets:   torch.Tensor,   # [B, 6, 6]    float32
        stats_mask:      torch.Tensor,   # [B, 6]       bool
    ) -> dict[str, float]:
        """
        Per-head accuracy on revealed slots only.

        Returns dict: item_acc, ability_acc, tera_acc, move_recall, stats_mae.
        Returns 0.0 for heads with no revealed slots in the batch.
        """
        def _masked_acc(logits_nd, targets_nd, mask_nd):
            flat_logits  = logits_nd.reshape(-1, logits_nd.shape[-1])
            flat_targets = targets_nd.reshape(-1)
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return 0.0
            preds = flat_logits[flat_mask].argmax(dim=-1)
            return preds.eq(flat_targets[flat_mask]).float().mean().item()

        def _masked_top4_recall(logits_nd, targets_nd, mask_nd):
            """
            Top-4 recall for multi-label move prediction.
            logits_nd:  [B, 6, N_MOVES]   — sigmoid logits
            targets_nd: [B, 6, N_MOVES]   — binary (0/1)
            mask_nd:    [B, 6]            — per-Pokémon bool
            Returns fraction of true moves that appear in the top-4 predictions.
            """
            flat_logits  = logits_nd.reshape(-1, N_MOVES)
            flat_targets = targets_nd.reshape(-1, N_MOVES)
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return 0.0

            logits  = flat_logits[flat_mask]
            targets = flat_targets[flat_mask]
            top4    = logits.topk(4, dim=-1).indices                     # [N_valid, 4]
            hits    = torch.gather(targets > 0.5, 1, top4)               # [N_valid, 4] bool
            n_true  = targets.sum(dim=-1).float().clamp(min=1)           # [N_valid]
            n_hits  = hits.any(dim=-1).float()                           # [N_valid] — at least 1 hit in top-4
            return (n_hits / n_true).mean().item()

        def _masked_mae(logits_nd, targets_nd, mask_nd):
            flat_logits  = logits_nd.reshape(-1, logits_nd.shape[-1])
            flat_targets = targets_nd.reshape(-1, targets_nd.shape[-1])
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return 0.0
            return (flat_logits[flat_mask] - flat_targets[flat_mask]).abs().mean().item()

        return {
            "item_acc":    _masked_acc(logits.item,    item_targets,    item_mask),
            "ability_acc": _masked_acc(logits.ability, ability_targets, ability_mask),
            "tera_acc":    _masked_acc(logits.tera,    tera_targets,    tera_mask),
            "move_recall": _masked_top4_recall(logits.moves, move_targets, move_mask),
            "stats_mae":   _masked_mae(logits.stats,   stats_targets,   stats_mask),
        }
    @staticmethod
    def compute_loss(
        logits:          PredictionLogits,
        item_targets:    torch.Tensor,   # [B, 6]       int64
        ability_targets: torch.Tensor,   # [B, 6]       int64
        tera_targets:    torch.Tensor,   # [B, 6]       int64
        move_targets:    torch.Tensor,   # [B, 6, 686]  float32  (multi-label BCE)
        item_mask:       torch.Tensor,   # [B, 6]       bool
        ability_mask:    torch.Tensor,   # [B, 6]       bool
        tera_mask:       torch.Tensor,   # [B, 6]       bool
        move_mask:       torch.Tensor,   # [B, 6]       bool    (per-Pokémon)
        stats_targets:   torch.Tensor,   # [B, 6, 6]    float32
        stats_mask:      torch.Tensor,   # [B, 6]       bool
        c_stats:         float = 0.001,  # P17b: stats MSE scale (raw values ~1-700)
    ) -> dict[str, torch.Tensor]:
        """
        Masked cross-entropy per head + masked BCE for moves + masked MSE for stats.

        Slots with mask=False are excluded from the loss (unrevealed info).
        Returns zero (grad-connected) when no targets are revealed in the batch
        so the training loop can always call .backward() on the total.

        Returns dict keys: "item", "ability", "tera", "moves", "stats", "total".
        """
        def _masked_ce(logits_nd: torch.Tensor, targets_nd: torch.Tensor, mask_nd: torch.Tensor) -> torch.Tensor:
            flat_logits  = logits_nd.reshape(-1, logits_nd.shape[-1])
            flat_targets = targets_nd.reshape(-1)
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return flat_logits.sum() * 0.0
            return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])

        def _masked_bce(logits_nd: torch.Tensor, targets_nd: torch.Tensor, mask_nd: torch.Tensor) -> torch.Tensor:
            flat_logits  = logits_nd.reshape(-1, N_MOVES)
            flat_targets = targets_nd.reshape(-1, N_MOVES)
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return flat_logits.sum() * 0.0
            return F.binary_cross_entropy_with_logits(
                flat_logits[flat_mask], flat_targets[flat_mask]
            )

        def _masked_mse(logits_nd: torch.Tensor, targets_nd: torch.Tensor, mask_nd: torch.Tensor) -> torch.Tensor:
            flat_logits  = logits_nd.reshape(-1, logits_nd.shape[-1])
            flat_targets = targets_nd.reshape(-1, targets_nd.shape[-1])
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return flat_logits.sum() * 0.0
            return F.mse_loss(flat_logits[flat_mask], flat_targets[flat_mask])

        item_loss    = _masked_ce(logits.item,    item_targets,    item_mask)
        ability_loss = _masked_ce(logits.ability, ability_targets, ability_mask)
        tera_loss    = _masked_ce(logits.tera,    tera_targets,    tera_mask)
        move_loss    = _masked_bce(logits.moves,  move_targets,    move_mask)
        stats_loss   = _masked_mse(logits.stats,  stats_targets,   stats_mask)

        return {
            "item":    item_loss,
            "ability": ability_loss,
            "tera":    tera_loss,
            "moves":   move_loss,
            "stats":   stats_loss,
            "total":   item_loss + ability_loss + tera_loss + move_loss + c_stats * stats_loss,
        }

    @staticmethod
    def build_targets(full_opp_batch: "PokemonBatch") -> tuple[torch.Tensor, ...]:
        """
        Build targets and masks from the simulator ground-truth opponent batch.

        full_opp_batch must be the FULL state (all fields filled by the simulator),
        NOT the agent-visible masked batch (which has UNK for unrevealed slots).
        Sliced from the full PokemonBatch as batch[:, 6:12, :] before calling.

        A slot is "revealed" (mask=True) when its index != UNK (0).
        For moves: multi-label BCE target [B, 6, N_MOVES] with 1s at each known move.
        For stats: mask = species_idx != UNK (stats are tied to species identity).
        Calling convention in the training loop:
            full_opp = PokemonBatch sliced to opponent Pokémon from ground-truth state
            targets  = PredictionHeads.build_targets(full_opp)
            losses   = PredictionHeads.compute_loss(logits, *targets)

        Returns 10 tensors in order:
            item_targets    [B, 6]       int64
            ability_targets [B, 6]       int64
            tera_targets    [B, 6]       int64
            move_targets    [B, 6, 686]  float32   — binary multi-label
            item_mask       [B, 6]       bool
            ability_mask    [B, 6]       bool
            tera_mask       [B, 6]       bool
            move_mask       [B, 6]       bool      — species_idx != UNK
            stats_targets   [B, 6, 6]    float32   — hp + atk, def, spa, spd, spe
            stats_mask      [B, 6]       bool      — species_idx != UNK
        """
        B  = full_opp_batch.species_idx.shape[0]
        P  = 6
        it = full_opp_batch.item_idx      # [B, 6]
        ab = full_opp_batch.ability_idx   # [B, 6]
        te = full_opp_batch.tera_idx      # [B, 6]
        mv = full_opp_batch.move_idx      # [B, 6, 4]
        sp = full_opp_batch.species_idx   # [B, 6]

        # P15c: stats targets from scalars[222, 8:13] (hp + atk, def, spa, spd, spe)
        st = full_opp_batch.scalars[:, :, [222, 8, 9, 10, 11, 12]].contiguous()  # [B, 6, 6]

        # Multi-label move targets: scatter 1s at each known move position
        move_tg = torch.zeros(B, P, N_MOVES, dtype=torch.float32, device=mv.device)
        for s in range(4):
            slot = mv[:, :, s]  # [B, 6]
            valid = slot != UNK
            move_tg.scatter_add_(2, slot.clamp(min=0).unsqueeze(-1),
                                 valid.float().unsqueeze(-1))
        move_tg = move_tg.clamp(max=1.0)

        return (
            it, ab, te,
            move_tg,                              # move_targets [B, 6, 686]
            it != UNK,                             # item_mask
            ab != UNK,                             # ability_mask
            te != UNK,                             # tera_mask
            sp != UNK,                             # move_mask (species known → full moveset)
            st,                                    # stats_targets [B, 6, 6]
            sp != UNK,                             # stats_mask
        )
