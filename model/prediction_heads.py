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

N_OPP        = 6   # opponent slots in current_tokens (indices 6-11)
N_MOVE_SLOTS = 4


@dataclass
class PredictionLogits:
    """Raw logits (pre-softmax) output by PredictionHeads.forward()."""
    item:    torch.Tensor   # [B, 6, N_ITEMS]
    ability: torch.Tensor   # [B, 6, N_ABILITIES]
    tera:    torch.Tensor   # [B, 6, N_TYPES]
    moves:   torch.Tensor   # [B, 6, 4, N_MOVES]


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
        # Move head: factorisé avec bottleneck 128 pour régularisation (P13d).
        # Si la prédiction des moves échoue (loss plafonnée), augmenter 128 → 256.
        self.move_head = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.ReLU(),
            nn.Linear(128, N_MOVES * N_MOVE_SLOTS),
        )

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
        moves   = self.move_head(opp_tokens)                          # [B, 6, N_MOVES*4]
        moves   = moves.reshape(B, P, N_MOVE_SLOTS, N_MOVES)         # [B, 6, 4, N_MOVES]
        return PredictionLogits(item=item, ability=ability, tera=tera, moves=moves)

    @staticmethod
    def compute_loss(
        logits:          PredictionLogits,
        item_targets:    torch.Tensor,   # [B, 6]    int64
        ability_targets: torch.Tensor,   # [B, 6]    int64
        tera_targets:    torch.Tensor,   # [B, 6]    int64
        move_targets:    torch.Tensor,   # [B, 6, 4] int64
        item_mask:       torch.Tensor,   # [B, 6]    bool
        ability_mask:    torch.Tensor,   # [B, 6]    bool
        tera_mask:       torch.Tensor,   # [B, 6]    bool
        move_mask:       torch.Tensor,   # [B, 6, 4] bool
    ) -> dict[str, torch.Tensor]:
        """
        Masked cross-entropy per head.

        Slots with mask=False are excluded from the loss (unrevealed info).
        Returns zero (grad-connected) when no targets are revealed in the batch
        so the training loop can always call .backward() on the total.

        Returns dict keys: "item", "ability", "tera", "moves", "total".
        """
        def _masked_ce(logits_nd: torch.Tensor, targets_nd: torch.Tensor, mask_nd: torch.Tensor) -> torch.Tensor:
            flat_logits  = logits_nd.reshape(-1, logits_nd.shape[-1])
            flat_targets = targets_nd.reshape(-1)
            flat_mask    = mask_nd.reshape(-1)
            if not flat_mask.any():
                return flat_logits.sum() * 0.0   # zero scalar, keeps gradient graph
            return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])

        item_loss    = _masked_ce(logits.item,    item_targets,    item_mask)
        ability_loss = _masked_ce(logits.ability, ability_targets, ability_mask)
        tera_loss    = _masked_ce(logits.tera,    tera_targets,    tera_mask)
        move_loss    = _masked_ce(logits.moves,   move_targets,    move_mask)

        return {
            "item":    item_loss,
            "ability": ability_loss,
            "tera":    tera_loss,
            "moves":   move_loss,
            "total":   item_loss + ability_loss + tera_loss + move_loss,
        }

    @staticmethod
    def build_targets(full_opp_batch: "PokemonBatch") -> tuple[torch.Tensor, ...]:
        """
        Build targets and masks from the simulator ground-truth opponent batch.

        full_opp_batch must be the FULL state (all fields filled by the simulator),
        NOT the agent-visible masked batch (which has UNK for unrevealed slots).
        Sliced from the full PokemonBatch as batch[:, 6:12, :] before calling.

        A slot is "revealed" (mask=True) when its index != UNK (0).
        Calling convention in the training loop:
            full_opp = PokemonBatch sliced to opponent Pokémon from ground-truth state
            targets  = PredictionHeads.build_targets(full_opp)
            losses   = PredictionHeads.compute_loss(logits, *targets)

        Returns 8 tensors in order:
            item_targets    [B, 6]    int64
            ability_targets [B, 6]    int64
            tera_targets    [B, 6]    int64
            move_targets    [B, 6, 4] int64
            item_mask       [B, 6]    bool
            ability_mask    [B, 6]    bool
            tera_mask       [B, 6]    bool
            move_mask       [B, 6, 4] bool
        """
        it = full_opp_batch.item_idx      # [B, 6]
        ab = full_opp_batch.ability_idx   # [B, 6]
        te = full_opp_batch.tera_idx      # [B, 6]
        mv = full_opp_batch.move_idx      # [B, 6, 4]

        return (
            it, ab, te, mv,
            it != UNK,
            ab != UNK,
            te != UNK,
            mv != UNK,
        )
