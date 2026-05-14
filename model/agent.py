"""
CynthAI_v2 Agent — Full forward pass wrapper.

Combines PokemonEmbeddings + BattleBackbone + ActionEncoder + PredictionHeads
into a single nn.Module with a clean interface for both rollout and training.

Key design: single Transformer pass.
  1. poke_emb(poke_batch)                          → pokemon_tokens [B, K*12, TOKEN_DIM]
  2. backbone.encode(pokemon_tokens, field_tensor) → current_tokens [B, 13, D_MODEL], value [B, 1]
  3. action_enc(current_tokens[:, 0], ...)         → action_embeds  [B, 13, D_MODEL]
  4. backbone.act(action_embeds, current_tokens)   → (action_logits, attn_entropy)
  5. predictor(current_tokens[:, 6:12])            → pred_logits

Inputs:
  poke_batch   : PokemonBatch [B, K*12]  — K turns × 12 Pokémon, already collated
  field_tensor : Tensor       [B, K, FIELD_DIM]
  move_idx     : Tensor       [B, 4]  int64  — active Pokémon's moves
  pp_ratio     : Tensor       [B, 4]  float32
  move_disabled: Tensor       [B, 4]  float32
  mechanic_id  : Tensor       [B]     int64  — MECH_* constant
  mechanic_type_idx : Tensor  [B]     int64
  action_mask  : Tensor       [B, 13] bool   — True = illegal
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import PokemonEmbeddings, PokemonBatch
from model.backbone import BattleBackbone, D_MODEL, OPP_SLOTS
from model.critic import IndependentCritic
from env.action_space import ActionEncoder
from model.prediction_heads import PredictionHeads, PredictionLogits


@dataclass
class AgentOutput:
    current_tokens: torch.Tensor     # [B, 13, D_MODEL]
    value:          torch.Tensor     # [B, 1]
    action_logits:  torch.Tensor     # [B, 13]  masked
    log_probs:      torch.Tensor     # [B, 13]  log_softmax
    pred_logits:    PredictionLogits
    attn_entropy:   torch.Tensor     # scalar — cross-attention entropy (P14)
    attn_rank:      torch.Tensor     # scalar — cross-attention rank (P18)


class CynthAIAgent(nn.Module):
    """
    Full CynthAI agent.

    All four sub-modules are owned here. The ActionEncoder shares
    move_embed and type_embed with PokemonEmbeddings (same weight objects).
    """

    def __init__(
        self,
        use_independent_critic: bool = False,
        critic_n_layers:        int  = 2,
    ):
        super().__init__()
        self.poke_emb   = PokemonEmbeddings()
        self.backbone   = BattleBackbone()
        self.action_enc = ActionEncoder(
            move_embed = self.poke_emb.move_embed,
            type_embed = self.poke_emb.type_embed,
        )
        self.predictor  = PredictionHeads()

        self.use_independent_critic = use_independent_critic
        if use_independent_critic:
            self.independent_critic = IndependentCritic(n_layers=critic_n_layers)

    def forward(
        self,
        poke_batch:        PokemonBatch,   # [B, K*12]
        field_tensor:      torch.Tensor,   # [B, K, FIELD_DIM]
        move_idx:          torch.Tensor,   # [B, 4]  int64
        pp_ratio:          torch.Tensor,   # [B, 4]  float32
        move_disabled:     torch.Tensor,   # [B, 4]  float32
        mechanic_id:       torch.Tensor,   # [B]     int64
        mechanic_type_idx: torch.Tensor,   # [B]     int64
        action_mask:       torch.Tensor,   # [B, 13] bool
    ) -> AgentOutput:
        # ── 1. Embed Pokémon tokens ───────────────────────────────────────────
        pokemon_tokens = self.poke_emb(poke_batch)          # [B, K*12, TOKEN_DIM]

        # ── 2. Single Transformer pass ────────────────────────────────────────
        pre_tokens, post_tokens, backbone_value = self.backbone.encode(pokemon_tokens, field_tensor)

        if self.use_independent_critic:
            value = self.independent_critic(pokemon_tokens, field_tensor)
        else:
            value = backbone_value

        # ── 3. Action embeddings (from PRE-transformer tokens — no self-match) ─
        action_embeds = self.action_enc(
            active_token      = pre_tokens[:, 0, :],
            move_idx          = move_idx,
            pp_ratio          = pp_ratio,
            move_disabled     = move_disabled,
            bench_tokens      = pre_tokens[:, 1:6, :],
            mechanic_id       = mechanic_id,
            mechanic_type_idx = mechanic_type_idx,
        )                                                    # [B, 13, D_MODEL]

        # ── 4. Actor head (keys = POST-transformer, enriched context) ──────────
        action_logits, attn_entropy, attn_rank = self.backbone.act(action_embeds, post_tokens, action_mask)
        log_probs     = F.log_softmax(action_logits, dim=-1)

        # ── 5. Predictor on opponent tokens (post-transformer) ─────────────────
        pred_logits = self.predictor(post_tokens[:, OPP_SLOTS, :])

        return AgentOutput(
            current_tokens = post_tokens,
            value          = value,
            action_logits  = action_logits,
            log_probs      = log_probs,
            pred_logits    = pred_logits,
            attn_entropy   = attn_entropy,
            attn_rank      = attn_rank,
        )
