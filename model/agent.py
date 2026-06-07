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
    current_tokens: torch.Tensor          # [B, 13, D_MODEL]
    value:          torch.Tensor          # [B, 1]
    action_logits:  torch.Tensor          # [B, 13]  masked
    log_probs:      torch.Tensor          # [B, 13]  log_softmax
    pred_logits:    PredictionLogits
    attn_entropy:   torch.Tensor          # scalar — cross-attention entropy (P14)
    attn_rank:      torch.Tensor          # scalar — cross-attention rank (P18)
    win_logit:      torch.Tensor | None = None  # [B, 1] or None — victory head output
    critic_action_attn_entropy: torch.Tensor | None = None  # scalar or None
    critic_action_attn_max:    torch.Tensor | None = None   # scalar or None


class CynthAIAgent(nn.Module):
    """
    Full CynthAI agent.

    All four sub-modules are owned here. The ActionEncoder shares
    move_embed and type_embed with PokemonEmbeddings (same weight objects).
    """

    def __init__(
        self,
        use_independent_critic: bool  = False,
        critic_n_layers:        int   = 2,
        critic_value_bound:     float = 0.0,
        use_victory_head:       bool  = False,
        cls_value_grad:         bool  = True,
        cls_victory_grad:       bool  = True,
        cls_backbone_grad:      bool  = True,
        critic_action_aware:    bool  = False,
        critic_n_cross_layers:  int   = 1,
        critic_mask_actions:    bool  = True,
        critic_from_backbone:   bool  = False,
    ):
        super().__init__()
        self.poke_emb   = PokemonEmbeddings()
        self.backbone   = BattleBackbone(
            use_victory_head  = use_victory_head,
            cls_value_grad    = cls_value_grad,
            cls_victory_grad  = cls_victory_grad,
            cls_backbone_grad = cls_backbone_grad,
        )
        self.action_enc = ActionEncoder(
            move_embed = self.poke_emb.move_embed,
            type_embed = self.poke_emb.type_embed,
        )
        self.predictor  = PredictionHeads()

        self.use_independent_critic = use_independent_critic
        self.critic_action_aware   = critic_action_aware
        self.critic_mask_actions   = critic_mask_actions
        self.critic_from_backbone  = critic_from_backbone
        if use_independent_critic:
            self.independent_critic = IndependentCritic(
                n_layers         = critic_n_layers,
                value_bound      = critic_value_bound,
                use_victory_head = use_victory_head,
                action_aware     = critic_action_aware,
                n_cross_layers   = critic_n_cross_layers,
                from_backbone    = critic_from_backbone,
            )

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
        if self.critic_from_backbone and self.use_independent_critic:
            pre_tokens, post_tokens, backbone_value, backbone_win_logit, full_seq, _cls = (
                self.backbone.encode(pokemon_tokens, field_tensor, return_full_seq=True)
            )
        else:
            pre_tokens, post_tokens, backbone_value, backbone_win_logit = self.backbone.encode(pokemon_tokens, field_tensor)
            full_seq = None

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

        # ── 3b. Critic (after action_embeds so action-aware critic can use them) ─
        critic_action_attn_entropy = None
        critic_action_attn_max = None
        if self.use_independent_critic:
            # Detach to prevent value loss gradient from contaminating poke_emb/action_enc
            value, win_logit = self.independent_critic(
                pokemon_tokens.detach(),
                field_tensor.detach(),
                action_embeds=action_embeds.detach() if self.critic_action_aware else None,
                action_mask=action_mask if self.critic_action_aware else None,
                mask_actions=self.critic_mask_actions,
                backbone_seq=full_seq.detach() if full_seq is not None else None,
            )
            # Extract cross-attention diagnostics if available
            attn_w = self.independent_critic._last_cross_attn_weights
            if attn_w is not None:
                # attn_w: [B, 1, 13] -> squeeze to [B, 13]
                w = attn_w.squeeze(1)
                # Shannon entropy: -sum(p * log(p))
                log_w = torch.log(w.clamp(min=1e-10))
                entropy = -(w * log_w).sum(dim=-1).mean()
                critic_action_attn_entropy = entropy
                critic_action_attn_max = w.max(dim=-1).values.mean()
        else:
            value     = backbone_value
            win_logit = backbone_win_logit

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
            win_logit      = win_logit,
            critic_action_attn_entropy = critic_action_attn_entropy,
            critic_action_attn_max     = critic_action_attn_max,
        )
