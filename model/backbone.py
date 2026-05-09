"""
CynthAI_v2 Backbone — Transformer over K turns of battle state.

Input:
  - PokemonBatch  : [B, K*12, TOKEN_DIM=438]  (K turns × 12 pokemon tokens)
  - FieldBatch    : [B, K, FIELD_DIM=72]       (K turns × 1 field token)
  - action_embeds : [B, 13, d_model]            (from action_space.py, already projected)
  - action_mask   : [B, 13] bool                (True = illegal action)

Token layout within each turn (13 tokens, slots 0-12):
  slots  0-5  : own Pokémon
  slots  6-11 : opponent Pokémon
  slot   12   : field token

Full sequence: K turns × 13 tokens = 52 tokens (with K=4).

Positional embeddings (additive, both learned):
  temporal_emb : Embedding(K,  d_model) — which turn (0=current, K-1=oldest)
  slot_emb     : Embedding(13, d_model) — which slot within a turn

Output:
  current_tokens : [B, 13, d_model]  — enriched tokens of the current turn
  value          : [B, 1]            — V(s) estimate
  action_logits  : [B, 13]           — policy scores (illegal actions masked)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import (
    PokemonEmbeddings, PokemonBatch,
    FieldBatch,
    TOKEN_DIM, FIELD_DIM,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
D_MODEL     = 256
N_HEADS     = 4
N_LAYERS    = 3
FFN_DIM     = 512
DROPOUT     = 0.1
K_TURNS     = 4       # sliding window size
N_SLOTS     = 13      # tokens per turn (12 pokemon + 1 field)
SEQ_LEN     = K_TURNS * N_SLOTS   # 52

# Slot indices
OWN_SLOTS   = slice(0, 6)    # own pokemon
OPP_SLOTS   = slice(6, 12)   # opponent pokemon
FIELD_SLOT  = 12             # field token


class BattleBackbone(nn.Module):
    """
    Transformer backbone for CynthAI_v2.

    Processes K turns of battle state and produces:
      - Enriched token representations for the current turn
      - V(s) via MLP on flattened current tokens (critic)
      - Action logits via cross-attention on action embeddings (actor)
    """

    def __init__(self):
        super().__init__()

        # ── Input projections (one per token type) ────────────────────────────
        self.pokemon_proj = nn.Linear(TOKEN_DIM, D_MODEL)
        self.field_proj   = nn.Linear(FIELD_DIM, D_MODEL)

        # ── Positional embeddings ─────────────────────────────────────────────
        self.temporal_emb = nn.Embedding(K_TURNS, D_MODEL)   # 0=oldest, K-1=current
        self.slot_emb     = nn.Embedding(N_SLOTS, D_MODEL)   # 0-11=pokemon, 12=field

        # ── Transformer ───────────────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = D_MODEL,
            nhead           = N_HEADS,
            dim_feedforward = FFN_DIM,
            dropout         = DROPOUT,
            batch_first     = True,
            norm_first      = True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS, enable_nested_tensor=False
        )

        # ── Value head (Critic) ───────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(N_SLOTS * D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

        # ── Policy head (Actor) ───────────────────────────────────────────────
        self.action_cross_attn = nn.MultiheadAttention(
            embed_dim   = D_MODEL,
            num_heads   = N_HEADS,
            dropout     = DROPOUT,
            batch_first = True,
        )
        self.action_score = nn.Linear(D_MODEL, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_sequence(
        self,
        pokemon_tokens: torch.Tensor,   # [B, K*12, TOKEN_DIM]
        field_tokens:   torch.Tensor,   # [B, K,    FIELD_DIM]
    ) -> torch.Tensor:
        """
        Project and interleave pokemon + field tokens into a full sequence
        of shape [B, K*13, D_MODEL], with temporal and slot embeddings added.

        Turn ordering: oldest turn first, current turn last.
        Within each turn: own(0-5) | opp(6-11) | field(12).
        """
        B   = pokemon_tokens.shape[0]
        dev = pokemon_tokens.device

        p = self.pokemon_proj(pokemon_tokens)   # [B, K*12, D_MODEL]
        f = self.field_proj(field_tokens)        # [B, K,    D_MODEL]

        p = p.reshape(B, K_TURNS, 12, D_MODEL)
        f = f.unsqueeze(2)                       # [B, K, 1, D_MODEL]

        seq = torch.cat([p, f], dim=2)           # [B, K, 13, D_MODEL]

        t_ids = torch.arange(K_TURNS, device=dev)
        t_emb = self.temporal_emb(t_ids)         # [K, D_MODEL]
        seq   = seq + t_emb.unsqueeze(0).unsqueeze(2)

        s_ids = torch.arange(N_SLOTS, device=dev)
        s_emb = self.slot_emb(s_ids)             # [13, D_MODEL]
        seq   = seq + s_emb.unsqueeze(0).unsqueeze(0)

        return seq.reshape(B, SEQ_LEN, D_MODEL)

    # ── Core passes (split for single-Transformer efficiency) ────────────────

    def encode(
        self,
        pokemon_tokens: torch.Tensor,   # [B, K*12, TOKEN_DIM]
        field_tokens:   torch.Tensor,   # [B, K,    FIELD_DIM]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the Transformer once and return enriched state tokens + value.

        Returns:
            current_tokens : [B, 13, D_MODEL]  — current turn tokens
            value          : [B, 1]             — V(s) estimate
        """
        seq = self._build_sequence(pokemon_tokens, field_tokens)   # [B, 52, D_MODEL]
        seq = self.transformer(seq)                                 # [B, 52, D_MODEL]

        current_tokens = seq[:, -N_SLOTS:, :]                      # [B, 13, D_MODEL]

        B    = current_tokens.shape[0]
        flat  = current_tokens.reshape(B, N_SLOTS * D_MODEL)
        value = self.value_head(flat)                               # [B, 1]

        return current_tokens, value

    def act(
        self,
        action_embeds:  torch.Tensor,   # [B, 13, D_MODEL]
        current_tokens: torch.Tensor,   # [B, 13, D_MODEL]
        action_mask:    torch.Tensor,   # [B, 13] bool — True = illegal
    ) -> torch.Tensor:                  # [B, 13]
        """
        Actor head only — cross-attention of action_embeds over current_tokens.
        Call after encode(); does NOT re-run the Transformer.
        """
        attn_out, _ = self.action_cross_attn(
            query = action_embeds,
            key   = current_tokens,
            value = current_tokens,
        )
        logits = self.action_score(attn_out).squeeze(-1)   # [B, 13]
        return logits.masked_fill(action_mask, -1e9)

    # ── Forward (convenience wrapper) ─────────────────────────────────────────

    def forward(
        self,
        pokemon_tokens: torch.Tensor,
        field_tokens:   torch.Tensor,
        action_embeds:  torch.Tensor,
        action_mask:    torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            current_tokens : [B, 13, D_MODEL]
            value          : [B, 1]
            action_logits  : [B, 13]  (illegal actions set to -1e9)
        """
        current_tokens, value = self.encode(pokemon_tokens, field_tokens)
        action_logits = self.act(action_embeds, current_tokens, action_mask)
        return current_tokens, value, action_logits
