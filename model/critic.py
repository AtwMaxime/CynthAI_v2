"""
CynthAI_v2 — Independent Critic Network.

A standalone Transformer that estimates V(s) with no shared weights with the actor.
Activated via TrainingConfig.use_independent_critic=True.

Same token layout and positional embeddings as BattleBackbone, but separate weights.
Uses attention pooling (same as backbone value head) over the 13 current-turn tokens.

Default: 2 Transformer layers (vs 3 for the actor backbone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import D_MODEL, N_HEADS, FFN_DIM, DROPOUT, K_TURNS, N_SLOTS, SEQ_LEN
from model.embeddings import TOKEN_DIM, FIELD_DIM


class IndependentCritic(nn.Module):
    """
    Standalone critic with its own Transformer — no shared weights with the actor.

    Takes the same raw token inputs as BattleBackbone (pokemon_tokens, field_tokens)
    and returns V(s) as [B, 1].
    """

    def __init__(self, n_layers: int = 2):
        super().__init__()

        self.pokemon_proj = nn.Linear(TOKEN_DIM, D_MODEL)
        self.field_proj   = nn.Linear(FIELD_DIM, D_MODEL)
        self.temporal_emb = nn.Embedding(K_TURNS, D_MODEL)
        self.slot_emb     = nn.Embedding(N_SLOTS, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = D_MODEL,
            nhead           = N_HEADS,
            dim_feedforward = FFN_DIM,
            dropout         = DROPOUT,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.pool_query = nn.Linear(D_MODEL, 1, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

    def forward(
        self,
        pokemon_tokens: torch.Tensor,   # [B, K*12, TOKEN_DIM]
        field_tokens:   torch.Tensor,   # [B, K,    FIELD_DIM]
    ) -> torch.Tensor:                  # [B, 1]
        B   = pokemon_tokens.shape[0]
        dev = pokemon_tokens.device

        p = self.pokemon_proj(pokemon_tokens)   # [B, K*12, D_MODEL]
        f = self.field_proj(field_tokens)        # [B, K,    D_MODEL]

        p = p.reshape(B, K_TURNS, 12, D_MODEL)
        f = f.unsqueeze(2)                       # [B, K, 1, D_MODEL]
        seq = torch.cat([p, f], dim=2)           # [B, K, 13, D_MODEL]

        t_ids = torch.arange(K_TURNS, device=dev)
        seq   = seq + self.temporal_emb(t_ids).unsqueeze(0).unsqueeze(2)

        s_ids = torch.arange(N_SLOTS, device=dev)
        seq   = seq + self.slot_emb(s_ids).unsqueeze(0).unsqueeze(0)

        seq = seq.reshape(B, SEQ_LEN, D_MODEL)   # [B, 52, D_MODEL]

        # P13b-style padding mask: turns where field features are all-zero are padding
        turn_norm    = field_tokens.abs().sum(dim=-1)           # [B, K]
        padding_mask = (turn_norm < 1e-6).repeat_interleave(13, dim=1)  # [B, 52]

        seq = self.transformer(seq, src_key_padding_mask=padding_mask)

        current = seq[:, -N_SLOTS:, :]                          # [B, 13, D_MODEL]
        scores  = self.pool_query(current).squeeze(-1)           # [B, 13]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)        # [B, 13, 1]
        pooled  = (weights * current).sum(dim=1)                 # [B, D_MODEL]
        return self.value_head(pooled)                           # [B, 1]
