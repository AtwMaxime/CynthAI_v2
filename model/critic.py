"""
CynthAI_v2 — Independent Critic Network.

A standalone Transformer that estimates V(s) with no shared weights with the actor.
Activated via TrainingConfig.use_independent_critic=True.

Same token layout and positional embeddings as BattleBackbone, but separate weights.

Architecture:
  - A learnable [CLS] token is prepended to the 52-token sequence (K turns × 13 slots).
  - After the Transformer, the CLS output aggregates global game-state information.
  - Value head: maps CLS → V(s)  [B, 1]
  - Victory head (optional): maps CLS → P(win | s) logit  [B, 1]
    Trained with BCE against the episode outcome. Forces the CLS token to encode
    win-relevant information, which bootstraps the value head.

Default: 2 Transformer layers (vs 3 for the actor backbone).
"""

import torch
import torch.nn as nn

from model.backbone import D_MODEL, N_HEADS, FFN_DIM, DROPOUT, K_TURNS, N_SLOTS, SEQ_LEN
from model.embeddings import TOKEN_DIM, FIELD_DIM


class IndependentCritic(nn.Module):
    """
    Standalone critic with its own Transformer — no shared weights with the actor.

    Takes the same raw token inputs as BattleBackbone (pokemon_tokens, field_tokens)
    and returns (v, win_logit) where win_logit is None if use_victory_head=False.
    """

    def __init__(
        self,
        n_layers:         int   = 2,
        value_bound:      float = 0.0,
        use_victory_head: bool  = False,
        action_aware:     bool  = False,
        n_cross_layers:   int   = 1,
    ):
        super().__init__()

        # Switch A: when > 0, the raw head output is squashed to ±value_bound via
        # tanh, so no single state can emit an exploded value that poisons GAE targets.
        self.value_bound      = value_bound
        self.use_victory_head = use_victory_head
        self.action_aware     = action_aware

        # Learnable [CLS] token — prepended to the sequence, never masked.
        # After the Transformer it aggregates global game-state context.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))

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

        # Value head: applied to CLS token output only
        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

        # Cross-attention: CLS queries action embeddings (post-Transformer)
        if action_aware:
            self.cross_attn_layers = nn.ModuleList()
            for _ in range(n_cross_layers):
                self.cross_attn_layers.append(nn.ModuleList([
                    nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True),
                    nn.LayerNorm(D_MODEL),
                ]))

        # Victory head: predicts P(win | state) as a logit from the CLS token.
        # Trained with BCE against episode outcome; forces CLS to encode win-relevant info.
        if use_victory_head:
            self.victory_head = nn.Linear(D_MODEL, 1)

    def forward(
        self,
        pokemon_tokens: torch.Tensor,           # [B, K*12, TOKEN_DIM]
        field_tokens:   torch.Tensor,           # [B, K,    FIELD_DIM]
        action_embeds:  torch.Tensor | None = None,  # [B, 13, D_MODEL] — detached from actor
        action_mask:    torch.Tensor | None = None,  # [B, 13] bool, True=illegal
        mask_actions:   bool = True,
        return_repr:    bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        Returns:
            v, win_logit                         (default)
            v, win_logit, cls_out, seq_52        (return_repr=True)
        """
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

        # Padding mask for the 52 game tokens
        turn_norm    = field_tokens.abs().sum(dim=-1)               # [B, K]
        padding_52   = (turn_norm < 1e-6).repeat_interleave(13, dim=1)  # [B, 52]

        # Prepend [CLS] — position 0, never masked
        cls = self.cls_token.expand(B, 1, D_MODEL)                 # [B, 1, D_MODEL]
        seq = torch.cat([cls, seq], dim=1)                          # [B, 53, D_MODEL]
        padding_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=dev),
            padding_52,
        ], dim=1)                                                    # [B, 53]

        seq = self.transformer(seq, src_key_padding_mask=padding_mask)

        cls_out = seq[:, 0, :]    # [B, D_MODEL] — CLS token output
        seq_52  = seq[:, 1:, :]   # [B, 52, D_MODEL] — battle tokens

        # Cross-attention: CLS queries action embeddings
        self._last_cross_attn_weights = None
        if self.action_aware and action_embeds is not None:
            if mask_actions and action_mask is not None:
                # Ensure at least 1 valid key per sample to prevent softmax-over-empty → NaN
                kpm = action_mask.clone()
                all_masked = action_mask.all(dim=1)
                if all_masked.any():
                    kpm[all_masked, 0] = False
            else:
                kpm = None
            for cross_attn, cross_ln in self.cross_attn_layers:
                cls_q = cls_out.unsqueeze(1)             # [B, 1, D_MODEL]
                attn_out, attn_w = cross_attn(
                    query=cls_q,
                    key=action_embeds,
                    value=action_embeds,
                    key_padding_mask=kpm,
                    need_weights=True,
                    average_attn_weights=True,
                )
                cls_out = cross_ln(cls_q + attn_out).squeeze(1)  # [B, D_MODEL]
            # Store last layer's weights for diagnostics: [B, 1, 13]
            self._last_cross_attn_weights = attn_w.detach()

        v = self.value_head(cls_out)             # [B, 1]
        if self.value_bound > 0:
            v = self.value_bound * torch.tanh(v / self.value_bound)

        win_logit = self.victory_head(cls_out) if self.use_victory_head else None

        if return_repr:
            return v, win_logit, cls_out, seq_52
        return v, win_logit
