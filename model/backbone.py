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
        # Pooled (mean over 13 slots) → Linear → ReLU → Linear
        # Mean pooling avoids the 3328→256 bottleneck and adds ~65k params
        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
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

        # ── Attention map storage (filled by hooks) ──────────────────────────
        self._attention_maps: list[torch.Tensor] = []
        self._register_attention_hooks()

    # ── Attention hooks ───────────────────────────────────────────────────

    def _register_attention_hooks(self) -> None:
        """
        Save original _sa_block methods so get_attention_maps can temporarily
        monkey-patch them without affecting training forward passes.
        """
        self._attention_maps: list[torch.Tensor] = []
        self._orig_sa_blocks = [
            layer._sa_block for layer in self.transformer.layers
        ]

    def _temporarily_patch_sa(self, enable: bool) -> None:
        """
        Replace or restore _sa_block on all encoder layers.
        Call with enable=True before, enable=False after get_attention_maps.
        """
        import types as _types

        self._attention_maps.clear()
        for i, layer in enumerate(self.transformer.layers):
            if enable:
                orig = self._orig_sa_blocks[i]

                def patched_sa(  # noqa: N807
                    _self, x, attn_mask, key_padding_mask, is_causal=False
                ):
                    attn_out = _self.self_attn(
                        x, x, x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                        average_attn_weights=False,
                        is_causal=is_causal,
                    )
                    self._attention_maps.append(attn_out[1].detach().cpu())
                    return _self.dropout1(attn_out[0])

                layer._sa_block = _types.MethodType(patched_sa, layer)
            else:
                layer._sa_block = self._orig_sa_blocks[i]

    def get_attention_maps(
        self,
        pokemon_tokens: torch.Tensor,   # [B, K*12, TOKEN_DIM]
        field_tokens:   torch.Tensor,   # [B, K,    FIELD_DIM]
    ) -> dict:
        """
        Run a forward pass and return attention maps from all layers.

        Iterates transformer layers manually so we can call self_attn with
        need_weights=True, bypassing any MHA fastpath that would skip _sa_block.

        Returns:
            attention_maps : list[Tensor]  — one per layer, each [B, H, T, T]
            value          : Tensor        — [B, 1]
            current_tokens : Tensor        — [B, 13, D_MODEL]
            token_labels   : list[str]     — label for each of the 52 sequence positions
        """
        self._attention_maps.clear()

        seq = self._build_sequence(pokemon_tokens, field_tokens)   # [B, 52, D_MODEL]

        # P13b: padding mask
        turn_norm = field_tokens.abs().sum(dim=-1)
        padding_mask = (turn_norm < 1e-6).repeat_interleave(13, dim=1)  # [B, 52]

        # Manual layer-by-layer forward to capture attention weights.
        # This avoids fastpath/bypass issues with monkey-patching _sa_block.
        for layer in self.transformer.layers:
            if layer.norm_first:
                # Pre-LN: norm before each sub-layer
                attn_input = layer.norm1(seq)
            else:
                attn_input = seq

            attn_out = layer.self_attn(
                attn_input, attn_input, attn_input,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=padding_mask,
            )
            self._attention_maps.append(attn_out[1].detach().cpu())

            # Residual + dropout (same as _sa_block)
            seq = seq + layer.dropout1(attn_out[0])

            # Feed-forward block
            if layer.norm_first:
                seq = seq + layer._ff_block(layer.norm2(seq))
            else:
                seq = layer.norm2(seq + layer._ff_block(seq))

        current_tokens = seq[:, -N_SLOTS:, :]                      # [B, 13, D_MODEL]
        B    = current_tokens.shape[0]
        pooled = current_tokens.mean(dim=1)                         # [B, D_MODEL]
        value = self.value_head(pooled)                             # [B, 1]

        # Build token labels: T0_OWN0..T0_OWN5, T0_OPP0..T0_OPP5, T0_FIELD, ...
        token_labels = []
        for turn in range(K_TURNS):
            for slot in range(N_SLOTS):
                if slot < 6:
                    token_labels.append(f"T{turn}_OWN{slot}")
                elif slot < 12:
                    token_labels.append(f"T{turn}_OPP{slot - 6}")
                else:
                    token_labels.append(f"T{turn}_FIELD")

        return {
            "attention_maps": list(self._attention_maps),
            "value":          value.detach().cpu(),
            "current_tokens": current_tokens.detach().cpu(),
            "token_labels":   token_labels,
        }

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the Transformer once and return pre-transformer + post-transformer tokens + value.

        Pre-tokens (before self-attention) encode only raw projected features —
        used for action queries to avoid self-match shortcuts (P13e).

        Post-tokens (after self-attention) are enriched with global battle context —
        used as cross-attention keys/values and prediction heads.

        Returns:
            pre_tokens    : [B, 13, D_MODEL]  — current turn BEFORE Transformer
            post_tokens   : [B, 13, D_MODEL]  — current turn AFTER Transformer
            value         : [B, 1]             — V(s) estimate
        """
        seq = self._build_sequence(pokemon_tokens, field_tokens)   # [B, 52, D_MODEL]

        # Current turn tokens BEFORE Transformer (raw projected + pos emb, no self-attn)
        pre_tokens = seq[:, -N_SLOTS:, :]                          # [B, 13, D_MODEL]

        # P13b: padding mask — zero-filled turns have all-zero field features
        turn_norm = field_tokens.abs().sum(dim=-1)                   # [B, K]
        padding_mask = (turn_norm < 1e-6).repeat_interleave(13, dim=1)  # [B, 52]
        seq = self.transformer(seq, src_key_padding_mask=padding_mask)  # [B, 52, D_MODEL]

        # Current turn tokens AFTER Transformer (enriched by self-attention)
        post_tokens = seq[:, -N_SLOTS:, :]                         # [B, 13, D_MODEL]

        B      = post_tokens.shape[0]
        pooled = post_tokens.mean(dim=1)                            # [B, D_MODEL]
        value  = self.value_head(pooled)                            # [B, 1]

        return pre_tokens, post_tokens, value

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
        pre_tokens, post_tokens, value = self.encode(pokemon_tokens, field_tokens)
        action_logits = self.act(action_embeds, post_tokens, action_mask)
        return post_tokens, value, action_logits
