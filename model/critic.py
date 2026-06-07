"""
CynthAI_v2 — ValueHead module.

Unified value head on top of backbone features.

When n_layers=0: MLP directly on backbone CLS token.
When n_layers>0: prepend own CLS, run N Transformer layers on full_seq, then MLP.
"""

import torch
import torch.nn as nn

from model.backbone import D_MODEL, N_HEADS, FFN_DIM, DROPOUT


class ValueHead(nn.Module):
    """
    Value head on top of backbone features.

    When n_layers=0: MLP directly on backbone CLS token (like the old backbone value head).
    When n_layers>0: prepend own CLS, run N Transformer layers on full_seq, then MLP.
    """

    def __init__(
        self,
        n_layers:         int   = 0,
        value_bound:      float = 0.0,
        use_victory_head: bool  = False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.value_bound = value_bound
        self.use_victory_head = use_victory_head

        if n_layers > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
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

        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )

        if use_victory_head:
            self.victory_head = nn.Linear(D_MODEL, 1)

    def forward(
        self,
        full_seq:     torch.Tensor,            # [B, 52, D_MODEL]
        padding_mask: torch.Tensor,            # [B, 52] bool — True = padded
        cls_out:      torch.Tensor | None = None,  # [B, D_MODEL] — backbone CLS (used when n_layers=0)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            v:         [B, 1]
            win_logit: [B, 1] or None
        """
        if self.n_layers > 0:
            B = full_seq.shape[0]
            dev = full_seq.device
            cls = self.cls_token.expand(B, 1, D_MODEL)
            seq = torch.cat([cls, full_seq], dim=1)              # [B, 53, D_MODEL]
            mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=dev),
                padding_mask,
            ], dim=1)                                             # [B, 53]
            seq = self.transformer(seq, src_key_padding_mask=mask)
            cls_out = seq[:, 0, :]                                # [B, D_MODEL]

        v = self.value_head(cls_out)                              # [B, 1]
        if self.value_bound > 0:
            v = self.value_bound * torch.tanh(v / self.value_bound)

        win_logit = self.victory_head(cls_out) if self.use_victory_head else None
        return v, win_logit
