"""
CynthAI_v2 Action Space — builds 13 action embeddings from live battle context.

Action slot layout (13 total):
  0-3  : regular moves (no mechanic)
  4-7  : mechanic moves (same moves with Tera/Mega/Z/Dmax modifier)
  8-12 : switch to bench Pokémon 0-4

ActionEncoder takes the active Pokémon's backbone token, 4 move indices + PP/disabled
scalars, 5 bench tokens, and a mechanic descriptor. It outputs [B, 13, D_MODEL].

Mechanic constants:
  MECH_NONE=0    no mechanic (mechanic action slots will be masked anyway)
  MECH_TERA=1    Terastallize
  MECH_MEGA=2    Mega Evolve
  MECH_ZMOVE=3   Z-Move
  MECH_DYNAMAX=4 Dynamax / G-Max
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import D_MODEL
from model.embeddings import D_MOVE, D_TYPE

MECH_NONE    = 0
MECH_TERA    = 1
MECH_MEGA    = 2
MECH_ZMOVE   = 3
MECH_DYNAMAX = 4
N_MECHANICS  = 5


class ActionEncoder(nn.Module):
    """
    Builds 13 action embeddings from live battle context.

    move_embed and type_embed are shared weight tensors owned by PokemonEmbeddings.
    They are passed in at construction time so the agent has a single copy of each.
    """

    def __init__(self, move_embed: nn.Embedding, type_embed: nn.Embedding):
        super().__init__()
        self.move_embed = move_embed
        self.type_embed = type_embed

        # D_MOVE + 2  (move emb + [pp_ratio, move_disabled])
        # active_token removed: it dominated queries and caused move identities to collapse (P22)
        self.move_proj     = nn.Linear(D_MOVE + 2, D_MODEL)
        self.mechanic_proj = nn.Linear(D_TYPE + N_MECHANICS, D_MODEL)
        self.switch_proj   = nn.Linear(D_MODEL, D_MODEL)

    def forward(
        self,
        active_token:       torch.Tensor,   # [B, D_MODEL]
        move_idx:           torch.Tensor,   # [B, 4]  int64
        pp_ratio:           torch.Tensor,   # [B, 4]  float32
        move_disabled:      torch.Tensor,   # [B, 4]  float32
        bench_tokens:       torch.Tensor,   # [B, 5, D_MODEL]
        mechanic_id:        torch.Tensor,   # [B]     int64
        mechanic_type_idx:  torch.Tensor,   # [B]     int64
    ) -> torch.Tensor:                      # [B, 13, D_MODEL]
        # ── Move embeddings ───────────────────────────────────────────────────
        mv_emb     = self.move_embed(move_idx)                        # [B, 4, D_MOVE]
        scalars    = torch.stack([pp_ratio, move_disabled], dim=-1)   # [B, 4, 2]
        move_input = torch.cat([mv_emb, scalars], dim=-1)             # [B, 4, D_MOVE+2]
        base_moves = self.move_proj(move_input)                       # [B, 4, D_MODEL]

        # ── Mechanic modifier ─────────────────────────────────────────────────
        type_emb   = self.type_embed(mechanic_type_idx)               # [B, D_TYPE]
        mech_oh    = F.one_hot(mechanic_id, N_MECHANICS).float()      # [B, N_MECHANICS]
        mech_input = torch.cat([type_emb, mech_oh], dim=-1)           # [B, D_TYPE+N_MECHANICS]
        mech_mod   = self.mechanic_proj(mech_input).unsqueeze(1)      # [B, 1, D_MODEL]
        mech_moves = base_moves + mech_mod                            # [B, 4, D_MODEL]

        # ── Switch actions ────────────────────────────────────────────────────
        switch_acts = self.switch_proj(bench_tokens)                  # [B, 5, D_MODEL]

        # ── Concatenate: [regular(4), mechanic(4), switch(5)] = 13 ────────────
        return torch.cat([base_moves, mech_moves, switch_acts], dim=1)  # [B, 13, D_MODEL]
