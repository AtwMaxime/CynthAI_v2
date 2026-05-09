"""
CynthAI_v2 Embeddings — categorical + scalar feature encoding for Pokémon tokens.

TOKEN_DIM = D_SPECIES + D_TYPE + D_TYPE + D_TYPE + D_ITEM + D_ABILITY + 4*D_MOVE + N_SCALARS
          = 32 + 8 + 8 + 8 + 16 + 16 + 128 + 222  =  438

PokemonBatch holds pre-collated integer tensors for embedding lookups + a float
scalar tensor. collate_features() assembles a list-of-lists of PokemonFeatures
into a PokemonBatch of shape [B, K*12, ...].

FieldBatch holds the flat [N, FIELD_DIM] field tensor; the caller reshapes to
[B, K, FIELD_DIM] after collation (see rollout._build_agent_inputs).
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from env.state_encoder import (
    PokemonFeatures,
    N_SPECIES, N_MOVES, N_ITEMS, N_ABILITIES,
    TYPE_EMBEDDING_PRIOR, MOVE_EMBEDDING_PRIOR,
    D_TYPE, N_VOLATILES,
    _species_data,
    FieldFeatures, SideFeatures,
    FIELD_DIM, SIDE_DIM, N_SIDE_CONDITIONS,
    N_WEATHER, N_TERRAIN, N_PSEUDO,
)

# ── Embedding dimensions ───────────────────────────────────────────────────────

D_SPECIES: int = 32
D_MOVE:    int = 32   # must match MOVE_EMBEDDING_PRIOR.shape[1]
D_ITEM:    int = 16
D_ABILITY: int = 16

N_SCALARS: int  = 222
TOKEN_DIM: int  = D_SPECIES + D_TYPE + D_TYPE + D_TYPE + D_ITEM + D_ABILITY + 4 * D_MOVE + N_SCALARS
# = 32 + 8 + 8 + 8 + 16 + 16 + 128 + 222 = 438


# ── PokemonBatch ───────────────────────────────────────────────────────────────

@dataclass
class PokemonBatch:
    """
    Pre-collated tensors for a batch of Pokémon tokens.
    All tensors should be on the same device.
      species_idx : [B, N]       int64
      type1_idx   : [B, N]       int64
      type2_idx   : [B, N]       int64
      tera_idx    : [B, N]       int64
      item_idx    : [B, N]       int64
      ability_idx : [B, N]       int64
      move_idx    : [B, N, 4]    int64
      scalars     : [B, N, 222]  float32
    """
    species_idx: torch.Tensor
    type1_idx:   torch.Tensor
    type2_idx:   torch.Tensor
    tera_idx:    torch.Tensor
    item_idx:    torch.Tensor
    ability_idx: torch.Tensor
    move_idx:    torch.Tensor
    scalars:     torch.Tensor

    def to(self, device) -> "PokemonBatch":
        return PokemonBatch(
            species_idx = self.species_idx.to(device),
            type1_idx   = self.type1_idx.to(device),
            type2_idx   = self.type2_idx.to(device),
            tera_idx    = self.tera_idx.to(device),
            item_idx    = self.item_idx.to(device),
            ability_idx = self.ability_idx.to(device),
            move_idx    = self.move_idx.to(device),
            scalars     = self.scalars.to(device),
        )


# ── collate_features ──────────────────────────────────────────────────────────

def collate_features(batch: list) -> PokemonBatch:
    """
    Collate a list of B items, each a flat list of K*12 PokemonFeatures,
    into a PokemonBatch of shape [B, K*12, ...].
    """
    B = len(batch)
    N = len(batch[0])

    species_np  = np.empty((B, N),    dtype=np.int64)
    type1_np    = np.empty((B, N),    dtype=np.int64)
    type2_np    = np.empty((B, N),    dtype=np.int64)
    tera_np     = np.empty((B, N),    dtype=np.int64)
    item_np     = np.empty((B, N),    dtype=np.int64)
    ability_np  = np.empty((B, N),    dtype=np.int64)
    move_np     = np.empty((B, N, 4), dtype=np.int64)
    scalars_np  = np.zeros((B, N, N_SCALARS), dtype=np.float32)

    for b, poke_list in enumerate(batch):
        for f, p in enumerate(poke_list):
            species_np[b, f]  = p.species_idx
            type1_np[b, f]    = p.type1_idx
            type2_np[b, f]    = p.type2_idx
            tera_np[b, f]     = p.tera_idx
            item_np[b, f]     = p.item_idx
            ability_np[b, f]  = p.ability_idx
            move_np[b, f]     = p.move_indices

            scalars_np[b, f, 0]      = p.level
            scalars_np[b, f, 1]      = p.hp_ratio
            scalars_np[b, f, 2]      = p.terastallized
            scalars_np[b, f, 3:8]    = p.base_stats
            scalars_np[b, f, 8:13]   = p.stats
            scalars_np[b, f, 13]     = p.is_predicted
            scalars_np[b, f, 14:21]  = p.boosts
            scalars_np[b, f, 21:25]  = p.move_pp
            scalars_np[b, f, 25:29]  = p.move_disabled
            scalars_np[b, f, 29:36]  = p.status
            scalars_np[b, f, 36:217] = p.volatiles
            scalars_np[b, f, 217]    = p.is_active
            scalars_np[b, f, 218]    = p.fainted
            scalars_np[b, f, 219]    = p.trapped
            scalars_np[b, f, 220]    = p.force_switch_flag
            scalars_np[b, f, 221]    = p.revealed

    return PokemonBatch(
        species_idx = torch.from_numpy(species_np),
        type1_idx   = torch.from_numpy(type1_np),
        type2_idx   = torch.from_numpy(type2_np),
        tera_idx    = torch.from_numpy(tera_np),
        item_idx    = torch.from_numpy(item_np),
        ability_idx = torch.from_numpy(ability_np),
        move_idx    = torch.from_numpy(move_np),
        scalars     = torch.from_numpy(scalars_np),
    )


# ── PokemonEmbeddings ─────────────────────────────────────────────────────────

class PokemonEmbeddings(nn.Module):
    """
    Embedding tables for all categorical Pokémon fields.

    forward(PokemonBatch) → [B, N, TOKEN_DIM=438]
    """

    def __init__(self):
        super().__init__()
        self.species_embed = nn.Embedding(N_SPECIES, D_SPECIES)
        self.move_embed    = nn.Embedding.from_pretrained(MOVE_EMBEDDING_PRIOR, freeze=False)
        self.item_embed    = nn.Embedding(N_ITEMS, D_ITEM)
        self.ability_embed = nn.Embedding(N_ABILITIES, D_ABILITY)
        self.type_embed    = nn.Embedding.from_pretrained(TYPE_EMBEDDING_PRIOR, freeze=False)
        self._init_species_from_base_stats()

    def _init_species_from_base_stats(self) -> None:
        """Seed species embeddings with normalised base stats for cold-start quality."""
        base_stats = _species_data.get("base_stats", {})
        _KEYS      = ("hp", "atk", "def", "spa", "spd", "spe")
        n_dims     = min(len(_KEYS), D_SPECIES)
        index      = _species_data["index"]
        with torch.no_grad():
            for name, stats in base_stats.items():
                idx = index.get(name)
                if idx is None:
                    continue
                _MAX = 255.0
                vals = [stats.get(k, 0) / _MAX for k in _KEYS[:n_dims]]
                self.species_embed.weight[idx, :n_dims] = torch.tensor(vals, dtype=torch.float32)

    def forward(self, batch: PokemonBatch) -> torch.Tensor:
        B, N = batch.species_idx.shape

        sp = self.species_embed(batch.species_idx)     # [B, N, D_SPECIES]
        t1 = self.type_embed(batch.type1_idx)          # [B, N, D_TYPE]
        t2 = self.type_embed(batch.type2_idx)          # [B, N, D_TYPE]
        tr = self.type_embed(batch.tera_idx)           # [B, N, D_TYPE]
        it = self.item_embed(batch.item_idx)           # [B, N, D_ITEM]
        ab = self.ability_embed(batch.ability_idx)     # [B, N, D_ABILITY]
        mv = self.move_embed(batch.move_idx)           # [B, N, 4, D_MOVE]
        mv = mv.reshape(B, N, 4 * D_MOVE)             # [B, N, 4*D_MOVE]

        return torch.cat([sp, t1, t2, tr, it, ab, mv, batch.scalars], dim=-1)
        # → [B, N, 32+8+8+8+16+16+128+222] = [B, N, 438]


# ── FieldBatch ────────────────────────────────────────────────────────────────

@dataclass
class FieldBatch:
    """
    Pre-collated field token for a batch of battle states.

    field: [N, FIELD_DIM=72]  — flat; caller reshapes to [B, K, FIELD_DIM].
    """
    field: torch.Tensor

    def to(self, device) -> "FieldBatch":
        return FieldBatch(field=self.field.to(device))


# ── collate_field_features ────────────────────────────────────────────────────

def collate_field_features(batch: list) -> FieldBatch:
    """
    Collate a flat list of N FieldFeatures into FieldBatch([N, FIELD_DIM]).
    Caller reshapes to [B, K, FIELD_DIM] (see rollout._build_agent_inputs).
    """
    B   = len(batch)
    arr = np.zeros((B, FIELD_DIM), dtype=np.float32)

    for b, f in enumerate(batch):
        o = 0
        arr[b, o:o + N_WEATHER] = f.weather;        o += N_WEATHER
        arr[b, o:o + N_TERRAIN] = f.terrain;        o += N_TERRAIN
        arr[b, o:o + N_PSEUDO]  = f.pseudo_weather; o += N_PSEUDO
        for side in [f.side0, f.side1]:
            arr[b, o:o + N_SIDE_CONDITIONS] = side.conditions; o += N_SIDE_CONDITIONS
            arr[b, o]     = side.pokemon_left
            arr[b, o + 1] = side.total_fainted
            o += 2

    return FieldBatch(field=torch.from_numpy(arr))
