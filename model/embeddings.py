"""
CynthAI_v2 Embeddings — categorical + scalar feature encoding for Pokémon tokens.

TOKEN_DIM = D_SPECIES + D_TYPE + D_TYPE + D_TYPE + D_ITEM + D_ABILITY + 4*D_MOVE + N_SCALARS
          = 32 + 8 + 8 + 8 + 16 + 16 + 128 + 223  =  439

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
    D_TYPE, N_VOLATILES, UNK,
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

N_SCALARS: int  = 223
TOKEN_DIM: int  = D_SPECIES + D_TYPE + D_TYPE + D_TYPE + D_ITEM + D_ABILITY + 4 * D_MOVE + N_SCALARS
# = 32 + 8 + 8 + 8 + 16 + 16 + 128 + 223 = 439


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
      scalars     : [B, N, 223]  float32
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
            scalars_np[b, f, 222]    = p.hp           # raw current HP

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


# ── Reveal mask (POMDP training augmentation) ────────────────────────────────

def apply_reveal_mask(
    batch:          PokemonBatch,
    reveal_species: torch.Tensor,   # [B, 6] bool — True = revealed
    reveal_item:    torch.Tensor,
    reveal_ability: torch.Tensor,
    reveal_tera:    torch.Tensor,
    reveal_moves:   torch.Tensor,   # [B, 6, 4] bool — per move slot
    mask_ratio:     float = 1.0,    # 0.0 = no masking, 1.0 = always mask unrevealed
) -> PokemonBatch:
    """
    Zero-mask opponent attributes that haven't been revealed through gameplay.

    Masking logic per opponent slot (per-Pokémon Bernoulli with mask_ratio):
      - mask_ratio=0.0 → no masking (full info), ratio=1.0 → always mask unrevealed
      - Species unknown → mask ALL categoricals + all scalars
      - Species known:
          Types & base_stats kept (deterministic from species)
          Item/ability/tera hidden if not yet revealed
          Moves hidden per-slot if not yet revealed
          Scalars: stats (idx 8:13) always hidden; move_pp (idx 21-24) hidden
            per-slot when move unrevealed
          hp_ratio, boosts, status, volatiles, move_disabled remain visible
    """
    B, N = batch.species_idx.shape
    device = batch.species_idx.device
    K_turns = N // 12
    N_SCAL = batch.scalars.shape[-1]

    # ── Per-Pokémon Bernoulli: which opponent slots get masked —───────────────
    should_mask = torch.rand(B, 6, device=device) < mask_ratio  # [B, 6] bool

    # ── Build opponent position index per K-block ─────────────────────────────
    def _to_N(per_slot: torch.Tensor) -> torch.Tensor:
        """Scatter [B, 6] into [B, N] at opponent positions across all K turns."""
        full = torch.ones(B, N, dtype=torch.bool, device=device)
        for t in range(K_turns):
            start = t * 12 + 6
            full[:, start:start + 6] = per_slot
        return full

    # ── Effective reveal per categorical attribute ────────────────────────────
    # True = keep data, False = mask to UNK
    # If should_mask=False → always keep (True)
    # If should_mask=True → keep only if revealed (and species known for derived fields)
    eff_species = (~should_mask) | reveal_species
    eff_types   = eff_species  # types deterministic from species
    eff_item    = (~should_mask) | (reveal_species & reveal_item)
    eff_ability = (~should_mask) | (reveal_species & reveal_ability)
    eff_tera    = (~should_mask) | (reveal_species & reveal_tera)
    eff_moves   = (~should_mask).unsqueeze(-1) | (reveal_species.unsqueeze(-1) & reveal_moves)

    # ── Scatter to [B, N] / [B, N, 4] ────────────────────────────────────────
    sp_N = _to_N(eff_species)
    it_N = _to_N(eff_item)
    ab_N = _to_N(eff_ability)
    te_N = _to_N(eff_tera)
    tp_N = _to_N(eff_types)

    mv_N = torch.ones(B, N, 4, dtype=torch.bool, device=device)
    for t in range(K_turns):
        start = t * 12 + 6
        mv_N[:, start:start + 6, :] = eff_moves

    # ── Scalar masking ────────────────────────────────────────────────────────
    scalar_mask = torch.ones(B, N, N_SCAL, dtype=torch.bool, device=device)

    for t in range(K_turns):
        opp_start = t * 12 + 6         # start of opponent block in N dim

        # Per-slot in this block
        for s in range(6):
            slot = opp_start + s
            bm = should_mask[:, s]                     # [B] — which envs mask this slot

            # 1. Species hidden: mask ALL scalars
            no_sp = bm & ~reveal_species[:, s]
            scalar_mask[no_sp, slot, :] = False

            # 2. Species known: mask stats (indices 8-13) + raw HP (index 222)
            known = bm & reveal_species[:, s]
            scalar_mask[known, slot, 8:14] = False
            scalar_mask[known, slot, 222]  = False

            # 3. Per-move-slot PP masking (indices 21-24)
            for j in range(4):
                mv_hidden = known & ~reveal_moves[:, s, j]
                scalar_mask[mv_hidden, slot, 21 + j] = False

    # ── Apply masks ───────────────────────────────────────────────────────────
    return PokemonBatch(
        species_idx = torch.where(sp_N, batch.species_idx, torch.zeros_like(batch.species_idx)),
        type1_idx   = torch.where(tp_N, batch.type1_idx,   torch.zeros_like(batch.type1_idx)),
        type2_idx   = torch.where(tp_N, batch.type2_idx,   torch.zeros_like(batch.type2_idx)),
        tera_idx    = torch.where(te_N, batch.tera_idx,    torch.zeros_like(batch.tera_idx)),
        item_idx    = torch.where(it_N, batch.item_idx,    torch.zeros_like(batch.item_idx)),
        ability_idx = torch.where(ab_N, batch.ability_idx, torch.zeros_like(batch.ability_idx)),
        move_idx    = torch.where(mv_N, batch.move_idx,   torch.zeros_like(batch.move_idx)),
        scalars     = torch.where(scalar_mask, batch.scalars, torch.zeros_like(batch.scalars)),
    )


# ── PokemonEmbeddings ─────────────────────────────────────────────────────────

class PokemonEmbeddings(nn.Module):
    """
    Embedding tables for all categorical Pokémon fields.

    forward(PokemonBatch) → [B, N, TOKEN_DIM=439]
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
        # → [B, N, 32+8+8+8+16+16+128+223] = [B, N, 439]


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
