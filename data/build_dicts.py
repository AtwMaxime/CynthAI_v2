"""
build_dicts.py — generate all index + embedding JSON files under data/dicts/.

Run once before training:
    python data/build_dicts.py

Outputs (data/dicts/):
  species_index.json   — name→index + base_stats, size
  move_index.json      — name→index, size
  item_index.json      — name→index, size
  ability_index.json   — name→index, size
  type_index.json      — name→index, size
  type_embeddings.json — [N_TYPES, D_TYPE] prior matrix + d_type
  move_embeddings.json — [N_MOVES, D_MOVE] prior matrix + d_move

Data source: poke-env's bundled Gen 9 data (poke_env.data.static.POKEDEX,
MOVES, ITEMS, ABILITIES, GEN9_MOVES …).  The embedding priors encode simple
structural features so the model has a warm start; all embeddings are fine-tuned
during self-play training.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

OUT = Path(__file__).parent / "dicts"
OUT.mkdir(parents=True, exist_ok=True)

# ── Type chart (Gen 9) — rows=attacker, cols=defender ─────────────────────────
# Order matches TYPE_INDEX: __unk__, normal, fire, water, electric, grass, ice,
#                           fighting, poison, ground, flying, psychic, bug, rock,
#                           ghost, dragon, dark, steel, fairy

_TYPES = [
    "__unk__", "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug", "rock",
    "ghost", "dragon", "dark", "steel", "fairy",
]
_N_TYPES = len(_TYPES)   # 19

# Effectiveness matrix [attacker][defender]: 0=immune, 0.5=not very, 1=normal, 2=super
# Row 0 (__unk__) is all 1.0 (neutral).
_EFF: list[list[float]] = [
    # unk  nor  fir  wat  ele  gra  ice  fig  poi  gro  fly  psy  bug  roc  gho  dra  dar  ste  fai
    [ 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1  ],  # __unk__
    [ 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 0.5,   0,   1,   1, 0.5,   1  ],  # normal
    [ 1,   1, 0.5, 0.5,   1,   2,   2,   1,   1,   1,   1,   1,   2, 0.5,   1, 0.5,   1,   2,   1  ],  # fire
    [ 1,   1,   2, 0.5,   1, 0.5,   1,   1,   1,   2,   1,   1,   1,   2,   1, 0.5,   1,   1,   1  ],  # water
    [ 1,   1,   1,   2, 0.5, 0.5,   1,   1,   1,   0,   2,   1,   1,   1,   1, 0.5,   1,   1,   1  ],  # electric
    [ 1,   1, 0.5,   2,   1, 0.5,   1,   1, 0.5,   2, 0.5,   1, 0.5,   2,   1, 0.5,   1, 0.5,   1  ],  # grass
    [ 1,   1, 0.5, 0.5,   1,   2, 0.5,   1,   1,   2,   2,   1,   1,   1,   1,   2,   1, 0.5,   1  ],  # ice
    [ 1,   2,   1,   1,   1,   1,   2,   1, 0.5,   1, 0.5, 0.5, 0.5,   2,   0,   1,   2,   2, 0.5  ],  # fighting
    [ 1,   1,   1,   1,   1,   2,   1,   1, 0.5, 0.5,   1,   1,   1, 0.5,   0,   1,   1,   0,   2  ],  # poison
    [ 1,   1,   2,   1,   2, 0.5,   1,   1,   2,   1,   0,   1, 0.5,   2,   1,   1,   1,   2,   1  ],  # ground
    [ 1,   1,   1,   1, 0.5,   2,   1,   2, 1,    1,   1,   1,   2, 0.5,   1,   1,   1, 0.5,   1  ],  # flying
    [ 1,   1,   1,   1,   1,   1,   1,   2,   2,   1,   1, 0.5,   1,   1,   1,   1,   0, 0.5,   1  ],  # psychic
    [ 1,   1, 0.5,   1,   1,   2,   1, 0.5, 0.5,   1, 0.5,   2,   1,   1, 0.5,   1,   2, 0.5, 0.5  ],  # bug
    [ 1,   1,   2,   1,   1,   1,   2, 0.5,   1, 0.5,   2,   1,   2,   1,   1,   1,   1, 0.5,   1  ],  # rock
    [ 1,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   2,   1, 0.5,   1,   1  ],  # ghost
    [ 1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1, 0.5,   0  ],  # dragon
    [ 1,   1,   1,   1,   1,   1,   1, 0.5,   1,   1,   1,   2,   1,   1,   2,   1, 0.5,   1, 0.5  ],  # dark
    [ 1,   1, 0.5,   0.5, 0.5,   1,   2,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1, 0.5,   2  ],  # steel
    [ 1,   1, 0.5,   1,   1,   1,   1,   2, 0.5,   1,   1,   1,   1,   1,   1,   2,   2, 0.5,   1  ],  # fairy
]

D_TYPE = 8


def _type_embedding_prior() -> list[list[float]]:
    """
    Build [N_TYPES, D_TYPE] type embedding prior.

    First D_TYPE = 8 dimensions per type = attacker effectiveness against types 1-8
    (skipping __unk__ and self for the input side).  This gives the model a
    structural initialisation encoding type match-up relationships.
    """
    embs = []
    for atk_row in _EFF:
        # Use log2 of effectiveness so values sit near 0, with ±1 for 2x/0.5x.
        vec = [math.log2(max(e, 1e-6)) for e in atk_row[1 : D_TYPE + 1]]
        embs.append(vec)
    return embs


def _build_type_data() -> None:
    index = {t: i for i, t in enumerate(_TYPES)}
    embs  = _type_embedding_prior()
    data  = {
        "index":      index,
        "size":       _N_TYPES,
        "embeddings": embs,
        "d_type":     D_TYPE,
        "types":      _TYPES,
    }
    with open(OUT / "type_index.json", "w") as f:
        json.dump({"index": index, "size": _N_TYPES}, f)
    with open(OUT / "type_embeddings.json", "w") as f:
        json.dump({"embeddings": embs, "d_type": D_TYPE, "types": _TYPES}, f)
    print(f"  type_index.json        ({_N_TYPES} types)")
    print(f"  type_embeddings.json   ({_N_TYPES}×{D_TYPE})")


# ── poke-env data helpers ──────────────────────────────────────────────────────

def _load_pokeenv() -> tuple[dict, dict, dict, dict]:
    """Load Gen 9 data dicts from poke_env.  Returns (pokedex, moves, items, abilities)."""
    try:
        from poke_env.data import GenData
        gd = GenData.from_gen(9)
        return gd.pokedex, gd.moves, gd.items, gd.abilities
    except Exception as e:
        print(f"  [warn] could not load poke-env data: {e}")
        return {}, {}, {}, {}


def _norm(name: str) -> str:
    return name.lower().replace("-", "").replace(" ", "").replace("'", "").replace(".", "")


def _build_species(pokedex: dict) -> None:
    UNK_NAME = "__unk__"
    index: dict[str, int] = {UNK_NAME: 0}
    bstats: dict[str, dict] = {}

    for raw_name, data in sorted(pokedex.items()):
        key = _norm(raw_name)
        if key not in index:
            index[key] = len(index)
        bs = data.get("baseStats", data.get("base_stats", {}))
        bstats[key] = {
            "hp":  bs.get("hp",  bs.get("HP",  0)),
            "atk": bs.get("atk", bs.get("Atk", 0)),
            "def": bs.get("def", bs.get("Def", 0)),
            "spa": bs.get("spa", bs.get("SpA", 0)),
            "spd": bs.get("spd", bs.get("SpD", 0)),
            "spe": bs.get("spe", bs.get("Spe", 0)),
        }

    out = {"index": index, "base_stats": bstats, "size": len(index)}
    with open(OUT / "species_index.json", "w") as f:
        json.dump(out, f)
    print(f"  species_index.json     ({len(index)} species)")


def _build_index(data_dict: dict, fname: str, label: str) -> None:
    UNK_NAME = "__unk__"
    index: dict[str, int] = {UNK_NAME: 0}
    for raw_name in sorted(data_dict.keys()):
        key = _norm(raw_name)
        if key not in index:
            index[key] = len(index)
    out = {"index": index, "size": len(index)}
    with open(OUT / fname, "w") as f:
        json.dump(out, f)
    print(f"  {fname:<26} ({len(index)} {label})")


D_MOVE = 32

def _build_move_embeddings(moves: dict, move_index: dict) -> None:
    """Build [N_MOVES, D_MOVE] prior from move type + category features."""
    N = len(move_index)
    type_to_idx = {t: i for i, t in enumerate(_TYPES)}

    # Simple prior: one-hot type (19-dim) + one-hot category (3-dim) + random padding
    rng = random.Random(42)
    embs: list[list[float]] = []
    for name, idx in sorted(move_index.items(), key=lambda x: x[1]):
        mv   = moves.get(name, {})
        t    = _norm(mv.get("type", ""))
        cat  = mv.get("category", "physical").lower()  # physical / special / status

        vec = [0.0] * D_MOVE
        # bits 0-18: type one-hot
        tidx = type_to_idx.get(t, 0)
        if tidx < D_MOVE:
            vec[tidx] = 1.0
        # bits 19-21: category one-hot
        cat_map = {"physical": 19, "special": 20, "status": 21}
        ci = cat_map.get(cat, 19)
        if ci < D_MOVE:
            vec[ci] = 1.0
        # bits 22-31: small random noise for diversity
        for j in range(22, D_MOVE):
            vec[j] = rng.gauss(0, 0.01)
        embs.append(vec)

    # Ensure length matches N
    while len(embs) < N:
        embs.append([0.0] * D_MOVE)

    with open(OUT / "move_embeddings.json", "w") as f:
        json.dump({"embeddings": embs, "d_move": D_MOVE}, f)
    print(f"  move_embeddings.json   ({N}×{D_MOVE})")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building data/dicts/ ...")
    pokedex, moves, items, abilities = _load_pokeenv()

    _build_type_data()

    if pokedex:
        _build_species(pokedex)
    else:
        print("  [skip] species_index.json  (no poke-env data)")

    if moves:
        _build_index(moves, "move_index.json", "moves")
        # reload to get the index for embedding generation
        with open(OUT / "move_index.json") as f:
            move_index = json.load(f)["index"]
        _build_move_embeddings(moves, move_index)
    else:
        print("  [skip] move_index.json / move_embeddings.json")

    if items:
        _build_index(items, "item_index.json", "items")
    else:
        print("  [skip] item_index.json")

    if abilities:
        _build_index(abilities, "ability_index.json", "abilities")
    else:
        print("  [skip] ability_index.json")

    print("Done.")
