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

_SHOWDOWN_DATA = Path(__file__).parent.parent.parent / "pokemon-showdown-rs" / "data"


def _load_pokeenv() -> tuple[dict, dict, dict, dict]:
    """Load Gen 9 data dicts from poke_env.  Returns (pokedex, moves, items, abilities)."""
    try:
        from poke_env.data import GenData
        gd = GenData.from_gen(9)
        return gd.pokedex, gd.moves, gd.items, gd.abilities
    except Exception as e:
        print(f"  [warn] could not load poke-env data: {e}")
        return {}, {}, {}, {}


def _load_showdown_moves() -> dict:
    """Load moves directly from pokemon-showdown-rs/data/moves.json as fallback."""
    p = _SHOWDOWN_DATA / "moves.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


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
N_MOVE_FEATURES = 47

_SECONDARY_STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")


def _build_move_features(moves: dict, move_index: dict) -> None:
    """Build [N_MOVES, 47] static feature matrix for all moves.

    Features (47 dims):
      0-18  : type one-hot (19)
      19-21 : category one-hot (Physical/Special/Status)
      22    : basePower (raw int)
      23    : accuracy (raw int, true→101)
      24    : pp (raw int)
      25    : priority (raw int)
      26-35 : flags (contact, sound, bullet, slicing, punch, bite, heal, recharge, bypasssub, defrost)
      36    : selfSwitch (0/1)
      37    : drain ratio (drain[0]/drain[1] or 0)
      38    : recoil ratio (recoil[0]/recoil[1] or 0)
      39    : hasCrashDamage (0/1)
      40    : secondary chance (raw int)
      41-46 : secondary status one-hot (brn/par/psn/tox/slp/frz)
    """
    N = len(move_index)
    type_to_idx = {t: i for i, t in enumerate(_TYPES)}
    flag_keys = ("contact", "sound", "bullet", "slicing", "punch", "bite",
                 "heal", "recharge", "bypasssub", "defrost")

    feature_names = (
        [f"type_{t}" for t in _TYPES]
        + ["cat_physical", "cat_special", "cat_status"]
        + ["basePower", "accuracy", "pp", "priority"]
        + list(flag_keys)
        + ["selfSwitch", "drain_ratio", "recoil_ratio", "hasCrashDamage"]
        + ["secondary_chance"]
        + [f"sec_{s}" for s in _SECONDARY_STATUSES]
    )
    assert len(feature_names) == N_MOVE_FEATURES, f"Expected {N_MOVE_FEATURES}, got {len(feature_names)}"

    feats: list[list[float]] = []
    for name, idx in sorted(move_index.items(), key=lambda x: x[1]):
        mv = moves.get(name, {})
        vec = [0.0] * N_MOVE_FEATURES

        # 0-18: type one-hot
        t = _norm(mv.get("type", ""))
        tidx = type_to_idx.get(t, 0)
        vec[tidx] = 1.0

        # 19-21: category one-hot
        cat = mv.get("category", "Physical")
        cat_map = {"Physical": 19, "Special": 20, "Status": 21}
        vec[cat_map.get(cat, 19)] = 1.0

        # 22-25: basePower, accuracy, pp, priority
        vec[22] = float(mv.get("basePower", 0))
        acc = mv.get("accuracy", True)
        vec[23] = 101.0 if acc is True else float(acc)
        vec[24] = float(mv.get("pp", 0))
        vec[25] = float(mv.get("priority", 0))

        # 26-35: flags
        flags = mv.get("flags", {})
        for i, fk in enumerate(flag_keys):
            vec[26 + i] = float(flags.get(fk, 0))

        # 36: selfSwitch
        vec[36] = 1.0 if mv.get("selfSwitch") else 0.0

        # 37: drain ratio
        drain = mv.get("drain")
        vec[37] = drain[0] / drain[1] if drain else 0.0

        # 38: recoil ratio
        recoil = mv.get("recoil")
        vec[38] = recoil[0] / recoil[1] if recoil else 0.0

        # 39: hasCrashDamage
        vec[39] = 1.0 if mv.get("hasCrashDamage") else 0.0

        # 40-46: secondary chance + status one-hot
        sec = mv.get("secondary")
        if sec and isinstance(sec, dict):
            vec[40] = float(sec.get("chance", 0))
            sec_status = sec.get("status", "")
            if sec_status in _SECONDARY_STATUSES:
                vec[41 + _SECONDARY_STATUSES.index(sec_status)] = 1.0

        feats.append(vec)

    while len(feats) < N:
        feats.append([0.0] * N_MOVE_FEATURES)

    data = {
        "features": feats,
        "n_features": N_MOVE_FEATURES,
        "feature_names": feature_names,
    }
    with open(OUT / "move_features.json", "w") as f:
        json.dump(data, f)
    print(f"  move_features.json     ({N}×{N_MOVE_FEATURES})")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building data/dicts/ ...")
    pokedex, moves, items, abilities = _load_pokeenv()

    _build_type_data()

    if pokedex:
        _build_species(pokedex)
    else:
        print("  [skip] species_index.json  (no poke-env data)")

    # Move index from poke-env; move features from showdown JSON (richer data)
    if moves:
        _build_index(moves, "move_index.json", "moves")
    else:
        print("  [skip] move_index.json  (no poke-env data)")

    # Build move features — prefer showdown moves.json (has full mechanical data)
    showdown_moves = _load_showdown_moves()
    moves_for_features = showdown_moves if showdown_moves else moves
    if moves_for_features and (OUT / "move_index.json").exists():
        with open(OUT / "move_index.json") as f:
            move_index = json.load(f)["index"]
        _build_move_features(moves_for_features, move_index)
    elif not moves_for_features:
        print("  [skip] move_features.json  (no move data)")

    if items:
        _build_index(items, "item_index.json", "items")
    else:
        print("  [skip] item_index.json")

    if abilities:
        _build_index(abilities, "ability_index.json", "abilities")
    else:
        print("  [skip] ability_index.json")

    print("Done.")
