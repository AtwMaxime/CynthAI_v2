"""
env/team_pool.py — Pool d'équipes officielles Gen 9 Random Battle.

Charge data/team_pool.json au premier import et expose sample_teams()
pour tirer deux équipes aléatoires indépendantes.

Utilisation :
    from env.team_pool import sample_teams
    t1, t2 = sample_teams()
    battle = PyBattle.from_packed_teams("gen9randombattle", seed, t1, t2)
"""

import json
import random
from pathlib import Path

_POOL_PATH = Path(__file__).resolve().parent.parent / "data" / "team_pool.json"

def _load_pool() -> list[str]:
    with open(_POOL_PATH, encoding="utf-8") as f:
        pool = json.load(f)
    if not pool:
        raise RuntimeError(f"team_pool.json est vide : {_POOL_PATH}")
    return pool

# Chargement unique au premier import
_POOL: list[str] = _load_pool()

def sample_teams() -> tuple[str, str]:
    """Retourne deux équipes packed aléatoires et indépendantes."""
    return random.choice(_POOL), random.choice(_POOL)

def pool_size() -> int:
    return len(_POOL)
