import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulator import PyBattle
from env.state_encoder import encode_pokemon, N_SPECIES, N_MOVES, N_ITEMS, N_ABILITIES, N_TYPES, D_TYPE

b = PyBattle('gen9randombattle', 42)
state = b.get_state()

print("=== Vocabulaire ===")
print(f"  Species  : {N_SPECIES}")
print(f"  Moves    : {N_MOVES}")
print(f"  Items    : {N_ITEMS}")
print(f"  Abilities: {N_ABILITIES}")
print(f"  Types    : {N_TYPES}  (d_type={D_TYPE})")

print()
print("=== Sections 1-3 : Identity + HP + Types ===")
for side_i, side in enumerate(state["sides"]):
    for poke in side["pokemon"]:
        feat = encode_pokemon(poke)
        unk  = "  << UNK" if feat.species_idx == 0 else ""
        types_raw = "/".join(poke.get("types", []))
        tera_raw  = poke.get("tera_type") or "-"
        print(f"  side{side_i}  {poke['species_id']:<28} idx={feat.species_idx:<5} lvl={feat.level:.2f}  hp={feat.hp_ratio:.2f}"
              f"  t1={feat.type1_idx:<3} t2={feat.type2_idx:<3} tera={feat.tera_idx:<3}({tera_raw})"
              f"  types={types_raw}{unk}")
