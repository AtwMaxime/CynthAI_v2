import sys
sys.path.insert(0, "C:/Users/Sun/Desktop/PokemonAI/CynthAI_v2")
import simulator

# Debug seeds 9 and 105
for seed, midx, label in [(9, 1, "steelroller"), (105, 3, "inferno")]:
    b = simulator.PyBattle("gen9randombattle", seed)
    state = b.get_state()
    side0 = state["sides"][0]
    side1 = state["sides"][1]
    active0 = next((i for i in side0["active"] if i is not None), None)
    active1 = next((i for i in side1["active"] if i is not None), None)
    p0 = side0["pokemon"][active0]
    p1 = side1["pokemon"][active1] if active1 is not None else None
    print(f"\nseed={seed} {label}")
    print(f"  p1: {p0['species_id']} types={p0['types']} tera={p0['tera_type']} moves={[m['id'] for m in p0['moves']]}")
    print(f"  p2: {p1['species_id'] if p1 else 'none'} types={p1['types'] if p1 else '?'}")
    print(f"  p2 HP: {p1['hp']}/{p1['maxhp'] if p1 else '?'}")

    ok = b.make_choices(f"move {midx} terastallize", "move 1")
    print(f"  make_choices ok={ok}")
    log = b.get_new_log_entries()
    state2 = b.get_state()
    print(f"  ended={state2.get('ended')} winner={state2.get('winner')}")
    for l in log:
        if any(x in l for x in ["-damage", "-miss", "faint", "immune", "terastallize"]):
            print(f"  > {l}")
