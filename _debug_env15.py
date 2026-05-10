"""Diagnostic: test env 15 to understand why make_choices fails."""
import sys
sys.path.insert(0, ".")

from simulator import PyBattle

def test_env(label, seed):
    print(f"\n=== {label} (seed={seed}) ===")
    b = PyBattle("gen9randombattle", seed)
    s = b.get_state()

    side0 = s["sides"][0]
    side1 = s["sides"][1]

    print(f"  turn={s['turn']}, ended={s['ended']}")
    print(f"  side0 request_state={side0['request_state']!r}")
    print(f"  side1 request_state={side1['request_state']!r}")

    for side_idx, side in enumerate([side0, side1]):
        active = [a for a in side["active"] if a is not None]
        print(f"  side{side_idx} active slots: {active}")
        for poke in side["pokemon"]:
            fainted = poke.get("fainted", False)
            fswitch = poke.get("force_switch_flag", False)
            moves = [m["id"] for m in poke.get("moves", [])[:4]]
            print(f"    {poke['species_id']:12s} fainted={fainted} force_sw={fswitch} hp={poke['hp']}/{poke['maxhp']} moves={moves}")

    # Try each action 0-12 for side 0 and side 1
    print(f"\n  Testing actions for side 0 (self):")
    for action in range(13):
        act_str = f"move {action+1}" if action < 4 else (f"move {action-3} terastallize" if action < 8 else f"switch {action-7}")
        ok = b.choose_side(0, act_str)
        # Undo to try next action
        b.undo_choice(0)
        print(f"    action {action} -> {act_str:30s} choose_side={ok}")

for seed in [0, 7, 15, 42, 99]:
    test_env(f"seed {seed}", seed)