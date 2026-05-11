"""Diagnostic: check position consistency mid-battle — no mutations, just reading."""
import sys
sys.path.insert(0, ".")

from simulator import PyBattle

def check_switch_slot_consistency(seed=42, max_turns=30):
    """
    Play a battle and check after each turn whether the action_to_choice
    switch mapping is consistent with the Rust sim's position-based check.

    The Rust choose_switch rejects if: target.position < active.len()
    The Python action_to_choice assumes: bench[k] = pokemon[index] where index is
    the array index, and builds "switch {index+1}".

    If position != array_index for bench Pokémon after switches, there's a mismatch.
    """
    b = PyBattle("gen9randombattle", seed)
    turn = 1

    while not b.ended and turn <= max_turns:
        state = b.get_state()
        print(f"\n=== Turn {turn} ===")
        mismatches = []

        for side_idx in [0, 1]:
            side = state["sides"][side_idx]
            req = side["request_state"]
            active = side["active"]  # Vec<Option<usize>> — indices into pokemon[]
            active_set = {i for i in active if i is not None}
            active_len = len([a for a in active if a is not None])

            print(f"  Side {side_idx}: request={req!r}, active={active}, active_len={active_len}")
            for i, p in enumerate(side["pokemon"]):
                pos = p.get("position", -1)
                fainted = p.get("fainted", False)
                is_active = i in active_set

                # Key check: does choose_switch(i) think this is active?
                # Rust: target.position < self.active.len() → considered active
                would_rust_consider_active = (pos < active_len)

                # Python: i in active_set → considered active
                python_considers_active = is_active

                conflict = "CONFLICT" if would_rust_consider_active != python_considers_active else ""
                if conflict:
                    mismatches.append(f"side{side_idx} pokemon[{i}] pos={pos} rust_active={would_rust_consider_active} python_active={python_considers_active}")

                flags = []
                if fainted: flags.append("FAINTED")
                if conflict: flags.append("CONFLICT")
                flag_str = " " + " ".join(flags) if flags else ""
                if flags:
                    print(f"    [{i}] pos={pos} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']}{flag_str}")

        if mismatches:
            print(f"\n  *** {len(mismatches)} MISMATCHES ***")
            for m in mismatches:
                print(f"    {m}")
            break  # Stop at first turn with mismatches

        # Advance the turn
        s = b.get_state()
        p1, p2 = "move 1", "move 1"
        for si, cmd_name in [(0, "p1"), (1, "p2")]:
            side = s["sides"][si]
            if side["request_state"] == "Switch":
                active_set = {i for i in side["active"] if i is not None}
                bench = [j for j, p in enumerate(side["pokemon"]) if j not in active_set and not p.get("fainted", False)]
                if bench:
                    if si == 0: p1 = f"switch {bench[0] + 1}"
                    else:       p2 = f"switch {bench[0] + 1}"

        ok = b.make_choices(p1, p2)
        if not ok:
            print(f"\n  FAILED: p1={p1!r} p2={p2!r}")
            return False
        turn += 1

    if not mismatches:
        print(f"\n  ✓ No mismatches found in {turn-1} turns (seed={seed})")
    return len(mismatches) == 0

if __name__ == "__main__":
    all_ok = True
    for seed in [0, 7, 15, 42, 99]:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")
        if not check_switch_slot_consistency(seed):
            all_ok = False

    if all_ok:
        print(f"\n\n✓ ALL SEEDS PASSED — no position mismatch")
    else:
        print(f"\n\n✗ SOME SEEDS HAVE MISMATCHES")