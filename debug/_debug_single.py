"""Debug a specific battle where invalid choices occur, capturing full state."""
import sys
sys.path.insert(0, ".")

from simulator import PyBattle
import random

def build_action_mask(state, side_idx):
    import torch
    mask = torch.ones(13, dtype=torch.bool)
    side = state["sides"][side_idx]
    active_set = {i for i in side["active"] if i is not None}
    if not active_set:
        return mask
    active_pos = next(iter(active_set))
    active = side["pokemon"][active_pos]
    fainted = bool(active.get("fainted", False))
    force_sw = bool(active.get("force_switch_flag", False))
    trapped = bool(active.get("trapped", False))
    req_state = side.get("request_state", "")
    if req_state == "Switch":
        # Check for Revival Blessing
        slot_conds = side.get("slot_conditions", {})
        is_revival = "revivalblessing" in slot_conds.get(active.get("position", 0), [])
        if is_revival:
            bench = [
                j for j, p in enumerate(side["pokemon"])
                if j not in active_set and p.get("fainted", False)
            ]
        else:
            bench = [
                j for j, p in enumerate(side["pokemon"])
                if j not in active_set and not p.get("fainted", False)
            ]
        for k in range(min(len(bench), 5)):
            mask[8 + k] = False
        return mask
    if not fainted:
        if not force_sw:
            for i, mv in enumerate(active.get("moves", [])[:4]):
                if not mv.get("disabled", False):
                    mask[i] = False
            tera_used = any(p.get("terastallized") is not None for p in side["pokemon"])
            if not tera_used:
                for i in range(4):
                    if not mask[i]:
                        mask[i + 4] = False
    if fainted or force_sw or not trapped:
        bench = [j for j, p in enumerate(side["pokemon"]) if j not in active_set and not p.get("fainted", False)]
        for k in range(min(len(bench), 5)):
            mask[8 + k] = False
    return mask

def action_to_choice(action, state, side_idx):
    if 0 <= action <= 3:
        return f"move {action + 1}"
    if 4 <= action <= 7:
        return f"move {action - 3} terastallize"
    if 8 <= action <= 12:
        side = state["sides"][side_idx]
        active_set = {i for i in side["active"] if i is not None}
        # Must match build_action_mask's bench computation exactly
        if active_set:
            active_pos = next(iter(active_set))
            active = side["pokemon"][active_pos]
            slot_conds = side.get("slot_conditions", {})
            is_revival = "revivalblessing" in slot_conds.get(active.get("position", 0), [])
        else:
            is_revival = False
        bench = [
            j for j, p in enumerate(side["pokemon"])
            if j not in active_set and (p.get("fainted", False) if is_revival else not p.get("fainted", False))
        ]
        team_pos = bench[action - 8]
        return f"switch {team_pos + 1}"
    raise ValueError(f"Invalid action index: {action}")

def debug_battle(seed):
    """Play one battle, printing full state before any invalid choice."""
    b = PyBattle("gen9randombattle", seed)
    turn = 0

    while not b.ended and turn < 100:
        state = b.get_state()
        turn += 1
        req_self = state["sides"][0].get("request_state", "")
        req_opp  = state["sides"][1].get("request_state", "")

        # Switch sub-turn
        if req_self == "None" or req_opp == "None":
            if req_self != "Switch" and req_opp != "Switch":
                continue
            c_self, c_opp = "", ""
            if req_self == "Switch":
                mask = build_action_mask(state, 0)
                legal = [a for a in range(8, 13) if not mask[a]]
                if legal:
                    c_self = action_to_choice(legal[0], state, 0)
            if req_opp == "Switch":
                mask = build_action_mask(state, 1)
                legal = [a for a in range(8, 13) if not mask[a]]
                if legal:
                    c_opp = action_to_choice(legal[0], state, 1)
            if not c_self and req_self == "Switch":
                c_self = "default"
            if not c_opp and req_opp == "Switch":
                c_opp = "default"
            ok = b.make_choices(c_self, c_opp)
            if not ok:
                print(f"\n  *** INVALID switch sub-turn turn {turn}: p1={c_self!r} p2={c_opp!r} ***")
                for si in [0, 1]:
                    side = state["sides"][si]
                    print(f"  side{si}: req={side['request_state']!r}")
                    for i, p in enumerate(side["pokemon"]):
                        position = p.get("position", -2)
                        fainted = p.get("fainted", False)
                        print(f"    [{i}] pos={position} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']} fainted={fainted}")
                return False
            continue

        choices = {}

        for si in [0, 1]:
            mask = build_action_mask(state, si)
            legal = [a for a in range(13) if not mask[a]]
            if not legal:
                legal = [0]
            a = random.choice(legal)
            c = action_to_choice(a, state, si)
            choices[si] = (a, c, mask)

        p1 = choices[0][1]
        p2 = choices[1][1]

        # Before make_choices, verify what the Rust sim thinks
        for si in [0, 1]:
            side = state["sides"][si]
            chosen = choices[si]
            if 8 <= chosen[0] <= 12:
                # This is a switch action
                mask = chosen[2]
                req = side["request_state"]
                active = side["active"]
                active_set = {i for i in active if i is not None}
                print(f"  pre turn {turn} side{si}: req={req!r} active={active} action={chosen[0]} choice={chosen[1]!r}")
                print(f"    pokemon positions:")
                for i, p in enumerate(side["pokemon"]):
                    position = p.get("position", -2)
                    fainted = p.get("fainted", False)
                    active_tag = "ACTIVE" if i in active_set else "BENCH"
                    print(f"      [{i}] pos={position} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']} {active_tag} fainted={fainted}")
                print(f"    bench from active_set: {[j for j, p in enumerate(side['pokemon']) if j not in active_set and not p.get('fainted', False)]}")
                print(f"    legal switch actions: {[a for a in range(8, 13) if not mask[a]]}")

        ok = b.make_choices(p1, p2)

        if not ok:
            print(f"\n  *** INVALID turn {turn}: p1={p1!r} p2={p2!r} ***")
            # Print full state
            for si in [0, 1]:
                side = state["sides"][si]
                print(f"  side{si}: req={side['request_state']!r}")
                for i, p in enumerate(side["pokemon"]):
                    position = p.get("position", -2)
                    fainted = p.get("fainted", False)
                    print(f"    [{i}] pos={position} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']} fainted={fainted}")
            return False

    return True

if __name__ == "__main__":
    # Try the seeds that failed
    random.seed(42)
    for seed in [761, 774, 917, 1242]:
        print(f"\n{'='*60}")
        print(f"  Debugging seed={seed}")
        print(f"{'='*60}")
        debug_battle(seed)
        print(f"  (completed for seed={seed})")