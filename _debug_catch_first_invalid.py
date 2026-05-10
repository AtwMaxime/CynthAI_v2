"""Play battles until the first invalid choice, dump full state."""
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
        # Check for Revival Blessing: when the active slot has revivalblessing
        # condition, we MUST target a fainted Pokemon (Rust sim rejects non-fainted)
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

random.seed(42)
for battle_idx in range(500):
    seed = battle_idx * 13 + 7
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
                    c_self = action_to_choice(random.choice(legal), state, 0)
            if req_opp == "Switch":
                mask = build_action_mask(state, 1)
                legal = [a for a in range(8, 13) if not mask[a]]
                if legal:
                    c_opp = action_to_choice(random.choice(legal), state, 1)
            if not c_self and req_self == "Switch":
                c_self = "default"
            if not c_opp and req_opp == "Switch":
                c_opp = "default"
            ok = b.make_choices(c_self, c_opp)
            if not ok:
                print(f"FIRST INVALID (sub-turn): battle={battle_idx} seed={seed} turn={turn}")
                print(f"  p1={c_self!r} p2={c_opp!r}")
                for si in [0, 1]:
                    side = state["sides"][si]
                    active_set = {i for i in side["active"] if i is not None}
                    print(f"  side{si}: req={side['request_state']!r} active={side['active']} total_fainted={side['total_fainted']}")
                    for i, p in enumerate(side["pokemon"]):
                        position = p.get("position", "?")
                        fainted = p.get("fainted", False)
                        tag = "ACTIVE" if i in active_set else "BENCH"
                        force_sw = p.get("force_switch_flag", False)
                        trapped = p.get("trapped", False)
                        print(f"    [{i}] pos={position} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']} {tag} fainted={fainted} force_sw={force_sw} trapped={trapped}")
                        for mv in (p.get("moves") or [])[:4]:
                            print(f"         move: {mv['id']:16s} pp={mv['pp']}/{mv['maxpp']} disabled={mv.get('disabled',False)}")
                sys.exit(0)
            continue

        choices = []

        for si in [0, 1]:
            mask = build_action_mask(state, si)
            legal = [a for a in range(13) if not mask[a]]
            if not legal:
                legal = [0]
            a = random.choice(legal)
            c = action_to_choice(a, state, si)
            choices.append((a, c))

        p1, p2 = choices[0][1], choices[1][1]
        ok = b.make_choices(p1, p2)

        if not ok:
            print(f"FIRST INVALID: battle={battle_idx} seed={seed} turn={turn}")
            print(f"  p1={p1!r} p2={p2!r}")
            print(f"  actions: p1_action={choices[0][0]} p2_action={choices[1][0]}")
            s = state  # state BEFORE make_choices
            for si in [0, 1]:
                side = s["sides"][si]
                active_set = {i for i in side["active"] if i is not None}
                print(f"  side{si}: req={side['request_state']!r} active={side['active']} total_fainted={side['total_fainted']}")
                for i, p in enumerate(side["pokemon"]):
                    position = p.get("position", "?")
                    fainted = p.get("fainted", False)
                    tag = "ACTIVE" if i in active_set else "BENCH"
                    force_sw = p.get("force_switch_flag", False)
                    trapped = p.get("trapped", False)
                    print(f"    [{i}] pos={position} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']} {tag} fainted={fainted} force_sw={force_sw} trapped={trapped}")
                    for mv in (p.get("moves") or [])[:4]:
                        print(f"         move: {mv['id']:16s} pp={mv['pp']}/{mv['maxpp']} disabled={mv.get('disabled',False)}")

            # Try to understand why the switch failed
            for si in [0, 1]:
                ac, ch = choices[si]
                if 8 <= ac <= 12:  # switch action
                    side = s["sides"][si]
                    active_set = {i for i in side["active"] if i is not None}
                    bench = [j for j, p in enumerate(side["pokemon"]) if j not in active_set and not p.get("fainted", False)]
                    team_pos = bench[ac - 8]
                    print(f"\n  side{si} switch analysis:")
                    print(f"    action {ac} -> bench[{ac-8}] = team_pos {team_pos} -> switch {team_pos+1}")
                    print(f"    team_pos {team_pos} pokemon: {side['pokemon'][team_pos]['species_id']}")
                    print(f"    pokemon.position = {side['pokemon'][team_pos].get('position', '?')}")
                    print(f"    check: position < active.len()? {side['pokemon'][team_pos].get('position', 0)} < {len([a for a in side['active'] if a is not None])}")
                    # also check choose_side directly
                    b2 = PyBattle("gen9randombattle", seed)
                    # can't replay to same state, skip

            sys.exit(0)

print("No invalid choices found in 500 battles")