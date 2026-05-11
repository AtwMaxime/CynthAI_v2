"""Test: play a full battle with random legal actions and check for invalid choices."""
import sys
sys.path.insert(0, ".")

from simulator import PyBattle
import random

def build_action_mask(state, side_idx):
    """Minimal copy of rollout.build_action_mask."""
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

def play_battle(seed, verbose=False):
    """Play one battle with random legal actions. Return (n_turns, n_invalid, finished_ok)."""
    b = PyBattle("gen9randombattle", seed)
    turn = 0
    n_invalid = 0

    while not b.ended and turn < 100:
        state = b.get_state()
        turn += 1
        req_self = state["sides"][0].get("request_state", "")
        req_opp  = state["sides"][1].get("request_state", "")

        # Switch sub-turn: un side attend (None) après KO
        if req_self == "None" or req_opp == "None":
            # Si les deux sont "None", la bataille n'attend aucune action
            if req_self != "Switch" and req_opp != "Switch":
                continue

            c_self, c_opp = "", ""

            def pick_switch(side_idx):
                mask = build_action_mask(state, side_idx)
                legal = [a for a in range(8, 13) if not mask[a]]
                if legal:
                    return action_to_choice(random.choice(legal), state, side_idx)
                return ""

            if req_self == "Switch":
                c_self = pick_switch(0)
            if req_opp == "Switch":
                c_opp = pick_switch(1)

            # Fallback si aucun switch légal trouvé
            if not c_self and req_self == "Switch":
                c_self = "default"
            if not c_opp and req_opp == "Switch":
                c_opp = "default"

            ok = b.make_choices(c_self, c_opp)
            if not ok:
                # Fallback: essayer "default" pour le côté Switch
                if req_self == "Switch" and c_self != "default":
                    c_self = "default"
                    ok = b.make_choices(c_self, c_opp)
                if not ok and req_opp == "Switch" and c_opp != "default":
                    c_opp = "default"
                    ok = b.make_choices(c_self, c_opp)
            if not ok:
                n_invalid += 1
                if verbose or n_invalid <= 3:
                    print(f"  INVALID switch sub-turn seed={seed} turn={turn} p1={c_self!r} p2={c_opp!r}")
                    # Debug dump for first failure
                    for si in [0, 1]:
                        side = state["sides"][si]
                        active_set = {i for i in side["active"] if i is not None}
                        print(f"    side{si}: req={side['request_state']!r} active={side['active']}")
                        for pi, p in enumerate(side["pokemon"]):
                            pos = p.get('position', '?')
                            print(f"      [{pi}] pos={pos} {p['species_id']:12s} hp={p['hp']}/{p['maxhp']} active={pi in active_set} fainted={p.get('fainted',False)} force_sw={p.get('force_switch_flag',False)}")
                b = PyBattle("gen9randombattle", seed + turn * 1000)
                turn = 0
            continue

        # Tour normal
        choices = {}

        for si in [0, 1]:
            mask = build_action_mask(state, si)
            legal = [a for a in range(13) if not mask[a]]
            if not legal:
                legal = [0]  # fallback
            a = random.choice(legal)
            c = action_to_choice(a, state, si)
            choices[si] = c

        p1 = choices[0] if True else ""  # side_self == 0
        p2 = choices[1] if True else ""
        ok = b.make_choices(p1, p2)

        if not ok:
            n_invalid += 1
            if verbose or n_invalid <= 3:
                print(f"  INVALID seed={seed} turn={turn} p1={p1!r} p2={p2!r}")
            # Reset the battle and continue to test more
            b = PyBattle("gen9randombattle", seed + turn * 1000)
            turn = 0  # reset counter for new battle
            continue

    return n_invalid

def main():
    random.seed(42)
    total_fail = 0
    n_battles = 100

    for battle_idx in range(n_battles):
        seed = battle_idx * 13 + 7
        fails = play_battle(seed, verbose=True)
        total_fail += fails
        if fails > 0:
            print(f"  Battle {battle_idx} (seed={seed}): {fails} invalid choices")
        else:
            print(f"  Battle {battle_idx} (seed={seed}): OK (no invalid)")

    print(f"\nTotal: {total_fail} invalid choices in {n_battles} battles")
    if total_fail == 0:
        print("ALL PASSED!")
    else:
        print(f"SOME FAILED ({total_fail} invalid choices total)")

if __name__ == "__main__":
    main()