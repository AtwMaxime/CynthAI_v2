"""
Debug script: catch SUB-TURN SWITCH FAIL and dump exact Rust error + state.

Usage:
    python -m debug._debug_catch_switch_fail [--seed SEED] [--n-battles N]

Plays battles with random legal actions and prints the Rust `choice_error`
whenever make_choices() or choose_side() returns False.

This is the same logic as the sub-turn handler in collect_rollout() but
with the error message exposed (requires simulator rebuild from 2026-05-12).
"""

import sys
sys.path.insert(0, ".")

import random
import torch
from simulator import PyBattle


def build_action_mask(state, side_idx):
    """Copy of rollout.build_action_mask for standalone use."""
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
        slot_conds = side.get("slot_conditions", {})
        is_revival = "revivalblessing" in slot_conds.get(active.get("position", 0), [])
        if is_revival:
            bench = [j for j, p in enumerate(side["pokemon"]) if j not in active_set and p.get("fainted", False)]
        else:
            bench = [j for j, p in enumerate(side["pokemon"]) if j not in active_set and not p.get("fainted", False)]
        for k in range(min(len(bench), 5)):
            mask[8 + k] = False
        return mask
    if not fainted:
        if not force_sw:
            for i, mv in enumerate(active.get("moves", [])[:4]):
                disabled = mv.get("disabled", False)
                no_pp = mv.get("pp", 0) <= 0
                if not disabled and not no_pp:
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
    """Copy of rollout.action_to_choice."""
    if 0 <= action <= 3:
        return f"move {action + 1}"
    if 4 <= action <= 7:
        return f"move {action - 3} terastallize"
    if 8 <= action <= 12:
        side = state["sides"][side_idx]
        active_set = {i for i in side["active"] if i is not None}
        if active_set:
            active_pos = next(iter(active_set))
            active = side["pokemon"][active_pos]
            slot_conds = side.get("slot_conditions", {})
            is_revival = "revivalblessing" in slot_conds.get(active.get("position", 0), [])
        else:
            is_revival = False
        bench = [j for j, p in enumerate(side["pokemon"]) if j not in active_set and (p.get("fainted", False) if is_revival else not p.get("fainted", False))]
        team_pos = bench[action - 8]
        return f"switch {team_pos + 1}"
    raise ValueError(f"Invalid action: {action}")


def dump_side_state(side, label):
    """Dump relevant state for debugging."""
    print(f"\n  --- {label} ---")
    print(f"  request_state: {side.get('request_state', '?')}")
    print(f"  choice_error: {side.get('choice_error', '(none)')!r}")
    active = [a for a in side["active"] if a is not None]
    print(f"  active slots: {active}")
    for idx, p in enumerate(side["pokemon"]):
        fainted = p.get("fainted", False)
        hp = f"{p.get('hp', 0)}/{p.get('maxhp', 0)}"
        print(f"    [{idx}] {p['species_id']:12s} hp={hp:8s} fainted={fainted} active={idx in active}")


def try_choices(b, state, side_idx, action, side_label):
    """Try make_choices with one action and show the Rust error on failure."""
    side = state["sides"][side_idx]
    req_self = state["sides"][0].get("request_state", "")
    req_opp = state["sides"][1].get("request_state", "")

    if req_self == "None" or req_opp == "None":
        # Sub-turn mode: one side chooses
        if req_self == "Switch" or req_opp == "Switch":
            # Only the side that needs to switch gets a choice
            if req_self == "Switch" and side_idx == 0:
                c = action_to_choice(action, state, 0)
                ok = b.make_choices(c, "")
                if not ok:
                    s2 = b.get_state()
                    print(f"\n  SUB-TURN FAIL | action={action} choice={c!r}")
                    print(f"  Rust error: {s2['sides'][0].get('choice_error', '(unknown)')!r}")
                    dump_side_state(s2['sides'][0], "side 0 (self)")
                    dump_side_state(s2['sides'][1], "side 1 (opp)")
                return ok
            elif req_opp == "Switch" and side_idx == 1:
                c = action_to_choice(action, state, 1)
                ok = b.make_choices("", c)
                if not ok:
                    s2 = b.get_state()
                    print(f"\n  SUB-TURN FAIL (opp) | action={action} choice={c!r}")
                    print(f"  Rust error: {s2['sides'][1].get('choice_error', '(unknown)')!r}")
                    dump_side_state(s2['sides'][0], "side 0 (self)")
                    dump_side_state(s2['sides'][1], "side 1 (opp)")
                return ok
            return True  # not our side to choose
        return True  # both None, skip
    else:
        # Normal turn: both sides choose simultaneously
        c_self = action_to_choice(action, state, 0) if side_idx == 0 else "move 1"
        c_opp = action_to_choice(action, state, 1) if side_idx == 1 else "move 1"
        ok = b.make_choices(c_self, c_opp)
        if not ok:
            s2 = b.get_state()
            print(f"\n  NORMAL TURN FAIL | p1={c_self!r} p2={c_opp!r}")
            print(f"  Rust error p1: {s2['sides'][0].get('choice_error', '(unknown)')!r}")
            print(f"  Rust error p2: {s2['sides'][1].get('choice_error', '(unknown)')!r}")
            dump_side_state(s2['sides'][0], "side 0 (self)")
            dump_side_state(s2['sides'][1], "side 1 (opp)")
        return ok


def run_debug(seed=42, max_turns=100):
    """Play a battle and catch any invalid choices with full debug output."""
    random.seed(seed)
    b = PyBattle("gen9randombattle", seed)
    turn = 0

    while not b.ended and turn < max_turns:
        turn += 1
        state = b.get_state()
        req_self = state["sides"][0].get("request_state", "")
        req_opp = state["sides"][1].get("request_state", "")

        # Sub-turn mode: one side chooses (the other has "None")
        if req_self == "None" or req_opp == "None":
            if req_self == "Switch" or req_opp == "Switch":
                for side_idx in [0, 1]:
                    side_req = state["sides"][side_idx].get("request_state", "")
                    if side_req == "Switch":
                        mask = build_action_mask(state, side_idx)
                        legal = (~mask).nonzero(as_tuple=True)[0].tolist()
                        if not legal:
                            continue
                        action = random.choice(legal)
                        ok = try_choices(b, state, side_idx, action, f"side {side_idx}")
                        if ok:
                            state = b.get_state()
                        else:
                            break
            # If both "None" (mid-turn), skip — make_choices was already called
            continue

        # Normal turn: both sides choose simultaneously
        # Pick legal actions for BOTH sides first
        p1_action = None
        p2_action = None
        for side_idx in [0, 1]:
            mask = build_action_mask(state, side_idx)
            legal = (~mask).nonzero(as_tuple=True)[0].tolist()
            req_state = state["sides"][side_idx].get("request_state", "")
            if not legal:
                print(f"  DEBUG: side {side_idx} req={req_state} NO LEGAL ACTIONS! active={state['sides'][side_idx]['active']}", file=sys.stderr)
            if legal:
                action = random.choice(legal)
                if side_idx == 0:
                    p1_action = action
                else:
                    p2_action = action
            if req_state == "Switch" and side_idx == 0 and p1_action is not None and 0 <= p1_action <= 7:
                print(f"  DEBUG: p1_action={p1_action} but req_state=Switch! legal={legal}", file=sys.stderr)

        if p1_action is None and p2_action is None:
            continue

        p1_choice = action_to_choice(p1_action, state, 0) if p1_action is not None else "default"
        p2_choice = action_to_choice(p2_action, state, 1) if p2_action is not None else "default"

        ok = b.make_choices(p1_choice, p2_choice)
        if not ok:
            s2 = b.get_state()
            print(f"\n  NORMAL TURN FAIL | turn={turn} p1={p1_choice!r} p2={p2_choice!r}")
            print(f"  Rust error p1: {s2['sides'][0].get('choice_error', '(unknown)')!r}")
            print(f"  Rust error p2: {s2['sides'][1].get('choice_error', '(unknown)')!r}")
            # Debug: show what masks produced
            dbg_mask0 = build_action_mask(state, 0)
            dbg_mask1 = build_action_mask(state, 1)
            dbg_legal0 = (~dbg_mask0).nonzero(as_tuple=True)[0].tolist()
            dbg_legal1 = (~dbg_mask1).nonzero(as_tuple=True)[0].tolist()
            print(f"  DBG: side0 req={state['sides'][0].get('request_state','')} legal={dbg_legal0} p1_action={p1_action}")
            print(f"  DBG: side1 req={state['sides'][1].get('request_state','')} legal={dbg_legal1} p2_action={p2_action}")
            dump_side_state(s2['sides'][0], "side 0 (self)")
            dump_side_state(s2['sides'][1], "side 1 (opp)")
            b = PyBattle("gen9randombattle", seed + turn)
        else:
            state = b.get_state()

        if b.ended:
            print(f"\nBattle ended in {turn} turns, winner={b.winner}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-battles", type=int, default=1)
    args = parser.parse_args()

    for i in range(args.n_battles):
        print(f"\n{'='*60}")
        print(f"Battle {i+1}/{args.n_battles} (seed={args.seed + i})")
        print(f"{'='*60}")
        run_debug(seed=args.seed + i)