"""
Explore the Rust battle log: run a few full battles and dump all log entries.
Focus on |-ability and |-item entries to understand how reveals work.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import PyBattle

from env.state_encoder import MOVE_INDEX, UNK

def run_battle(seed: int, max_turns: int = 100) -> list[dict]:
    """Run a full battle and record turn-by-turn log entries."""
    battle = PyBattle("gen9randombattle", seed)
    turns = []

    for turn_num in range(1, max_turns + 1):
        if battle.ended:
            break

        state = battle.get_state()

        # Dumb random choices for both sides
        p1_choice = "default"
        p2_choice = "default"

        for side_idx in [0, 1]:
            side = state["sides"][side_idx]
            req_state = side.get("request_state", "")

            if req_state == "Switch":
                active_set = {i for i in side["active"] if i is not None}
                bench = []
                for j, p in enumerate(side["pokemon"]):
                    if j not in active_set and not p.get("fainted", False):
                        bench.append(j)
                choice = f"switch {bench[0] + 1}" if bench else "default"
            else:
                active_set = {i for i in side["active"] if i is not None}
                active_pos = next(iter(active_set)) if active_set else 0
                active = side["pokemon"][active_pos]
                moves = active.get("moves", [])
                choice = "default"
                for mv in moves[:4]:
                    if not mv.get("disabled", False) and mv.get("pp", 0) > 0:
                        choice = f"move {moves.index(mv) + 1}"
                        break

            if side_idx == 0:
                p1_choice = choice
            else:
                p2_choice = choice

        # Get fresh log entries before make_choices (the turn start)
        log_before = battle.get_new_log_entries()

        try:
            battle.make_choices(p1_choice, p2_choice)
        except BaseException as e:
            turns.append({
                "turn": turn_num,
                "p1_choice": p1_choice,
                "p2_choice": p2_choice,
                "error": str(e),
                "log_before": log_before[:20],
            })
            break

        # Get log entries produced by this turn
        log_after = battle.get_new_log_entries()

        # Filter relevant entries
        ability_entries = [e for e in log_after if "|-ability|" in e or "|ability|" in e]
        item_entries = [e for e in log_after if "|-item|" in e or "|item|" in e or "|-enditem|" in e]
        switch_entries = [e for e in log_after if "|switch|" in e or "|drag|" in e]
        faint_entries = [e for e in log_after if "|-faint|" in e or "|-damage|" in e]

        turns.append({
            "turn": turn_num,
            "p1_choice": p1_choice,
            "p2_choice": p2_choice,
            "winner": battle.winner,
            "log_entries": log_after[:50],  # first 50 entries
            "ability_entries": ability_entries,
            "item_entries": item_entries,
            "switch_entries": switch_entries,
            "faint_entries": faint_entries[:10],
        })

    return {
        "seed": seed,
        "total_turns": len(turns),
        "winner": battle.winner if battle.ended else None,
        "turns": turns,
    }


def main():
    # Run 5 battles with different seeds
    all_battles = []
    for seed in [1, 42, 99, 123, 256]:
        print(f"Running battle seed={seed}...")
        battle_data = run_battle(seed)
        all_battles.append(battle_data)
        print(f"  {battle_data['total_turns']} turns, winner={battle_data['winner']}")
        # Count how many ability/item reveal events
        n_ability = sum(len(t.get("ability_entries", [])) for t in battle_data["turns"])
        n_item = sum(len(t.get("item_entries", [])) for t in battle_data["turns"])
        print(f"  {n_ability} ability events, {n_item} item events")

    # Save full dump
    out_path = os.path.join(os.path.dirname(__file__), "battle_log_dump.json")
    with open(out_path, "w") as f:
        json.dump(all_battles, f, indent=1, default=str)
    print(f"\nSaved to {out_path}")

    # Also save a summary text file
    out_txt = os.path.join(os.path.dirname(__file__), "battle_log_summary.txt")
    with open(out_txt, "w") as f:
        for battle in all_battles:
            f.write(f"Battle seed={battle['seed']} ({battle['total_turns']} turns, winner={battle['winner']})\n")
            f.write("=" * 60 + "\n")
            for t in battle["turns"]:
                f.write(f"\nTurn {t['turn']}:\n")
                f.write(f"  P1: {t.get('p1_choice', '?')}  |  P2: {t.get('p2_choice', '?')}\n")

                for e in t.get("item_entries", []):
                    f.write(f"  [ITEM] {e}\n")
                for e in t.get("ability_entries", []):
                    f.write(f"  [ABILITY] {e}\n")
                for e in t.get("switch_entries", []):
                    f.write(f"  [SWITCH] {e}\n")

                if t.get("error"):
                    f.write(f"  ERROR: {t['error']}\n")
            f.write("\n\n")
    print(f"Summary saved to {out_txt}")


if __name__ == "__main__":
    main()