"""Diagnostic v2: test each action on separate battle objects."""
import sys
sys.path.insert(0, ".")

from simulator import PyBattle

def test_action(seed, side_idx, action_str, label):
    b = PyBattle("gen9randombattle", seed)
    ok = b.choose_side(side_idx, action_str)
    print(f"  {label:30s} choose_side={ok}")
    return ok

# Test each action 0-12 for side 0 on SEPARATE battles
print("=== Action validity on fresh battles (seed=15) ===")
print("Side 0 (p1):")
for action in range(13):
    if action < 4:
        s = f"move {action+1}"
    elif action < 8:
        s = f"move {action-3} terastallize"
    else:
        s = f"switch {action-7}"
    test_action(15, 0, s, f"action {action} -> {s}")

print("\nSide 1 (p2):")
for action in range(13):
    if action < 4:
        s = f"move {action+1}"
    elif action < 8:
        s = f"move {action-3} terastallize"
    else:
        s = f"switch {action-7}"
    test_action(15, 1, s, f"action {action} -> {s}")

# Test make_choices with various action pairs on fresh battles
print("\n=== make_choices on fresh battles (seed=15) ===")
for a_self in range(13):
    for a_opp in range(13):
        b = PyBattle("gen9randombattle", 15)
        if a_self < 4: s_self = f"move {a_self+1}"
        elif a_self < 8: s_self = f"move {a_self-3} terastallize"
        else: s_self = f"switch {a_self-7}"

        if a_opp < 4: s_opp = f"move {a_opp+1}"
        elif a_opp < 8: s_opp = f"move {a_opp-3} terastallize"
        else: s_opp = f"switch {a_opp-7}"

        ok = b.make_choices(s_self, s_opp)
        if ok:
            print(f"  OK: p1={s_self:30s} p2={s_opp:30s} turn={b.turn} ended={b.ended}")