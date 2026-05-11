"""Test new choose_side / commit_choices / undo_choice methods."""
import sys
sys.path.insert(0, ".")

from simulator import PyBattle

b = PyBattle("gen9randombattle", seed=42)

# Test choose_side + commit_choices
ok1 = b.choose_side(0, "move 1")
ok2 = b.choose_side(1, "move 1")
print(f"choose_side(0, move 1): {ok1}")
print(f"choose_side(1, move 1): {ok2}")
assert ok1 and ok2

b.commit_choices()
print(f"after commit: turn={b.turn}, ended={b.ended}")

# Test invalid choice
b3 = PyBattle("gen9randombattle", seed=7)
ok = b3.choose_side(0, "invalid choice")
print(f"invalid choice returns: {ok}")
assert not ok  # should return False

# Test undo then re-choose
b2 = PyBattle("gen9randombattle", seed=99)
ok1 = b2.choose_side(0, "move 1")
print(f"choose_side(0, move 1): {ok1}")
b2.undo_choice(0)
ok1b = b2.choose_side(0, "move 3")
print(f"undo then choose_side(0, move 3): {ok1b}")
assert ok1b

ok2 = b2.choose_side(1, "switch 6")
print(f"choose_side(1, switch 6): {ok2}")

if ok1b and ok2:
    b2.commit_choices()
    print(f"after commit: turn={b2.turn}, ended={b2.ended}")
else:
    print("one side failed, skipping commit")

print("ALL OK")