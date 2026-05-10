"""Analyse heal and damage entries — look for [from] item: Leftovers etc."""
import json

with open("C:/Users/Sun/Desktop/PokemonAI/CynthAI_v2/scripts/battle_log_dump.json") as f:
    data = json.load(f)

all_entries = []
for battle in data:
    for turn in battle["turns"]:
        all_entries.extend(turn.get("log_entries", []))

# Show all heal and damage entries
print("=== |-heal| entries ===")
for e in all_entries:
    if e.startswith("|-heal|"):
        print(f"  {e}")

print("\n=== |-damage| entries (first 30) ===")
n = 0
for e in all_entries:
    if e.startswith("|-damage|"):
        print(f"  {e}")
        n += 1
        if n >= 30:
            break

print("\n=== |-sidestart| entries ===")
for e in all_entries:
    if e.startswith("|-sidestart|"):
        print(f"  {e}")

print("\n=== |-start| entries (first 20) ===")
n = 0
for e in all_entries:
    if e.startswith("|-start|"):
        print(f"  {e}")
        n += 1
        if n >= 20:
            break

print("\n=== |-fieldstart| entries ===")
for e in all_entries:
    if e.startswith("|-fieldstart|"):
        print(f"  {e}")

# Also count total of each type
from collections import Counter
types = Counter()
for e in all_entries:
    # Extract the message prefix up to the first | after the leading |
    parts = e.split("|")
    key = parts[0] if parts else "?"
    if key.startswith("|-"):
        types[key] += 1

print("\n=== All entry type counts ===")
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {t:25s} x{c}")