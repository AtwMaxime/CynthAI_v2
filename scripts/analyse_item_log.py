"""Analyse item reveal patterns in battle log."""
import json
import sys

with open("C:/Users/Sun/Desktop/PokemonAI/CynthAI_v2/scripts/battle_log_dump.json") as f:
    data = json.load(f)

all_entries = []
for battle in data:
    for turn in battle["turns"]:
        all_entries.extend(turn.get("log_entries", []))

# Search for item-related patterns
print(f"Total log entries: {len(all_entries)}")

# Find ALL entries that mention "item" in any way
item_related = [e for e in all_entries if "item" in e.lower()]
print(f"\n=== All 'item' mentions ({len(item_related)}) ===")
for e in item_related:
    print(f"  {e}")

# Find entries with [from] item: (Leftovers, Life Orb, etc.)
from_item = [e for e in all_entries if "[from] item:" in e]
print(f"\n=== [from] item: entries ({len(from_item)}) ===")
for e in from_item:
    print(f"  {e}")

# Find entries with [from] ability:
from_ability = [e for e in all_entries if "[from] ability:" in e]
print(f"\n=== [from] ability: entries ({len(from_ability)}) ===")
for e in from_ability[:20]:
    print(f"  {e}")

# Look at unique entry prefixes (first 3 chars)
import re
prefixes = set()
for e in all_entries:
    m = re.match(r'^(\|[^-][^|]*\||\|[^|]+\||[^|]+)', e)
    if m:
        prefixes.add(m.group(0))
    elif e:
        prefixes.add(e[:20])
    else:
        prefixes.add("(empty)")

print(f"\n=== Unique entry patterns (first 3 fields max) ===")
for p in sorted(prefixes)[:50]:
    print(f"  {p}")