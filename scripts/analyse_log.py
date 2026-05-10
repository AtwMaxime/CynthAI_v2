"""Analyse the battle log dump to understand reveal event formats."""
import json
import sys
from collections import Counter

with open(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Sun/Desktop/PokemonAI/CynthAI_v2/scripts/battle_log_dump.json") as f:
    data = json.load(f)

# Collect all unique log entry prefixes
all_entries = []
ability_entries = []
item_entries = []
reveal_entries = []

for battle in data:
    for turn in battle["turns"]:
        for e in turn.get("log_entries", []):
            # Get the prefix (first field up to |)
            prefix = e.split("|")[0] if "|" in e else e
            all_entries.append(prefix)

            if "ability" in e.lower():
                ability_entries.append(e)
            if "item" in e.lower():
                item_entries.append(e)
            if "|switch|" in e or "|drag|" in e:
                reveal_entries.append(e)

print("=== Log entry prefixes ===")
for prefix, count in sorted(Counter(all_entries).items()):
    print(f"  {prefix:30s} x{count}")

print(f"\n=== Ability entries ({len(ability_entries)}) ===")
for e in ability_entries[:20]:
    print(f"  {e}")

print(f"\n=== Item entries ({len(item_entries)}) ===")
for e in item_entries[:20]:
    print(f"  {e}")

print(f"\n=== Switch/drag entries ({len(reveal_entries)}) ===")
for e in reveal_entries[:20]:
    print(f"  {e}")