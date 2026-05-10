"""Look for all [from] patterns in the log to understand item/ability reveals."""
import json

with open("C:/Users/Sun/Desktop/PokemonAI/CynthAI_v2/scripts/battle_log_dump.json") as f:
    data = json.load(f)

all_entries = []
for battle in data:
    for turn in battle["turns"]:
        all_entries.extend(turn.get("log_entries", []))

# Show ALL entries that have [from]
from_entries = [e for e in all_entries if "[from]" in e]
print(f"=== All [from] entries ({len(from_entries)}) ===")
for e in from_entries:
    print(f"  {e}")

# Show ALL entries with non-empty second field (entries that do something)
print(f"\n=== All entry types (counts) ===")
from collections import Counter
types = Counter()
for e in all_entries:
    parts = e.split("|")
    key = parts[0] if parts else "?"
    types[key] += 1
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {t:20s} x{c}")