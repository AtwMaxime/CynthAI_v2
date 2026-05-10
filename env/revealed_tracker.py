"""
RevealedTracker — tracks what information about opponent Pokémon has been
revealed through natural gameplay events in the battle log.

The Rust simulator exposes a battle log (PS protocol format) that records all
game events. We parse this log after each make_choices() call to detect when:
  - A Pokémon switches in → species revealed (|switch|)
  - An ability activates → ability revealed (|-ability|)
  - An item is consumed/knocked off → item revealed (|-enditem|)
  - An item heals/deals damage → item revealed (|-heal|/|-damage| [from] <item>)
  - A move is used → move revealed (PP < maxPP in state)
  - A Pokémon terastallizes → tera revealed (terastallized flag)

For each opponent Pokémon (6 slots), we maintain a RevealedState:
  species: bool  — species known (switched in at least once)
  item:    bool  — item known
  ability: bool  — ability known
  tera:    bool  — tera type known
  moves:   [bool; 4] — each move slot known (PP < maxPP)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from env.state_encoder import ITEM_INDEX, ABILITY_INDEX

if TYPE_CHECKING:
    from collections.abc import Sequence

# ── Log identifier parsing ────────────────────────────────────────────────────

def _parse_log_id(log_id: str) -> tuple[int, str, str | None]:
    """
    Parse a Pokémon Showdown log identifier into (side_idx, form, name).

    Formats:
      "p1a"              → (0, "active", None)
      "p2a"              → (1, "active", None)
      "p1a: Squawkabilly" → (0, "active", "squawkabilly")
      "p2: Zacian"        → (1, "named", "zacian")
      "p1: Latias"        → (0, "named", "latias")
    """
    log_id = log_id.strip()

    # Determine side
    if log_id.startswith("p1"):
        side = 0
        rest = log_id[2:]
    elif log_id.startswith("p2"):
        side = 1
        rest = log_id[2:]
    else:
        return (0, "unknown", None)

    # Determine form
    is_active = rest.startswith("a")
    if is_active:
        rest = rest[1:]  # skip 'a'

    # Extract name after colon
    name = None
    if rest.startswith(": "):
        name = rest[2:].split(",")[0].strip().lower()
    elif rest.startswith(":"):
        name = rest[1:].split(",")[0].strip().lower()

    form = "active" if is_active else "named"
    return (side, form, name)


def _log_id_to_slot(log_id: str, side: dict) -> int | None:
    """
    Map a log identifier (e.g. 'p2a', 'p2: Zacian') to a team slot index (0-5).
    Returns None if the Pokémon cannot be found.
    """
    side_idx, form, name = _parse_log_id(log_id)
    team = side["pokemon"]
    active_set = {i for i in side.get("active", []) if i is not None}

    if form == "active":
        # Active Pokémon — find which slot is currently active
        if active_set:
            return next(iter(active_set))
        # Fallback: look for Pokémon marked is_active
        for j, poke in enumerate(team):
            if poke.get("is_active", False):
                return j
        return 0

    if form == "named" and name is not None:
        # Named Pokémon — find by name or species
        for j, poke in enumerate(team):
            poke_name = poke.get("name", "").lower()
            poke_species = str(poke.get("species_id", "")).lower()
            if poke_name == name or poke_species == name:
                return j
        # Try matching by species_id only (name can be a nickname)
        for j, poke in enumerate(team):
            poke_species = str(poke.get("species_id", "")).lower()
            if poke_species == name:
                return j

    return None


# ── Log parsing helpers ───────────────────────────────────────────────────────

def _extract_from_entries(log_entries: list[str]) -> list[tuple[str, str]]:
    """
    Extract all [from] clauses from log entries.
    Returns list of (pokemon_id, from_value) tuples.
    """
    results = []
    for entry in log_entries:
        if "[from]" not in entry:
            continue

        parts = entry.split("|")

        # Find the pokemon identifier (usually second field for |- prefixed)
        # |-heal|p1a|...|[from] leftovers
        # |-damage|p1a|...|[from] lifeorb
        # |-enditem|p2a|Vile Vial|[from] move: Knock Off|[of] p1a
        # |-weather|SunnyDay|[from] ability: Orichalcum Pulse|[of] p1a

        pokemon_id = None
        if parts[0].startswith("|-"):
            # First field is the event type, second is often the pokemon
            for part in parts[1:]:
                if part.startswith("p1") or part.startswith("p2"):
                    pokemon_id = part.split("[")[0].strip()
                    break

        # Extract the [from] value
        for part in parts:
            if part.startswith("[from] "):
                from_val = part[7:].strip()
                if pokemon_id:
                    results.append((pokemon_id, from_val))

    return results


def _extract_ability_entries(log_entries: list[str]) -> list[tuple[str, str, str]]:
    """
    Extract ability reveal entries from log.
    Returns list of (pokemon_id, ability_name, source_type).
    source_type is "direct", "activate", or "from".
    """
    results = []
    for entry in log_entries:
        parts = entry.split("|")

        # |-ability|pokemon|ability|...
        if entry.startswith("|-ability|") and len(parts) >= 3:
            pokemon_id = parts[1].split("[")[0].strip()
            ability = parts[2].split("[")[0].strip()
            if pokemon_id and ability:
                results.append((pokemon_id, ability, "direct"))

        # |-activate|pokemon|ability: Name|...
        if entry.startswith("|-activate|") and len(parts) >= 3:
            rest = parts[2]
            if rest.startswith("ability: "):
                pokemon_id = parts[1].split("[")[0].strip()
                ability = rest[9:].split("[")[0].strip()
                if pokemon_id and ability:
                    results.append((pokemon_id, ability, "activate"))

        # |-endability|pokemon — ability suppressed (was active, therefore known)
        if entry.startswith("|-endability|") and len(parts) >= 2:
            pokemon_id = parts[1].split("[")[0].strip()
            if pokemon_id:
                results.append((pokemon_id, "", "endability"))

    return results


def _extract_item_entries(log_entries: list[str]) -> list[tuple[str, str, str]]:
    """
    Extract item reveal entries from log.
    Returns list of (pokemon_id, item_name, source_type).
    source_type is "enditem", "from", or "from_ability".
    """
    results = []
    for entry in log_entries:
        parts = entry.split("|")

        # |-enditem|pokemon|item|...
        if entry.startswith("|-enditem|") and len(parts) >= 3:
            pokemon_id = parts[1].split("[")[0].strip()
            item = parts[2].split("[")[0].strip()
            if pokemon_id and item:
                results.append((pokemon_id, item, "enditem"))

    # [from] entries: check against ITEM_INDEX and ABILITY_INDEX
    for pokemon_id, from_val in _extract_from_entries(log_entries):
        # Check if it's an item name
        if from_val.lower() in ITEM_INDEX:
            results.append((pokemon_id, from_val, "from"))
        # Check if it's from ability: prefix (ability revealed, not item)
        # Handled by _extract_ability_entries via "activate" pattern

    return results


def _extract_switch_entries(log_entries: list[str]) -> list[str]:
    """Extract switch-in entries (species reveal). Returns list of pokemon_ids."""
    results = []
    for entry in log_entries:
        if entry.startswith("|switch|"):
            parts = entry.split("|")
            if len(parts) >= 2:
                pokemon_id = parts[1].split("[")[0].strip()
                if pokemon_id:
                    results.append(pokemon_id)
        # Also handle |drag| which is a forced switch
        if entry.startswith("|drag|"):
            parts = entry.split("|")
            if len(parts) >= 2:
                pokemon_id = parts[1].split("[")[0].strip()
                if pokemon_id:
                    results.append(pokemon_id)
    return results


# ── RevealedTracker ────────────────────────────────────────────────────────────

class RevealedTracker:
    """
    Tracks what information has been revealed about opponent Pokémon per env.

    Usage in collect_rollout():
        tracker = RevealedTracker(n_envs)
        # After each make_choices():
        tracker.update(env_idx, log_entries, curr_state, side_opp)
        # To get current state for storage:
        reveal_state = tracker.get_state(env_idx)
        # On episode end:
        tracker.reset(env_idx)
    """

    __slots__ = (
        "_n_envs",
        "_species",
        "_item",
        "_ability",
        "_tera",
        "_moves",
        "_ever_active",
    )

    def __init__(self, n_envs: int):
        self._n_envs = n_envs
        self._reset_env_data()

    def _reset_env_data(self):
        n = self._n_envs
        self._species = [[False] * 6 for _ in range(n)]
        self._item    = [[False] * 6 for _ in range(n)]
        self._ability = [[False] * 6 for _ in range(n)]
        self._tera    = [[False] * 6 for _ in range(n)]
        self._moves   = [[[False] * 4 for _ in range(6)] for _ in range(n)]
        self._ever_active = [[False] * 6 for _ in range(n)]

    def reset(self, env_idx: int):
        """Reset tracked state for a specific env (on episode end)."""
        self._species[env_idx] = [False] * 6
        self._item[env_idx]    = [False] * 6
        self._ability[env_idx] = [False] * 6
        self._tera[env_idx]    = [False] * 6
        self._moves[env_idx]   = [[False] * 4 for _ in range(6)]
        self._ever_active[env_idx] = [False] * 6

    def update(
        self,
        env_idx: int,
        log_entries: list[str],
        curr_state: dict,
        side_idx: int,
    ):
        """
        Update revealed state from log entries and current state.

        Args:
            env_idx: Which env's tracker to update.
            log_entries: New log entries from get_new_log_entries().
            curr_state: Current battle state from get_state().
            side_idx: Which side (0 or 1) the TRACKED player is on (usually side_opp).
        """
        if env_idx < 0 or env_idx >= self._n_envs:
            return

        side = curr_state["sides"][side_idx]
        team = side["pokemon"]
        active_set = {i for i in side.get("active", []) if i is not None}

        # ── 1. Species reveal via switch/drag entries ─────────────────
        for pokemon_id in _extract_switch_entries(log_entries):
            slot = _log_id_to_slot(pokemon_id, side)
            if slot is not None and 0 <= slot < 6:
                self._species[env_idx][slot] = True
                self._ever_active[env_idx][slot] = True

        # ── 2. Ability reveal ─────────────────────────────────────────
        for pokemon_id, ability, source in _extract_ability_entries(log_entries):
            if source == "endability":
                # Ability was suppressed — mark all active Pokémon's ability as revealed
                for slot in active_set:
                    if slot < 6:
                        self._ability[env_idx][slot] = True
            else:
                slot = _log_id_to_slot(pokemon_id, side)
                if slot is not None and 0 <= slot < 6:
                    self._ability[env_idx][slot] = True

        # ── 3. Item reveal ────────────────────────────────────────────
        for pokemon_id, item_name, source in _extract_item_entries(log_entries):
            slot = _log_id_to_slot(pokemon_id, side)
            if slot is not None and 0 <= slot < 6:
                self._item[env_idx][slot] = True

        # ── 4. Tera reveal ────────────────────────────────────────────
        for slot, poke in enumerate(team[:6]):
            if poke.get("terastallized", None) is not None:
                self._tera[env_idx][slot] = True

        # ── 5. Move reveal via PP tracking ────────────────────────────
        for slot, poke in enumerate(team[:6]):
            moves = poke.get("moves", [])
            for j, mv in enumerate(moves[:4]):
                if mv.get("pp", 0) < mv.get("maxpp", 100):
                    self._moves[env_idx][slot][j] = True

        # ── 6. Species: also reveal for any currently active Pokémon ──
        # (in case the switch entry was missed or it's turn 1)
        for slot in active_set:
            if slot < 6:
                self._species[env_idx][slot] = True
                self._ever_active[env_idx][slot] = True

    def get_state(self, env_idx: int) -> dict:
        """
        Return the current reveal state for a specific env.
        Used for storage in Transition.

        Returns a dict with keys:
          species: tuple[bool, ...] of length 6
          item:    tuple[bool, ...] of length 6
          ability: tuple[bool, ...] of length 6
          tera:    tuple[bool, ...] of length 6
          moves:   tuple[tuple[bool, ...], ...] of length 6, each length 4
        """
        return {
            "species": tuple(self._species[env_idx]),
            "item":    tuple(self._item[env_idx]),
            "ability": tuple(self._ability[env_idx]),
            "tera":    tuple(self._tera[env_idx]),
            "moves":   tuple(tuple(m) for m in self._moves[env_idx]),
        }

    @staticmethod
    def get_initial_state() -> dict:
        """Return a blank reveal state (nothing revealed)."""
        return {
            "species": (False,) * 6,
            "item":    (False,) * 6,
            "ability": (False,) * 6,
            "tera":    (False,) * 6,
            "moves":   ((False,) * 4,) * 6,
        }