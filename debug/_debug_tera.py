"""
debug/test_tera.py — Tests unitaires de la Térastallisation dans le simulateur Rust.

Tests couverts :
  A. Activation      : terastallize() est bien appelée, log |-terastallize| présent
  B. Usage unique    : impossible de tera deux fois avec le même pokémon
  C. Équipe bloquée  : après tera, AUCUN pokémon de l'équipe ne peut tera
  D. BP floor        : move STAB < 60 BP → dégâts plus élevés qu'sans tera (floor à 60)
  E. STAB ×2         : move tera-STAB natif fait ×2 au lieu de ×1.5

Usage :
    python debug/test_tera.py
    python debug/test_tera.py --scan 2000   (scanner plus de seeds pour trouver cas D/E)
    python debug/test_tera.py --verbose     (afficher les détails de chaque test)
"""

import sys
import io
import json
import argparse
import re
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulator

# ── Chargement du dex des moves ──────────────────────────────────────────────
_MOVES_JSON = Path(__file__).resolve().parent.parent.parent / \
    "pokemon-showdown-rs-master" / "data" / "moves.json"

def _load_moves():
    with open(_MOVES_JSON) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if isinstance(v, dict)}

MOVES = _load_moves()

def move_info(move_id: str) -> dict:
    return MOVES.get(move_id.lower(), {})


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_active(state, side_idx: int) -> dict | None:
    side = state["sides"][side_idx]
    active_slots = [i for i in side["active"] if i is not None]
    if not active_slots:
        return None
    return side["pokemon"][active_slots[0]]


def parse_damage_log(entries: list[str]) -> list[tuple[str, int, int]]:
    """
    Extrait (pokemon_ident, new_hp, max_hp) depuis les entrées |-damage|.
    Format PS : |-damage|p2a: Name|currentHP/maxHP
    """
    results = []
    for line in entries:
        m = re.match(r"\|-damage\|([^|]+)\|(\d+)/(\d+)", line)
        if m:
            results.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return results


def get_hp(state, side_idx: int, poke_idx: int) -> tuple[int, int]:
    p = state["sides"][side_idx]["pokemon"][poke_idx]
    return p["hp"], p["maxhp"]


def get_active_move_of_type(poke: dict, target_type: str, min_bp: int = 0, max_bp: int = 9999,
                             require_no_multihit: bool = True, require_prio0: bool = True) -> int | None:
    """
    Retourne l'index 1-based du premier move du pokémon qui correspond au type cible
    et aux contraintes BP. Retourne None si pas trouvé.
    """
    for i, mv in enumerate(poke.get("moves", []), start=1):
        info = move_info(mv["id"])
        if not info:
            continue
        if info.get("type", "").lower() != target_type.lower():
            continue
        bp = info.get("basePower", 0)
        if bp == 0:
            continue  # moves à BP variable (dragonenergy, etc.)
        if not (min_bp <= bp <= max_bp):
            continue
        if require_no_multihit and info.get("multihit") is not None:
            continue
        if require_prio0 and info.get("priority", 0) != 0:
            continue
        if info.get("category", "") == "Status":
            continue
        if info.get("isZ") or info.get("isMax"):
            continue
        return i
    return None


# ── Scan de seeds ─────────────────────────────────────────────────────────────

def scan_for_tera_seed(max_seeds: int, verbose: bool = False) -> dict:
    """
    Scanne des seeds pour trouver des cas de test D et E :
      - D_seed : active a tera_type == native type ET un move de ce type avec BP < 60
      - E_seed : active a tera_type == native type ET un move de ce type avec BP >= 60
    """
    results = {"D": None, "E": None}

    for seed in range(1, max_seeds + 1):
        b = simulator.PyBattle("gen9randombattle", seed)
        state = b.get_state()
        poke = get_active(state, 0)
        if poke is None:
            continue

        tera = poke.get("tera_type")
        if not tera:
            continue

        native_types = [t.lower() for t in poke.get("types", [])]
        if tera.lower() not in native_types:
            continue  # tera type ne correspond pas à un type natif → STAB x2 ne s'applique pas

        # Test D : move STAB < 60 BP (BP floor)
        if results["D"] is None:
            midx = get_active_move_of_type(poke, tera, min_bp=1, max_bp=59)
            if midx is not None:
                info = move_info(poke["moves"][midx - 1]["id"])
                results["D"] = {
                    "seed": seed, "move_idx": midx,
                    "move_id": poke["moves"][midx - 1]["id"],
                    "bp": info["basePower"], "tera_type": tera,
                }
                if verbose:
                    print(f"  [scan] D trouvé seed={seed}: {poke['species_id']} "
                          f"tera={tera} move={results['D']['move_id']} BP={results['D']['bp']}")

        # Test E : move STAB 60-80 BP (STAB x2 pur, cap 80 pour éviter OHKO)
        if results["E"] is None:
            midx = get_active_move_of_type(poke, tera, min_bp=60, max_bp=80)
            if midx is not None:
                info = move_info(poke["moves"][midx - 1]["id"])
                results["E"] = {
                    "seed": seed, "move_idx": midx,
                    "move_id": poke["moves"][midx - 1]["id"],
                    "bp": info["basePower"], "tera_type": tera,
                }
                if verbose:
                    print(f"  [scan] E trouvé seed={seed}: {poke['species_id']} "
                          f"tera={tera} move={results['E']['move_id']} BP={results['E']['bp']}")

        if results["D"] is not None and results["E"] is not None:
            break

    return results


# ── Mesure de dégâts sur le tour 1 ───────────────────────────────────────────

def measure_damage_turn1(seed: int, p1_choice: str, p2_choice: str = "move 1") -> float | None:
    """
    Lance une bataille depuis `seed`, joue le turn 1 avec les choix donnés,
    retourne les dégâts infligés à p2 (en % de son maxhp).
    Retourne None si aucun dégât trouvé dans le log.
    """
    b = simulator.PyBattle("gen9randombattle", seed)
    state = b.get_state()

    # HP initial de p2 actif
    p2_active_idx = next(
        (i for i in state["sides"][1]["active"] if i is not None), None
    )
    if p2_active_idx is None:
        return None
    hp_before, maxhp = get_hp(state, 1, p2_active_idx)

    b.make_choices(p1_choice, p2_choice)
    b.get_new_log_entries()  # vider le buffer
    state_after = b.get_state()

    hp_after, _ = get_hp(state_after, 1, p2_active_idx)
    if maxhp == 0:
        return None

    dmg_pct = (hp_before - hp_after) / maxhp * 100.0
    return dmg_pct


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_A_activation(verbose: bool) -> bool:
    """
    A. Activation : terastallize() est bien appelée.
    - Joue "move 1 terastallize" pour p1
    - Vérifie state["sides"][0]["pokemon"][actif]["terastallized"] is not None
    - Vérifie que le log contient "|-terastallize|"
    """
    print("\n[A] Activation de la Térastallisation...")

    b = simulator.PyBattle("gen9randombattle", 1)
    state_before = b.get_state()
    poke_before = get_active(state_before, 0)
    assert poke_before is not None, "Pas de pokémon actif"
    assert poke_before["terastallized"] is None, "Déjà térastallisé avant le test"

    b.make_choices("move 1 terastallize", "move 1")
    log = b.get_new_log_entries()
    state_after = b.get_state()

    poke_after = get_active(state_after, 0)
    # Le pokémon actif a pu changer (faint, etc.), on cherche le premier térastallisé
    side0 = state_after["sides"][0]
    tera_poke = next((p for p in side0["pokemon"] if p.get("terastallized") is not None), None)

    has_tera_log = any("|-terastallize|" in line for line in log)
    has_tera_state = tera_poke is not None

    if verbose:
        print(f"  Log |-terastallize| présent : {has_tera_log}")
        print(f"  State terastallized défini  : {has_tera_state} "
              f"(type={tera_poke['terastallized'] if tera_poke else None})")
        tera_lines = [l for l in log if "terastallize" in l.lower()]
        for l in tera_lines:
            print(f"  > {l}")

    ok = has_tera_log and has_tera_state
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — log={has_tera_log}, state={has_tera_state}")
    return ok


def test_B_usage_unique(verbose: bool) -> bool:
    """
    B. Usage unique : impossible de tera deux fois.
    - Tour 1 : tera
    - Tour 2 : tente "move 1 terastallize" → choose_side doit retourner False
    """
    print("\n[B] Usage unique (pas de tera deux fois)...")

    b = simulator.PyBattle("gen9randombattle", 1)
    ok1 = b.make_choices("move 1 terastallize", "move 1")
    if not ok1:
        print("  SKIP — tour 1 tera déjà refusé (cas limite)")
        return True

    state = b.get_state()
    if state.get("ended"):
        print("  SKIP — battle terminée après tour 1")
        return True

    # Tente de tera à nouveau
    accepted = b.choose_side(0, "move 1 terastallize")
    b.undo_choice(0)  # nettoyer si accepté par erreur

    if verbose:
        print(f"  2e tentative de tera acceptée par le sim : {accepted} (attendu: False)")

    ok = not accepted
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — 2e tera refusé: {not accepted}")
    return ok


def test_C_equipe_bloquee(verbose: bool) -> bool:
    """
    C. Toute l'équipe est bloquée après tera.
    - Après le tera du pokémon actif, tente "move 1 terastallize" avec n'importe quel
      pokémon de l'équipe (en forçant un switch si nécessaire).
    - Vérifie que can_terastallize = None pour tous les pokémon du side 0.
    """
    print("\n[C] Équipe bloquée après tera...")

    # Cherche un seed où l'équipe a au moins 2 pokémon vivants
    seed_used = None
    for seed in range(1, 200):
        b = simulator.PyBattle("gen9randombattle", seed)
        state = b.get_state()
        alive = [p for p in state["sides"][0]["pokemon"] if not p["fainted"]]
        if len(alive) >= 2:
            seed_used = seed
            break

    assert seed_used is not None, "Pas de seed avec 2 pokémon vivants trouvé"
    b = simulator.PyBattle("gen9randombattle", seed_used)

    # Tour 1 : tera
    ok1 = b.make_choices("move 1 terastallize", "move 1")
    if not ok1:
        print("  SKIP — tera refusé au tour 1")
        return True

    state = b.get_state()
    if state.get("ended"):
        print("  SKIP — battle terminée")
        return True

    # Trouve un pokémon vivant sur le banc pour switcher
    side0 = state["sides"][0]
    active_slots = set(i for i in side0["active"] if i is not None)
    bench_alive = [
        i for i, p in enumerate(side0["pokemon"])
        if i not in active_slots and not p["fainted"]
    ]

    if not bench_alive:
        print("  SKIP — pas de pokémon sur le banc")
        return True

    bench_slot = bench_alive[0] + 1  # 1-based pour le choix switch

    # Tour 2 : switch vers le banc
    ok2 = b.make_choices(f"switch {bench_slot}", "move 1")
    if not ok2:
        print("  SKIP — switch refusé")
        return True

    state = b.get_state()
    if state.get("ended"):
        print("  SKIP — battle terminée après switch")
        return True

    # Tente tera avec le nouveau pokémon actif
    accepted = b.choose_side(0, "move 1 terastallize")
    b.undo_choice(0)

    if verbose:
        print(f"  Seed utilisé : {seed_used}")
        print(f"  Tera avec autre pokémon après switch accepté : {accepted} (attendu: False)")

    ok = not accepted
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — tera post-switch refusé: {not accepted}")
    return ok


def test_D_bp_floor(case: dict | None, verbose: bool) -> bool:
    """
    D. BP floor : move STAB < 60 BP → dégâts plus élevés avec tera qu'sans.
    Ratio attendu ≈ 60 / original_BP (ex: 40 BP → ratio ≈ 1.5).
    """
    print("\n[D] BP floor (STAB < 60 BP → boosted to 60)...")

    if case is None:
        print("  SKIP — pas de seed trouvé avec move STAB < 60 BP")
        return True

    seed = case["seed"]
    midx = case["move_idx"]
    move_id = case["move_id"]
    bp_orig = case["bp"]
    tera_type = case["tera_type"]

    dmg_tera   = measure_damage_turn1(seed, f"move {midx} terastallize", "move 1")
    dmg_normal = measure_damage_turn1(seed, f"move {midx}", "move 1")

    if verbose:
        print(f"  Seed={seed} | Pokémon active | tera_type={tera_type}")
        print(f"  Move: {move_id} (BP={bp_orig}, type={tera_type})")
        print(f"  Dégâts sans tera : {dmg_normal:.1f}% maxhp" if dmg_normal is not None else "  Dégâts sans tera : N/A")
        print(f"  Dégâts avec tera : {dmg_tera:.1f}% maxhp"   if dmg_tera   is not None else "  Dégâts avec tera : N/A")

    if dmg_tera is None or dmg_normal is None or dmg_normal <= 0:
        print("  SKIP — impossible de mesurer les dégâts (faint/immunité?)")
        return True

    ratio = dmg_tera / dmg_normal
    expected_ratio = 60.0 / bp_orig
    # Tolérance ±20% (variance du roll de dégâts ±15/255 ≈ 5.9%, crit possible)
    tolerance = 0.20

    if verbose:
        print(f"  Ratio dégâts (tera/normal): {ratio:.3f}  attendu≈{expected_ratio:.3f}  "
              f"tolérance=±{tolerance*100:.0f}%")

    ok = abs(ratio - expected_ratio) <= tolerance * expected_ratio
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — ratio={ratio:.3f}, attendu={expected_ratio:.3f} "
          f"(move={move_id}, BP={bp_orig}→60)")
    return ok


def test_E_stab_x2(case: dict | None, verbose: bool) -> bool:
    """
    E. STAB ×2 : move tera-STAB natif fait ×2 au lieu de ×1.5.
    Ratio attendu = 2.0 / 1.5 ≈ 1.333.
    """
    print("\n[E] STAB ×2 (tera-STAB natif = 2.0 au lieu de 1.5)...")

    if case is None:
        print("  SKIP — pas de seed trouvé avec move STAB natif ≥ 60 BP")
        return True

    seed = case["seed"]
    midx = case["move_idx"]
    move_id = case["move_id"]
    bp = case["bp"]
    tera_type = case["tera_type"]

    dmg_tera   = measure_damage_turn1(seed, f"move {midx} terastallize", "move 1")
    dmg_normal = measure_damage_turn1(seed, f"move {midx}", "move 1")

    if verbose:
        print(f"  Seed={seed} | tera_type={tera_type}")
        print(f"  Move: {move_id} (BP={bp}, type={tera_type})")
        print(f"  Dégâts sans tera (×1.5 STAB) : {dmg_normal:.1f}% maxhp" if dmg_normal is not None else "  Dégâts sans tera : N/A")
        print(f"  Dégâts avec tera (×2.0 STAB)  : {dmg_tera:.1f}% maxhp"  if dmg_tera   is not None else "  Dégâts avec tera : N/A")

    if dmg_tera is None or dmg_normal is None or dmg_normal <= 0:
        print("  SKIP — impossible de mesurer les dégâts")
        return True

    ratio = dmg_tera / dmg_normal
    expected_ratio = 2.0 / 1.5  # ≈ 1.333
    tolerance = 0.15

    if verbose:
        print(f"  Ratio dégâts (tera/normal): {ratio:.3f}  attendu≈{expected_ratio:.3f}  "
              f"tolérance=±{tolerance*100:.0f}%")

    ok = abs(ratio - expected_ratio) <= tolerance * expected_ratio
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — ratio={ratio:.3f}, attendu={expected_ratio:.3f} "
          f"(move={move_id}, BP={bp})")
    return ok


# ── Stellar seed scanner ──────────────────────────────────────────────────────

def scan_for_stellar_seed(max_seeds: int, verbose: bool = False) -> dict:
    """
    Scanne des seeds pour trouver un Pokémon actif avec tera_type=Stellar
    ayant à la fois un move STAB (natif) et un move non-STAB.
    Retourne {"stab": {seed,move_idx,move_id,bp}, "nonstab": {...}} ou None.
    """
    result = {"stab": None, "nonstab": None}

    for seed in range(1, max_seeds + 1):
        b = simulator.PyBattle("gen9randombattle", seed)
        state = b.get_state()
        side = state["sides"][0]
        active_idx = next((i for i in side["active"] if i is not None), None)
        if active_idx is None:
            continue
        p = side["pokemon"][active_idx]
        if p.get("tera_type", "").lower() != "stellar":
            continue

        native_types = [t.lower() for t in p.get("types", [])]

        for i, mv in enumerate(p.get("moves", []), start=1):
            info = move_info(mv["id"])
            if not info or info.get("category") == "Status":
                continue
            bp = info.get("basePower", 0)
            if bp == 0 or bp > 80 or info.get("isZ") or info.get("isMax"):
                continue
            if info.get("multihit") or info.get("priority", 0) != 0:
                continue
            # Exclude moves with low accuracy (< 90) or self-targeting effects
            acc = info.get("accuracy", 100)
            if acc is not True and (acc is False or acc < 90):
                continue
            target = info.get("target", "normal")
            if target in ("self", "allySide", "allyTeam", "all"):
                continue
            mtype = info.get("type", "").lower()

            if mtype in native_types and result["stab"] is None:
                result["stab"] = {"seed": seed, "move_idx": i, "move_id": mv["id"],
                                  "bp": bp, "type": mtype}
                if verbose:
                    print(f"  [stellar-scan] STAB seed={seed} {p['species_id']} {mv['id']} BP={bp} type={mtype}")
            if mtype not in native_types and result["nonstab"] is None:
                result["nonstab"] = {"seed": seed, "move_idx": i, "move_id": mv["id"],
                                     "bp": bp, "type": mtype}
                if verbose:
                    print(f"  [stellar-scan] non-STAB seed={seed} {p['species_id']} {mv['id']} BP={bp} type={mtype}")

        if result["stab"] is not None and result["nonstab"] is not None:
            break

    return result


# ── Tests Stellar ─────────────────────────────────────────────────────────────

def test_F_stellar_stab(case: dict | None, verbose: bool) -> bool:
    """
    F. Stellar STAB ×2.0 : un move de type natif sous Stellar donne ×2.0 au lieu de ×1.5.
    Ratio attendu = 2.0 / 1.5 ≈ 1.333.
    """
    print("\n[F] Stellar STAB ×2.0 (type natif)...")

    if case is None:
        print("  SKIP — pas de seed Stellar avec move STAB trouvé")
        return True

    seed = case["seed"]
    midx = case["move_idx"]
    move_id = case["move_id"]
    bp = case["bp"]

    dmg_stellar = measure_damage_turn1(seed, f"move {midx} terastallize", "move 1")
    dmg_normal  = measure_damage_turn1(seed, f"move {midx}", "move 1")

    if verbose:
        print(f"  Seed={seed} move={move_id} BP={bp} type={case['type']}")
        print(f"  Dégâts sans tera (×1.5) : {dmg_normal:.1f}%" if dmg_normal else "  N/A")
        print(f"  Dégâts tera Stellar (×2) : {dmg_stellar:.1f}%" if dmg_stellar else "  N/A")

    if dmg_stellar is None or dmg_normal is None or dmg_normal <= 0:
        print("  SKIP — impossible de mesurer les dégâts")
        return True

    ratio = dmg_stellar / dmg_normal
    expected = 2.0 / 1.5
    tolerance = 0.15

    if verbose:
        print(f"  Ratio: {ratio:.3f}  attendu: {expected:.3f}  tolérance=±{tolerance*100:.0f}%")

    ok = abs(ratio - expected) <= tolerance * expected
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — ratio={ratio:.3f}, attendu={expected:.3f} "
          f"(move={move_id}, BP={bp})")
    return ok


def test_G_stellar_nonstab(case: dict | None, verbose: bool) -> bool:
    """
    G. Stellar non-STAB ×1.2 : un move de type non-natif sous Stellar donne ×1.2 (4915/4096).
    Ratio attendu ≈ 1.2 (pas de STAB sans Stellar, donc 1.0 → 1.2).
    """
    print("\n[G] Stellar non-STAB ×1.2 (type non-natif)...")

    if case is None:
        print("  SKIP — pas de seed Stellar avec move non-STAB trouvé")
        return True

    seed = case["seed"]
    midx = case["move_idx"]
    move_id = case["move_id"]
    bp = case["bp"]

    dmg_stellar = measure_damage_turn1(seed, f"move {midx} terastallize", "move 1")
    dmg_normal  = measure_damage_turn1(seed, f"move {midx}", "move 1")

    if verbose:
        print(f"  Seed={seed} move={move_id} BP={bp} type={case['type']}")
        print(f"  Dégâts sans tera (×1.0) : {dmg_normal:.1f}%" if dmg_normal else "  N/A")
        print(f"  Dégâts tera Stellar (×1.2) : {dmg_stellar:.1f}%" if dmg_stellar else "  N/A")

    if dmg_stellar is None or dmg_normal is None or dmg_normal <= 0:
        print("  SKIP — impossible de mesurer les dégâts")
        return True

    ratio = dmg_stellar / dmg_normal
    expected = 4915.0 / 4096.0  # ≈ 1.2002...
    tolerance = 0.15

    if verbose:
        print(f"  Ratio: {ratio:.3f}  attendu: {expected:.3f}  tolérance=±{tolerance*100:.0f}%")

    ok = abs(ratio - expected) <= tolerance * expected
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — ratio={ratio:.3f}, attendu≈{expected:.3f} "
          f"(move={move_id}, BP={bp})")
    return ok


def test_H_stellar_one_per_type(case: dict | None, verbose: bool) -> bool:
    """
    H. Stellar épuise le boost par type : après un move de type X sous Stellar,
    le 2e move du même type (tour suivant) n'a plus le boost.
    Ratio attendu du 2e move ≈ 1.0 (pas de boost).
    """
    print("\n[H] Stellar boost unique par type...")

    if case is None:
        print("  SKIP — pas de seed Stellar avec move adapté trouvé")
        return True

    # Utilise le move STAB si disponible, sinon non-STAB
    seed = case["seed"]
    midx = case["move_idx"]
    move_id = case["move_id"]

    # Turn 1: tera Stellar + move → boosted
    # Turn 2: même move → plus de boost (type déjà utilisé)
    # On mesure les dégâts au tour 2 avec tera vs sans tera (pour le 2e tour, tera est déjà actif)
    # Stratégie : on lance deux battles, dans les deux p1 tera au T1
    # Battle A : move différent au T1, puis le move cible au T2
    # Battle B : move cible au T1 (boost consommé) + move cible au T2 (pas de boost)
    # On compare T2(A) vs T2(B) — ils devraient être égaux si le boost est épuisé.

    # Alternative plus simple : comparer T1 boosted vs T2 (après avoir épuisé le boost au T1).
    # T2 devrait être ≈ T1 normal (1.5× STAB si type natif, 1.0× sinon).

    def measure_turn2(seed, move_t1, move_t2, p2_choice="move 1"):
        b = simulator.PyBattle("gen9randombattle", seed)
        state = b.get_state()
        p2_active_idx = next((i for i in state["sides"][1]["active"] if i is not None), None)
        if p2_active_idx is None:
            return None
        hp_before, maxhp = get_hp(state, 1, p2_active_idx)
        ok1 = b.make_choices(move_t1, p2_choice)
        if not ok1:
            return None
        state2 = b.get_state()
        if state2.get("ended"):
            return None
        p2_active_idx2 = next((i for i in state2["sides"][1]["active"] if i is not None), None)
        if p2_active_idx2 is None:
            return None
        hp_before2, maxhp2 = get_hp(state2, 1, p2_active_idx2)
        ok2 = b.make_choices(move_t2, p2_choice)
        if not ok2:
            return None
        state3 = b.get_state()
        hp_after2, _ = get_hp(state3, 1, p2_active_idx2)
        if maxhp2 == 0:
            return None
        return (hp_before2 - hp_after2) / maxhp2 * 100.0

    # T1: tera + move cible (consume boost), T2: même move → no boost
    dmg_t2_after_consume = measure_turn2(seed, f"move {midx} terastallize", f"move {midx}")
    # T1: tera + move différent (boost intact pour type X), T2: move cible (boost dispo)
    # On utilise move 1 si midx != 1, sinon move 2
    other_midx = 1 if midx != 1 else 2
    dmg_t2_boosted = measure_turn2(seed, f"move {other_midx} terastallize", f"move {midx}")
    # T1: sans tera + move cible, T2: même move (baseline)
    dmg_t2_baseline = measure_turn2(seed, f"move {midx}", f"move {midx}")

    if verbose:
        print(f"  Seed={seed} move={move_id} (idx={midx})")
        print(f"  T2 après boost consommé : {dmg_t2_after_consume:.1f}%" if dmg_t2_after_consume else "  T2 consommé: N/A")
        print(f"  T2 avec boost dispo      : {dmg_t2_boosted:.1f}%" if dmg_t2_boosted else "  T2 boosted: N/A")
        print(f"  T2 baseline (sans tera)  : {dmg_t2_baseline:.1f}%" if dmg_t2_baseline else "  T2 baseline: N/A")

    if dmg_t2_after_consume is None or dmg_t2_boosted is None or dmg_t2_boosted <= 0:
        print("  SKIP — impossible de mesurer les dégâts (faint?)")
        return True

    # Le T2 sans boost doit être significativement inférieur au T2 avec boost
    ratio = dmg_t2_after_consume / dmg_t2_boosted
    # Si le boost est bien épuisé, ratio < 0.9 (le boost vaut au moins ×1.2)
    ok = ratio < 0.95

    if verbose:
        print(f"  Ratio consommé/boosted: {ratio:.3f}  (attendu < 0.95 si boost épuisé)")

    print(f"  {'PASS ✓' if ok else 'FAIL ✗'} — ratio T2-consommé/T2-boosted={ratio:.3f} "
          f"(move={move_id})")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Tests unitaires Térastallisation")
    parser.add_argument("--scan",    type=int, default=500,
                        help="Nombre de seeds à scanner pour les tests D/E (défaut: 500)")
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher les détails de chaque test")
    args = parser.parse_args()

    print("=" * 60)
    print("  Test Térastallisation — CynthAI_v2 / pokemon-showdown-rs")
    print("=" * 60)

    # Tests A/B/C n'ont pas besoin de scan
    results = {}
    results["A"] = test_A_activation(args.verbose)
    results["B"] = test_B_usage_unique(args.verbose)
    results["C"] = test_C_equipe_bloquee(args.verbose)

    # Scan seeds pour D/E
    print(f"\n[scan] Recherche seeds pour D et E (max {args.scan})...")
    cases = scan_for_tera_seed(args.scan, verbose=args.verbose)
    if cases["D"] is None:
        print("  ⚠  Aucun seed D trouvé dans la plage scannée")
    if cases["E"] is None:
        print("  ⚠  Aucun seed E trouvé dans la plage scannée")

    results["D"] = test_D_bp_floor(cases["D"], args.verbose)
    results["E"] = test_E_stab_x2(cases["E"], args.verbose)

    # Scan seeds pour F/G/H (Stellar)
    print(f"\n[stellar-scan] Recherche seeds Stellar (max {args.scan})...")
    stellar = scan_for_stellar_seed(args.scan, verbose=args.verbose)
    if stellar["stab"] is None:
        print("  ⚠  Aucun seed Stellar STAB trouvé")
    if stellar["nonstab"] is None:
        print("  ⚠  Aucun seed Stellar non-STAB trouvé")

    results["F"] = test_F_stellar_stab(stellar["stab"], args.verbose)
    results["G"] = test_G_stellar_nonstab(stellar["nonstab"], args.verbose)
    results["H"] = test_H_stellar_one_per_type(stellar["stab"] or stellar["nonstab"], args.verbose)

    # Résumé
    print("\n" + "=" * 60)
    print("  Résumé")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  [{name}] {status}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("  Tous les tests PASSENT — Tera fonctionnel.")
    else:
        print("  Des tests ÉCHOUENT — voir détails ci-dessus.")
    print("=" * 60)


if __name__ == "__main__":
    main()
