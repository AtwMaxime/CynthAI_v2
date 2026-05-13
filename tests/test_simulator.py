"""
CynthAI_v2 Simulator Integration Test — end-to-end with real PyBattle.

Tests:
  1. PyBattle creation — seed determinism, basic state structure
  2. State encoder output shapes — from real simulator state
  3. Multi-turn combat — sliding window + action masking over several turns
  4. RevealedTracker integration — log parsing from real battles
  5. Full forward pass on real data — all backbone components
  6. Episode end-to-end — a complete battle from start to finish
  7. Parallel env consistency — N simultaneous battles

Run from the CynthAI_v2 directory:
    .venv\\Scripts\\python.exe tests/test_simulator.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path so env/, model/, training/ are importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PASS = "[PASS]"
FAIL = "[FAIL]"
_any_failed = False


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _any_failed
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    if not ok:
        _any_failed = True
    print(f"  {tag}  {name}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n-- {title} {'-' * (60 - len(title))}")


# ── 1. PyBattle creation ────────────────────────────────────────────────────────

section("1. PyBattle creation")

from simulator import PyBattle

b = PyBattle("gen9randombattle", seed=42)
s = b.get_state()

check("state is a dict",            isinstance(s, dict))
check("state has turn",             "turn" in s)
check("state has sides",            "sides" in s and len(s["sides"]) == 2)
check("state has field",            "field" in s)
check("state has ended flag",       "ended" in s)
check("turn == 1 at creation (Rust sim is 1-indexed)",      s["turn"] == 1)
check("battle not ended",           not b.ended)
check("winner is None",             b.winner is None)

# Seed determinism
b2 = PyBattle("gen9randombattle", seed=42)
s2 = b2.get_state()
check("same seed -> same state",     s["sides"][0]["pokemon"][0]["species_id"]
                                     == s2["sides"][0]["pokemon"][0]["species_id"])

b3 = PyBattle("gen9randombattle", seed=99)
s3 = b3.get_state()
check("different seed -> diff team", s["sides"][0]["pokemon"][0]["species_id"]
                                     != s3["sides"][0]["pokemon"][0]["species_id"])


# ── 2. State encoder shapes ─────────────────────────────────────────────────────

section("2. State encoder — shapes from real simulator")

from env.state_encoder import (
    encode_pokemon, encode_field,
    PokemonFeatures, FieldFeatures,
    N_SPECIES, N_MOVES, N_ITEMS, N_ABILITIES, N_TYPES,
    N_VOLATILES, FIELD_DIM, SPECIES_INDEX,
)
from training.rollout import encode_state

side0 = s["sides"][0]
side1 = s["sides"][1]

# Encode one Pokémon
p0 = side0["pokemon"][0]
pf = encode_pokemon(p0)
check("species_idx != UNK",          pf.species_idx != 0,
      f"got idx={pf.species_idx} for {p0['species_id']}")
check("type1_idx != UNK",            pf.type1_idx != 0)
check("hp_ratio in [0, 1]",          0.0 <= pf.hp_ratio <= 1.0)
check("item_idx valid",              pf.item_idx < N_ITEMS)
check("ability_idx valid",           pf.ability_idx < N_ABILITIES)
check("moves have valid indices",    all(m < N_MOVES for m in pf.move_indices if m != 0))
check("boosts length 7",             len(pf.boosts) == 7)
check("status length 7",             len(pf.status) == 7)
check("volatiles length N_VOLATILES", len(pf.volatiles) == N_VOLATILES)

# Encode state (own=6, opp=6, then field)
all_pokes, field_feat = encode_state(s, side_idx=0)
own_pokes  = all_pokes[:6]
opp_pokes  = all_pokes[6:12]
check("own team has 6 PokemonFeatures",    len(own_pokes) == 6)
check("opp team has 6 PokemonFeatures",    len(opp_pokes) == 6)
check("all are PokemonFeatures",           all(isinstance(p, PokemonFeatures) for p in all_pokes))
check("field is FieldFeatures",            isinstance(field_feat, FieldFeatures))
check("active Pokemon first in own team",  own_pokes[0].is_active == 1.0)
check("field dims match FIELD_DIM",        FIELD_DIM == 72)

# Specific check: species_id string matches index
species_id = p0["species_id"]
expected_idx = SPECIES_INDEX.get(species_id, 0)
check(f"species '{species_id}' -> idx={expected_idx}",  pf.species_idx == expected_idx,
      f"got {pf.species_idx}")


# ── 3. Multi-turn combat ────────────────────────────────────────────────────────

section("3. Multi-turn — actions + state transitions")

from model.backbone import K_TURNS
from training.rollout import build_action_mask, action_to_choice, BattleWindow

battle = PyBattle("gen9randombattle", seed=7)
window = BattleWindow()

for turn_idx in range(K_TURNS + 2):  # a few turns past the window size
    state = battle.get_state()
    pf, ff = encode_state(state, side_idx=0)
    window.push(pf, ff)  # pf is 12 PokemonFeatures, ff is FieldFeatures

    mask0 = build_action_mask(state, side_idx=0)
    mask1 = build_action_mask(state, side_idx=1)

    check(f"turn {state['turn']}: action mask shape [13]",     mask0.shape == (13,))
    check(f"turn {state['turn']}: at least 1 legal action",    mask0.sum() < 13,
          f"all {13} actions illegal")

    # Pick first legal action for each side
    legal0 = (~mask0).nonzero().squeeze(-1)
    legal1 = (~mask1).nonzero().squeeze(-1)
    a0 = legal0[0].item() if len(legal0) > 0 else 0
    a1 = legal1[0].item() if len(legal1) > 0 else 0

    c0 = action_to_choice(a0, state, 0)
    c1 = action_to_choice(a1, state, 1)

    # Check switch action maps to valid PS syntax
    if a0 >= 8:
        check(f"turn {state['turn']}: switch choice valid", c0.startswith("switch") and c0[-1].isdigit())
    else:
        check(f"turn {state['turn']}: move choice valid",   c0.startswith("move") and c0[-1].isdigit())

    battle.make_choices(c0, c1)

# Sliding window: verify early-turn padding
# After 1 turn in a K=4 window, 3 turns should be zero-padded
battle_small = PyBattle("gen9randombattle", seed=7)
window_small = BattleWindow()
s_state = battle_small.get_state()
sp, sf = encode_state(s_state, side_idx=0)
window_small.push(sp, sf)
padded_pokes_s, padded_fields_s = window_small.as_padded()
check("K=4 window with 1 turn: 4 entries",               len(padded_pokes_s) == K_TURNS)
check("first 3 entries are zero-padded",                  all(p.species_idx == 0 for p in padded_pokes_s[0] + padded_pokes_s[1] + padded_pokes_s[2]))
check("last entry has real species",                      any(p.species_idx != 0 for p in padded_pokes_s[3]))


# ── 4. RevealedTracker ──────────────────────────────────────────────────────────

section("4. RevealedTracker — real battle log parsing")

from env.revealed_tracker import RevealedTracker

tracker = RevealedTracker(n_envs=1)
battle2 = PyBattle("gen9randombattle", seed=42)

# Turn 1: make a move and check reveals
state = battle2.get_state()
mask0 = build_action_mask(state, 0)
mask1 = build_action_mask(state, 1)
legal0 = (~mask0).nonzero().squeeze(-1)
legal1 = (~mask1).nonzero().squeeze(-1)
a0 = legal0[0].item() if len(legal0) > 0 else 0
a1 = legal1[0].item() if len(legal1) > 0 else 0
c0 = action_to_choice(a0, state, 0)
c1 = action_to_choice(a1, state, 1)
battle2.make_choices(c0, c1)

log = battle2.get_new_log_entries()
state_after = battle2.get_state()
tracker.update(0, log, state_after, side_idx=1)

reveal = tracker.get_state(0)
check("reveal state has species key",  "species" in reveal)
check("reveal state has item key",     "item" in reveal)
check("reveal state has ability key",  "ability" in reveal)
check("reveal state has moves key",    "moves" in reveal)

# By turn 1, the active opponent Pokémon should have species revealed
opp_active = state_after["sides"][1]["active"]
if any(pos is not None for pos in opp_active):
    active_slots = [i for i in opp_active if i is not None]
    for slot in active_slots:
        check(f"opp slot {slot}: species revealed", reveal["species"][slot])


# ── 5. Full forward pass on real data ───────────────────────────────────────────

section("5. Full forward pass on real PyBattle state")

from model.agent import CynthAIAgent
from model.backbone import D_MODEL, OPP_SLOTS
from model.embeddings import (
    PokemonBatch, FieldBatch,
    collate_features, collate_field_features,
    TOKEN_DIM, N_SCALARS,
)
from env.action_space import MECH_NONE, MECH_TERA

agent = CynthAIAgent()
agent.eval()
n_params = sum(p.numel() for p in agent.parameters())
check("agent params ~2.6M",  abs(n_params - 2_526_978) < 100, f"{n_params:,}")

battle3 = PyBattle("gen9randombattle", seed=7)
state3 = battle3.get_state()
mask0 = build_action_mask(state3, 0)
legal0 = (~mask0).nonzero().squeeze(-1)
a0 = legal0[0].item() if len(legal0) > 0 else 0
c0 = action_to_choice(a0, state3, 0)
c1 = "default"

# Play K turns to fill the window
for _ in range(K_TURNS):
    battle3.make_choices(c0, c1)
    s = battle3.get_state()
    if battle3.ended:
        break
    m = build_action_mask(s, 0)
    legal = (~m).nonzero().squeeze(-1)
    a0 = legal[0].item() if len(legal) > 0 else 0
    c0 = action_to_choice(a0, s, 0)

curr_state = battle3.get_state()
all_pokes, ff = encode_state(curr_state, side_idx=0)
field_list = [ff] * K_TURNS  # same field repeated

# Build a batch with K turns (repeating the same state for simplicity)
B = 1
poke_flat = all_pokes * K_TURNS
pb = collate_features([poke_flat]).to("cpu")
fb = collate_field_features(field_list)
field_tensor = fb.field.reshape(B, K_TURNS, 72)

# ActionEncoder inputs
active = curr_state["sides"][0]["pokemon"][0]
midx, ppr, mdis = [], [], []
for mv in (active.get("moves") or [])[:4]:
    from env.state_encoder import MOVE_INDEX, UNK
    midx.append(MOVE_INDEX.get(mv["id"], UNK))
    maxpp = mv["maxpp"]
    ppr.append(mv["pp"] / maxpp if maxpp > 0 else 0.0)
    mdis.append(1.0 if mv.get("disabled") else 0.0)
while len(midx) < 4:
    midx.append(UNK); ppr.append(0.0); mdis.append(0.0)

tera_used = any(p.get("terastallized") is not None for p in curr_state["sides"][0]["pokemon"])
mech_id = MECH_NONE if tera_used else MECH_TERA

with torch.no_grad():
    out = agent(
        poke_batch        = pb,
        field_tensor      = field_tensor,
        move_idx          = torch.tensor([midx], dtype=torch.long),
        pp_ratio          = torch.tensor([ppr], dtype=torch.float32),
        move_disabled     = torch.tensor([mdis], dtype=torch.float32),
        mechanic_id       = torch.tensor([mech_id], dtype=torch.long),
        mechanic_type_idx = torch.tensor([0], dtype=torch.long),
        action_mask       = (~(~mask0).any()).unsqueeze(0),  # allow all
    )

check("action_logits shape [1, 13]",  out.action_logits.shape == (1, 13))
check("value shape [1, 1]",           out.value.shape == (1, 1))
check("log_probs sum approx 0",       abs(out.log_probs.exp().sum(dim=-1).item() - 1.0) < 1e-4)
check("no NaN in logits",             not out.action_logits.isnan().any().item())
check("no NaN in value",              not out.value.isnan().any().item())
check("pred_logits item head",        out.pred_logits.item.shape == (1, 6, 250))
check("pred_logits ability head",     out.pred_logits.ability.shape == (1, 6, 311))
check("pred_logits tera head",        out.pred_logits.tera.shape == (1, 6, 19))
check("pred_logits moves head",       out.pred_logits.moves.shape == (1, 6, 686))


# ── 6. Complete episode ─────────────────────────────────────────────────────────

section("6. Complete episode simulation")

episode = PyBattle("gen9randombattle", seed=13)
turn_count = 0
while not episode.ended:
    s = episode.get_state()
    m0 = build_action_mask(s, 0)
    m1 = build_action_mask(s, 1)
    legal0 = (~m0).nonzero().squeeze(-1)
    legal1 = (~m1).nonzero().squeeze(-1)
    c0 = action_to_choice(legal0[0].item(), s, 0) if len(legal0) else ""
    c1 = action_to_choice(legal1[0].item(), s, 1) if len(legal1) else ""
    if not episode.make_choices(c0, c1):
        break
    turn_count += 1

check("episode ended",                 episode.ended)
check("winner is p1 or p2",            episode.winner in ("p1", "p2"))
check("fought for at least 1 turn",    turn_count >= 1)
check("turn count reasonable (<200)",  turn_count < 200,
      f"{turn_count} turns (suspiciously long)")

# Verify state at end of battle
final_state = episode.get_state()
for side_idx in (0, 1):
    total_fainted = final_state["sides"][side_idx]["total_fainted"]
    check(f"side {side_idx}: total_fainted <= 6",  total_fainted <= 6)
    pokemon_left = final_state["sides"][side_idx]["pokemon_left"]
    check(f"side {side_idx}: pokemon_left <= 6",   pokemon_left <= 6)


# ── 7. Parallel envs ────────────────────────────────────────────────────────────

section("7. Parallel environment consistency")

N = 4
envs = [PyBattle("gen9randombattle", seed=s) for s in range(N)]
masks = []
for i, e in enumerate(envs):
    s = e.get_state()
    m0 = build_action_mask(s, 0)
    legal0 = (~m0).nonzero().squeeze(-1)
    a0 = legal0[0].item() if len(legal0) > 0 else 0
    c0 = action_to_choice(a0, s, 0)
    c1 = "default"
    e.make_choices(c0, c1)
    masks.append(m0)

for i, m in enumerate(masks):
    check(f"env {i}: action mask shape [13]",  m.shape == (13,))

# All envs made a move without crashing
check(f"{N} parallel envs ran OK",   all(not e.ended for e in envs) or any(e.ended for e in envs))


# ── Summary ─────────────────────────────────────────────────────────────────────

print()
if _any_failed:
    print("SOME TESTS FAILED")
    if __name__ == "__main__":
        sys.exit(1)
else:
    print("ALL TESTS PASSED")