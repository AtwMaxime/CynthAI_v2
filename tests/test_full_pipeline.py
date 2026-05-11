"""
CynthAI_v2 Full Pipeline Test — end-to-end smoke test without a live battle server.

Tests:
  1. State encoder — encode_pokemon / encode_field on dummy dicts
  2. Collation — PokemonBatch + FieldBatch shapes
  3. PokemonEmbeddings forward
  4. BattleBackbone encode + act
  5. ActionEncoder forward
  6. CynthAIAgent full forward pass — output shapes + no NaN
  7. PredictionHeads build_targets + compute_loss
  8. Training losses (compute_losses)
  9. Checkpoint save + load round-trip

Run from the CynthAI_v2 directory:
    .venv\\Scripts\\python.exe tests/test_full_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import tempfile
import os

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {name}{suffix}")
    return ok


def section(title: str) -> None:
    print(f"\n-- {title} {'-' * (50 - len(title))}")


# ── 1. State encoder ──────────────────────────────────────────────────────────

section("1. State encoder")

from env.state_encoder import (
    encode_pokemon, encode_field,
    PokemonFeatures, FieldFeatures,
    N_SPECIES, N_MOVES, N_ITEMS, N_ABILITIES, N_TYPES,
    N_VOLATILES, FIELD_DIM,
)

dummy_poke = {
    "species_id":        "pikachu",
    "level":             50,
    "hp":                50,
    "maxhp":             100,
    "types":             ["electric"],
    "tera_type":         "electric",
    "terastallized":     False,
    "base_stats":        {"hp": 35, "atk": 55, "def": 40, "spa": 50, "spd": 50, "spe": 90},
    "stats":             {"atk": 80, "def": 55, "spa": 65, "spd": 65, "spe": 120},
    "is_predicted":      False,
    "boosts":            {"atk": 1, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0},
    "item":              "",
    "ability":           "static",
    "status":            "",
    "moves":             [
        {"id": "thunderbolt", "pp": 15, "maxpp": 24, "disabled": False},
        {"id": "quickattack", "pp": 30, "maxpp": 30, "disabled": False},
        {"id": "ironhead",    "pp": 15, "maxpp": 15, "disabled": False},
        {"id": "voltswitch",  "pp": 20, "maxpp": 32, "disabled": False},
    ],
    "volatiles":         {},
    "is_active":         True,
    "fainted":           False,
    "trapped":           False,
    "force_switch_flag": False,
    "revealed":          True,
}

feat = encode_pokemon(dummy_poke)
check("encode_pokemon returns PokemonFeatures",   isinstance(feat, PokemonFeatures))
check("hp_ratio = 0.5",                           abs(feat.hp_ratio - 0.5) < 1e-6)
check("volatiles length = N_VOLATILES",           len(feat.volatiles) == N_VOLATILES,
      f"{len(feat.volatiles)} vs {N_VOLATILES}")
check("move_indices length = 4",                  len(feat.move_indices) == 4)
check("base_stats length = 5",                    len(feat.base_stats) == 5)

dummy_field = {
    "field": {
        "weather":        "sunnyday",
        "terrain":        "electricterrain",
        "pseudo_weather": ["trickroom"],
    },
    "sides": [
        {"side_conditions": {"stealthrock": 1}, "pokemon_left": 5, "total_fainted": 1},
        {"side_conditions": {"spikes": 2},       "pokemon_left": 4, "total_fainted": 2},
    ],
}

ff = encode_field(dummy_field)
check("encode_field returns FieldFeatures",       isinstance(ff, FieldFeatures))
check("weather set",                              ff.weather[1] == 1.0)
check("terrain set",                              ff.terrain[1] == 1.0)
check("pseudo_weather trickroom",                 ff.pseudo_weather[2] == 1.0)
check("side0 stealthrock = 1.0",                  ff.side0.conditions[0] == 1.0)
check("side1 spikes = 2/3",                       abs(ff.side1.conditions[1] - 2/3) < 1e-6)
check("side0 pokemon_left = 5/6",                 abs(ff.side0.pokemon_left - 5/6) < 1e-6)
check("side1 total_fainted = 2/6",                abs(ff.side1.total_fainted - 2/6) < 1e-6)


# ── 2. Collation ──────────────────────────────────────────────────────────────

section("2. Collation — PokemonBatch + FieldBatch")

from model.embeddings import (
    PokemonBatch, FieldBatch,
    collate_features, collate_field_features,
    TOKEN_DIM, D_SPECIES, D_ITEM, D_ABILITY,
)
from model.backbone import K_TURNS

B = 2
K = K_TURNS   # 4

# Build K*12 features per batch item
poke_feat_list = [[encode_pokemon(dummy_poke) for _ in range(K * 12)] for _ in range(B)]
pb = collate_features(poke_feat_list)
check("PokemonBatch species_idx shape",   tuple(pb.species_idx.shape) == (B, K * 12))
check("PokemonBatch move_idx shape",      tuple(pb.move_idx.shape)    == (B, K * 12, 4))
check("PokemonBatch scalars shape",       tuple(pb.scalars.shape)     == (B, K * 12, 222))

field_feat_list = [encode_field(dummy_field) for _ in range(B * K)]
fb = collate_field_features(field_feat_list)
check("FieldBatch shape",                 tuple(fb.field.shape) == (B * K, FIELD_DIM),
      f"{tuple(fb.field.shape)} vs ({B * K}, {FIELD_DIM})")


# ── 3. PokemonEmbeddings ──────────────────────────────────────────────────────

section("3. PokemonEmbeddings")

from model.embeddings import PokemonEmbeddings

emb = PokemonEmbeddings()
emb.eval()
with torch.no_grad():
    tok = emb(pb)
check("output shape [B, K*12, TOKEN_DIM]",  tuple(tok.shape) == (B, K * 12, TOKEN_DIM),
      str(tuple(tok.shape)))
check("no NaN in embeddings",               not tok.isnan().any().item())


# ── 4. BattleBackbone ─────────────────────────────────────────────────────────

section("4. BattleBackbone encode + act")

from model.backbone import BattleBackbone, D_MODEL

backbone = BattleBackbone()
backbone.eval()

field_t = fb.field.reshape(B, K, FIELD_DIM)

with torch.no_grad():
    cur_tok, value = backbone.encode(tok, field_t)

check("current_tokens shape [B, 13, D_MODEL]",  tuple(cur_tok.shape) == (B, 13, D_MODEL),
      str(tuple(cur_tok.shape)))
check("value shape [B, 1]",                     tuple(value.shape)   == (B, 1))
check("no NaN in backbone output",              not cur_tok.isnan().any().item())

action_mask = torch.zeros(B, 13, dtype=torch.bool)
action_mask[:, 4:8] = True   # mask mechanic moves

# dummy action embeds
act_emb = torch.randn(B, 13, D_MODEL)
with torch.no_grad():
    logits = backbone.act(act_emb, cur_tok, action_mask)
check("action logits shape [B, 13]",  tuple(logits.shape) == (B, 13))
check("masked logits are -1e9",       (logits[:, 4:8] < -1e8).all().item())


# ── 5. ActionEncoder ──────────────────────────────────────────────────────────

section("5. ActionEncoder")

from env.action_space import ActionEncoder, MECH_NONE

ae = ActionEncoder(move_embed=emb.move_embed, type_embed=emb.type_embed)
ae.eval()

move_idx       = pb.move_idx[:, K * 12 - 12, :]   # [B, 4]
pp_ratio       = torch.rand(B, 4)
move_disabled  = torch.zeros(B, 4)
mechanic_id    = torch.zeros(B, dtype=torch.long)
mech_type_idx  = torch.zeros(B, dtype=torch.long)
bench_tokens   = cur_tok[:, 1:6, :]

with torch.no_grad():
    action_embeds = ae(
        active_token      = cur_tok[:, 0, :],
        move_idx          = move_idx,
        pp_ratio          = pp_ratio,
        move_disabled     = move_disabled,
        bench_tokens      = bench_tokens,
        mechanic_id       = mechanic_id,
        mechanic_type_idx = mech_type_idx,
    )
check("action embeds shape [B, 13, D_MODEL]",  tuple(action_embeds.shape) == (B, 13, D_MODEL),
      str(tuple(action_embeds.shape)))


# ── 6. CynthAIAgent full forward ──────────────────────────────────────────────

section("6. CynthAIAgent full forward")

from model.agent import CynthAIAgent

agent = CynthAIAgent()
agent.eval()

n_params = sum(p.numel() for p in agent.parameters())
check("parameter count > 1M",  n_params > 1_000_000, f"{n_params:,}")

with torch.no_grad():
    out = agent(
        poke_batch        = pb,
        field_tensor      = field_t,
        move_idx          = move_idx,
        pp_ratio          = pp_ratio,
        move_disabled     = move_disabled,
        mechanic_id       = mechanic_id,
        mechanic_type_idx = mech_type_idx,
        action_mask       = action_mask,
    )

check("action_logits shape [B, 13]",  tuple(out.action_logits.shape) == (B, 13))
check("value shape [B, 1]",           tuple(out.value.shape)         == (B, 1))
check("no NaN in logits",             not out.action_logits.isnan().any().item())
check("no NaN in value",              not out.value.isnan().any().item())
check("log_probs sum to ~0",
      (out.log_probs[:, ~action_mask[0]].exp().sum(dim=-1) - 1.0).abs().max().item() < 1e-4)


# ── 7. PredictionHeads ────────────────────────────────────────────────────────

section("7. PredictionHeads build_targets + compute_loss")

from model.prediction_heads import PredictionHeads
from model.embeddings import N_SCALARS

# slice opponent current-turn tokens from poke_batch
from model.backbone import K_TURNS as K
_OPP_CUR = slice((K - 1) * 12 + 6, K * 12)
opp_batch = PokemonBatch(
    species_idx = pb.species_idx[:, _OPP_CUR],
    type1_idx   = pb.type1_idx[:, _OPP_CUR],
    type2_idx   = pb.type2_idx[:, _OPP_CUR],
    tera_idx    = pb.tera_idx[:, _OPP_CUR],
    item_idx    = pb.item_idx[:, _OPP_CUR],
    ability_idx = pb.ability_idx[:, _OPP_CUR],
    move_idx    = pb.move_idx[:, _OPP_CUR, :],
    scalars     = pb.scalars[:, _OPP_CUR, :],
)

targets = PredictionHeads.build_targets(opp_batch)
with torch.no_grad():
    loss_dict = PredictionHeads.compute_loss(out.pred_logits, *targets)

check("pred loss dict has 'total'",  "total" in loss_dict)
check("total pred loss is scalar",   loss_dict["total"].shape == torch.Size([]))
check("pred loss is finite",         loss_dict["total"].isfinite().item())


# ── 8. Training losses ────────────────────────────────────────────────────────

section("8. compute_losses")

from training.losses import compute_losses

T = 4
log_prob_old = torch.randn(T)
actions      = torch.randint(0, 13, (T,))
advantages   = torch.randn(T)
returns      = torch.randn(T)
values       = torch.randn(T, 1)
mask_t       = torch.zeros(T, 13, dtype=torch.bool)
pred_loss_t  = torch.tensor(0.5)

logits_new = torch.randn(T, 13)
logits_new.masked_fill_(mask_t, float("-inf"))

losses = compute_losses(
    logits_new   = logits_new,
    log_prob_old = log_prob_old,
    actions      = actions,
    advantages   = advantages,
    returns      = returns,
    values       = values,
    action_mask  = mask_t,
    pred_loss    = pred_loss_t,
)

for key in ("policy", "value", "entropy", "pred", "total"):
    check(f"losses['{key}'] is finite",  losses[key].isfinite().item())


# ── 9. Checkpoint round-trip ──────────────────────────────────────────────────

section("9. Checkpoint save + load")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "test_ckpt.pt")
    torch.save({"update": 1, "model": agent.state_dict()}, path)

    ckpt   = torch.load(path, map_location="cpu", weights_only=True)
    agent2 = CynthAIAgent()
    agent2.load_state_dict(ckpt["model"])
    check("checkpoint round-trip",  ckpt["update"] == 1)

    # verify weights are identical
    p1 = next(agent.parameters())
    p2 = next(agent2.parameters())
    check("weights match after reload",  torch.allclose(p1, p2))


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n-- Done --------------------------------------------------------------")
