"""
CynthAI_v2 Full Pipeline Test — end-to-end smoke test without a live battle server.

Tests:
  1. State encoder — encode_pokemon / encode_field on dummy dicts
  2. Collation — PokemonBatch + FieldBatch shapes
  3. PokemonEmbeddings forward
  4. BattleBackbone encode + act (with CLS token)
  5. ActionEncoder forward
  6. CynthAIAgent full forward — critic_n_layers=0 (MLP on CLS)
  7. CynthAIAgent full forward — critic_n_layers=2 (Transformer value head)
  8. ValueHead standalone forward
  9. PredictionHeads build_targets + compute_loss (with pos_weight BCE)
 10. move_recall metric correctness (sum not any)
 11. Training losses (compute_losses)
 12. Checkpoint round-trip — critic_n_layers=0
 13. Checkpoint round-trip — strict=False (architecture change simulation)
 14. Gradient flow — detached vs non-detached critic

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
n_passed = 0
n_failed = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global n_passed, n_failed
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {name}{suffix}")
    if ok:
        n_passed += 1
    else:
        n_failed += 1
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

poke_feat_list = [[encode_pokemon(dummy_poke) for _ in range(K * 12)] for _ in range(B)]
pb = collate_features(poke_feat_list)
check("PokemonBatch species_idx shape",   tuple(pb.species_idx.shape) == (B, K * 12))
check("PokemonBatch move_idx shape",      tuple(pb.move_idx.shape)    == (B, K * 12, 4))
check("PokemonBatch scalars shape",       tuple(pb.scalars.shape)     == (B, K * 12, 223))

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
    pre_tok, cur_tok, full_seq, padding_mask, cls_out = backbone.encode(tok, field_t)

check("pre_tokens shape [B, 13, D_MODEL]",     tuple(pre_tok.shape) == (B, 13, D_MODEL))
check("current_tokens shape [B, 13, D_MODEL]", tuple(cur_tok.shape) == (B, 13, D_MODEL),
      str(tuple(cur_tok.shape)))
check("full_seq shape [B, 52, D_MODEL]",        tuple(full_seq.shape) == (B, 52, D_MODEL))
check("padding_mask shape [B, 52]",             tuple(padding_mask.shape) == (B, 52))
check("cls_out shape [B, D_MODEL]",             tuple(cls_out.shape) == (B, D_MODEL))
check("no NaN in backbone output",             not cur_tok.isnan().any().item())
check("cls_token exists",                      hasattr(backbone, "cls_token"))

action_mask = torch.zeros(B, 13, dtype=torch.bool)
action_mask[:, 4:8] = True

act_emb = torch.randn(B, 13, D_MODEL)
with torch.no_grad():
    logits, attn_entropy, attn_rank = backbone.act(act_emb, cur_tok, action_mask)
check("action logits shape [B, 13]",  tuple(logits.shape) == (B, 13))
check("masked logits are -1e9",       (logits[:, 4:8] < -1e8).all().item())
check("attn_entropy is scalar",       attn_entropy.shape == torch.Size([]))
check("attn_rank is scalar",          attn_rank.shape    == torch.Size([]))


# ── 5. ActionEncoder ──────────────────────────────────────────────────────────

section("5. ActionEncoder")

from env.action_space import ActionEncoder

ae = ActionEncoder(move_embed=emb.move_embed, type_embed=emb.type_embed)
ae.eval()

move_idx       = pb.move_idx[:, K * 12 - 12, :]
pp_ratio       = torch.rand(B, 4)
move_disabled  = torch.zeros(B, 4)
mechanic_id    = torch.zeros(B, dtype=torch.long)
mech_type_idx  = torch.zeros(B, dtype=torch.long)

with torch.no_grad():
    action_embeds = ae(
        active_token      = pre_tok[:, 0, :],
        move_idx          = move_idx,
        pp_ratio          = pp_ratio,
        move_disabled     = move_disabled,
        bench_tokens      = pre_tok[:, 1:6, :],
        mechanic_id       = mechanic_id,
        mechanic_type_idx = mech_type_idx,
    )
check("action embeds shape [B, 13, D_MODEL]",  tuple(action_embeds.shape) == (B, 13, D_MODEL),
      str(tuple(action_embeds.shape)))


# ── 6. CynthAIAgent — critic_n_layers=0 (MLP on CLS) ────────────────────────

section("6. CynthAIAgent full forward — critic_n_layers=0")

from model.agent import CynthAIAgent

agent_mlp = CynthAIAgent(critic_n_layers=0, critic_detach=False)
agent_mlp.eval()

n_params = sum(p.numel() for p in agent_mlp.parameters())
check("parameter count > 1M",              n_params > 1_000_000, f"{n_params:,}")

with torch.no_grad():
    out = agent_mlp(
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
check("log_probs sum to ~1 on legal actions",
      (out.log_probs[:, ~action_mask[0]].exp().sum(dim=-1) - 1.0).abs().max().item() < 1e-4)


# ── 7. CynthAIAgent — critic_n_layers=2 (Transformer value head) ─────────────

section("7. CynthAIAgent full forward — critic_n_layers=2")

agent_transformer = CynthAIAgent(critic_n_layers=2, critic_detach=True)
agent_transformer.eval()

n_params_transformer = sum(p.numel() for p in agent_transformer.parameters())
n_critic = sum(p.numel() for p in agent_transformer.value_head.parameters())
check("value_head attr exists",          hasattr(agent_transformer, "value_head"))
check("more params than MLP mode",      n_params_transformer > n_params, f"{n_params_transformer:,}")
check("value_head has its own params",   n_critic > 0, f"{n_critic:,}")

with torch.no_grad():
    out_transformer = agent_transformer(
        poke_batch        = pb,
        field_tensor      = field_t,
        move_idx          = move_idx,
        pp_ratio          = pp_ratio,
        move_disabled     = move_disabled,
        mechanic_id       = mechanic_id,
        mechanic_type_idx = mech_type_idx,
        action_mask       = action_mask,
    )

check("value shape [B, 1]",           tuple(out_transformer.value.shape) == (B, 1))
check("logits shape [B, 13]",         tuple(out_transformer.action_logits.shape) == (B, 13))
check("no NaN in value",              not out_transformer.value.isnan().any().item())
check("no NaN in logits",             not out_transformer.action_logits.isnan().any().item())


# ── 8. ValueHead standalone ─────────────────────────────────────────────────

section("8. ValueHead standalone forward")

from model.critic import ValueHead

# n_layers=0: MLP on CLS
vh0 = ValueHead(n_layers=0)
vh0.eval()
with torch.no_grad():
    v0, wl0 = vh0(full_seq, padding_mask, cls_out)
check("n_layers=0 -> value shape [B, 1]",      tuple(v0.shape) == (B, 1))
check("n_layers=0 -> no NaN",                   not v0.isnan().any().item())
check("n_layers=0 -> win_logit is None",         wl0 is None)

# n_layers=1,2,3
for n_layers in [1, 2, 3]:
    vh = ValueHead(n_layers=n_layers)
    vh.eval()
    with torch.no_grad():
        v, wl = vh(full_seq, padding_mask, cls_out)
    check(f"n_layers={n_layers} -> value shape [B, 1]", tuple(v.shape) == (B, 1))
    check(f"n_layers={n_layers} -> no NaN",             not v.isnan().any().item())
    check(f"n_layers={n_layers} -> win_logit is None",   wl is None)

# Victory head
vh_victory = ValueHead(n_layers=2, use_victory_head=True)
vh_victory.eval()
with torch.no_grad():
    v_vh, wl_vh = vh_victory(full_seq, padding_mask, cls_out)
check("victory head -> value shape [B, 1]",     tuple(v_vh.shape) == (B, 1))
check("victory head -> win_logit shape [B, 1]", tuple(wl_vh.shape) == (B, 1))

# Value bound
vh_bound = ValueHead(n_layers=0, value_bound=5.0)
vh_bound.eval()
with torch.no_grad():
    v_bound, _ = vh_bound(full_seq, padding_mask, cls_out)
check("value_bound=5 -> |v| <= 5",  (v_bound.abs() <= 5.0 + 1e-6).all().item())


# ── 9. PredictionHeads ────────────────────────────────────────────────────────

section("9. PredictionHeads build_targets + compute_loss")

from model.prediction_heads import PredictionHeads
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
check("move loss is finite",         loss_dict["moves"].isfinite().item())


# ── 10. move_recall metric ────────────────────────────────────────────────────

section("10. move_recall metric (sum not any)")

# Simulate a Pokémon with 4 known moves, model predicts all 4 correctly
N_MOVES_TEST = 686
logits_perfect = torch.full((1, 1, N_MOVES_TEST), -10.0)
targets_4      = torch.zeros(1, 1, N_MOVES_TEST)
true_indices   = [10, 50, 100, 200]
for idx in true_indices:
    logits_perfect[0, 0, idx] = 10.0
    targets_4[0, 0, idx]      = 1.0
mask_1 = torch.ones(1, 1, dtype=torch.bool)

acc = PredictionHeads.compute_accuracy(
    logits          = type("L", (), {"item": torch.zeros(1,1,250), "ability": torch.zeros(1,1,311),
                                     "tera": torch.zeros(1,1,19), "moves": logits_perfect,
                                     "stats": torch.zeros(1,1,6)})(),
    item_targets    = torch.zeros(1,1,dtype=torch.long),
    ability_targets = torch.zeros(1,1,dtype=torch.long),
    tera_targets    = torch.zeros(1,1,dtype=torch.long),
    move_targets    = targets_4,
    item_mask       = torch.zeros(1,1,dtype=torch.bool),
    ability_mask    = torch.zeros(1,1,dtype=torch.bool),
    tera_mask       = torch.zeros(1,1,dtype=torch.bool),
    move_mask       = mask_1,
    stats_targets   = torch.zeros(1,1,6),
    stats_mask      = torch.zeros(1,1,dtype=torch.bool),
)
check("perfect prediction -> recall = 1.0",
      abs(acc["move_recall"] - 1.0) < 1e-4, f"got {acc['move_recall']:.4f}")

# Model predicts 2/4 correct -> recall = 0.5
logits_half = torch.full((1, 1, N_MOVES_TEST), -10.0)
logits_half[0, 0, 10]  = 10.0   # correct
logits_half[0, 0, 50]  = 10.0   # correct
logits_half[0, 0, 300] = 9.0    # wrong
logits_half[0, 0, 400] = 9.0    # wrong

acc2 = PredictionHeads.compute_accuracy(
    logits          = type("L", (), {"item": torch.zeros(1,1,250), "ability": torch.zeros(1,1,311),
                                     "tera": torch.zeros(1,1,19), "moves": logits_half,
                                     "stats": torch.zeros(1,1,6)})(),
    item_targets    = torch.zeros(1,1,dtype=torch.long),
    ability_targets = torch.zeros(1,1,dtype=torch.long),
    tera_targets    = torch.zeros(1,1,dtype=torch.long),
    move_targets    = targets_4,
    item_mask       = torch.zeros(1,1,dtype=torch.bool),
    ability_mask    = torch.zeros(1,1,dtype=torch.bool),
    tera_mask       = torch.zeros(1,1,dtype=torch.bool),
    move_mask       = mask_1,
    stats_targets   = torch.zeros(1,1,6),
    stats_mask      = torch.zeros(1,1,dtype=torch.bool),
)
check("2/4 correct -> recall = 0.5",
      abs(acc2["move_recall"] - 0.5) < 1e-4, f"got {acc2['move_recall']:.4f}")


# ── 11. Training losses ───────────────────────────────────────────────────────

section("11. compute_losses")

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


# ── 12. Checkpoint round-trip — critic_n_layers=0 ────────────────────────────

section("12. Checkpoint round-trip — critic_n_layers=0")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "test_ckpt_mlp.pt")
    torch.save({"update": 42, "model": agent_mlp.state_dict()}, path)

    ckpt   = torch.load(path, map_location="cpu", weights_only=True)
    agent_reload = CynthAIAgent(critic_n_layers=0, critic_detach=False)
    missing, unexpected = agent_reload.load_state_dict(ckpt["model"], strict=False)
    check("checkpoint update=42",         ckpt["update"] == 42)
    check("no missing keys",              len(missing) == 0, str(missing))
    check("no unexpected keys",           len(unexpected) == 0, str(unexpected))
    p1 = next(agent_mlp.parameters())
    p2 = next(agent_reload.parameters())
    check("weights match after reload",   torch.allclose(p1, p2))


# ── 13. Checkpoint round-trip — strict=False (arch change) ────────────────────

section("13. Checkpoint strict=False — load n_layers=0 into n_layers=2")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "test_ckpt_migration.pt")
    torch.save({"update": 100, "model": agent_mlp.state_dict()}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    agent_new = CynthAIAgent(critic_n_layers=2, critic_detach=True)
    missing, unexpected = agent_new.load_state_dict(ckpt["model"], strict=False)
    # value_head.cls_token, value_head.transformer.* should be missing (new params)
    has_vh_missing = any("value_head" in k for k in missing)
    check("value_head transformer params are missing (random init)", has_vh_missing,
          f"missing={missing[:3]}")
    check("no unexpected keys",  len(unexpected) == 0, str(unexpected))
    check("forward still works after partial load",
          agent_new(pb, field_t, move_idx, pp_ratio, move_disabled,
                    mechanic_id, mech_type_idx, action_mask).value.shape == (B, 1))


# ── 14. Gradient flow — detached vs non-detached critic ──────────────────────

section("14. Gradient flow — detached critic")

agent_grad = CynthAIAgent(critic_n_layers=2, critic_detach=True)
agent_grad.train()

out_grad = agent_grad(
    poke_batch=pb, field_tensor=field_t,
    move_idx=move_idx, pp_ratio=pp_ratio, move_disabled=move_disabled,
    mechanic_id=mechanic_id, mechanic_type_idx=mech_type_idx,
    action_mask=action_mask,
)

# 14a. Value loss backward should reach value_head params
out_grad.value.sum().backward(retain_graph=True)

vh_param = next(agent_grad.value_head.parameters())
check("value_head has grad after value.backward()",
      vh_param.grad is not None and vh_param.grad.abs().sum() > 0)

# 14b. poke_emb should NOT get gradient from critic (detached)
poke_emb_param = next(agent_grad.poke_emb.parameters())
check("poke_emb has NO grad from critic (detached)",
      poke_emb_param.grad is None or poke_emb_param.grad.abs().sum().item() == 0)

# 14c. backbone should NOT get gradient from critic (detached)
backbone_param = next(agent_grad.backbone.parameters())
check("backbone has NO grad from critic (detached)",
      backbone_param.grad is None or backbone_param.grad.abs().sum().item() == 0)

# 14d. Value head transformer params should get grad
if agent_grad.value_head.n_layers > 0:
    critic_transformer_param = next(agent_grad.value_head.transformer.parameters())
    check("value_head transformer has grad",
          critic_transformer_param.grad is not None and critic_transformer_param.grad.abs().sum() > 0)


# ── 15. Gradient flow — non-detached critic ──────────────────────────────────

section("15. Gradient flow — non-detached critic")

agent_nodetach = CynthAIAgent(critic_n_layers=0, critic_detach=False)
agent_nodetach.train()

out_nodetach = agent_nodetach(
    poke_batch=pb, field_tensor=field_t,
    move_idx=move_idx, pp_ratio=pp_ratio, move_disabled=move_disabled,
    mechanic_id=mechanic_id, mechanic_type_idx=mech_type_idx,
    action_mask=action_mask,
)

out_nodetach.value.sum().backward(retain_graph=True)

backbone_param_nd = next(agent_nodetach.backbone.parameters())
check("non-detached: backbone gets grad from value loss",
      backbone_param_nd.grad is not None and backbone_param_nd.grad.abs().sum().item() > 0)


# ── 16. Checkpoint round-trip — critic_n_layers=2 ──────────────────────────

section("16. Checkpoint round-trip — critic_n_layers=2")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "test_ckpt_transformer.pt")
    torch.save({"update": 200, "model": agent_transformer.state_dict()}, path)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    agent_reload2 = CynthAIAgent(critic_n_layers=2, critic_detach=True)
    missing, unexpected = agent_reload2.load_state_dict(ckpt["model"], strict=True)
    check("strict load: no missing keys",     len(missing) == 0, str(missing))
    check("strict load: no unexpected keys",   len(unexpected) == 0, str(unexpected))

    # Verify outputs match
    agent_reload2.eval()
    with torch.no_grad():
        out_reload = agent_reload2(
            poke_batch=pb, field_tensor=field_t,
            move_idx=move_idx, pp_ratio=pp_ratio, move_disabled=move_disabled,
            mechanic_id=mechanic_id, mechanic_type_idx=mech_type_idx,
            action_mask=action_mask,
        )
    check("reloaded value matches original",
          torch.allclose(agent_transformer(pb, field_t, move_idx, pp_ratio, move_disabled,
                         mechanic_id, mech_type_idx, action_mask).value,
                         out_reload.value, atol=1e-5))


# ── 17. MoveEmbedding — additive structure + feature integrity ──────────────

section("17. MoveEmbedding — additive prior features")

from model.embeddings import MoveEmbedding, D_MOVE
from env.state_encoder import MOVE_FEATURES, N_MOVE_FEATURES, MOVE_INDEX

me = MoveEmbedding()

# Shape checks
idx3 = torch.tensor([0, 1, 2])
out3 = me(idx3)
check("MoveEmbedding([0,1,2]) shape [3, D_MOVE]",  tuple(out3.shape) == (3, D_MOVE))

batch_idx = torch.randint(0, N_MOVES, (4, 4))
out_batch = me(batch_idx)
check("MoveEmbedding([4,4]) shape [4, 4, D_MOVE]", tuple(out_batch.shape) == (4, 4, D_MOVE))

# Buffer is not a parameter (not trained directly)
check("move_features is buffer, not parameter",
      "move_features" not in dict(me.named_parameters()))
check("move_features is registered buffer",
      "move_features" in dict(me.named_buffers()))
check("move_features shape [N_MOVES, N_MOVE_FEATURES]",
      tuple(me.move_features.shape) == (N_MOVES, N_MOVE_FEATURES))

# Additive structure: output = id_embed + prior_proj
with torch.no_grad():
    test_idx = torch.tensor([5, 10, 100])
    id_part    = me.move_id_embed(test_idx)
    prior_part = me.move_prior_proj(me.move_features[test_idx])
    full_out   = me(test_idx)
    check("additive structure: id + prior = output",
          torch.allclose(id_part + prior_part, full_out, atol=1e-5))

# Gradient flow to both components
me.zero_grad()
loss = me(torch.tensor([1, 2, 3])).sum()
loss.backward()
check("grad flows to move_id_embed",   me.move_id_embed.weight.grad is not None
      and me.move_id_embed.weight.grad.any().item())
check("grad flows to move_prior_proj", me.move_prior_proj.weight.grad is not None
      and me.move_prior_proj.weight.grad.any().item())

# Spot-check feature values for known moves
def _check_move_feat(move_name, feat_name, expected, tol=1e-3):
    idx = MOVE_INDEX.get(move_name)
    if idx is None:
        check(f"{move_name}.{feat_name} (move not in index)", False, "missing from MOVE_INDEX")
        return
    feat_names = [
        *[f"type_{t}" for t in ["__unk__","normal","fire","water","electric","grass","ice",
          "fighting","poison","ground","flying","psychic","bug","rock","ghost","dragon",
          "dark","steel","fairy"]],
        "cat_physical","cat_special","cat_status",
        "basePower","accuracy","pp","priority",
        "contact","sound","bullet","slicing","punch","bite","heal","recharge","bypasssub","defrost",
        "selfSwitch","drain_ratio","recoil_ratio","hasCrashDamage",
        "secondary_chance","sec_brn","sec_par","sec_psn","sec_tox","sec_slp","sec_frz",
    ]
    fi = feat_names.index(feat_name)
    actual = MOVE_FEATURES[idx, fi].item()
    check(f"{move_name}.{feat_name} = {expected}",
          abs(actual - expected) < tol, f"got {actual}")

_check_move_feat("thunderbolt", "basePower",  90.0)
_check_move_feat("thunderbolt", "type_electric", 1.0)
_check_move_feat("thunderbolt", "cat_special", 1.0)
_check_move_feat("flareblitz",  "recoil_ratio", 0.33)
_check_move_feat("flareblitz",  "sec_brn",      1.0)
_check_move_feat("flareblitz",  "defrost",       1.0)
_check_move_feat("uturn",       "selfSwitch",    1.0)
_check_move_feat("uturn",       "contact",       1.0)
_check_move_feat("absorb",      "drain_ratio",   0.5)
_check_move_feat("spore",       "cat_status",    1.0)
_check_move_feat("boomburst",   "sound",         1.0)
_check_move_feat("drainpunch",  "punch",         1.0)


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n-- Done --- {n_passed} passed, {n_failed} failed {'=' * 30}")
if n_failed:
    sys.exit(1)
