# CynthAI v2

PPO self-play agent for Pokémon Showdown. Transformer backbone with spatio-temporal position embeddings, cross-attention actor head, and auxiliary opponent-prediction heads.

---

## Architecture overview

```
Rust simulator  ──►  state_encoder  ──►  embeddings  ──►  BattleBackbone
                           │                                     │
                    (12 Pokémon tokens                  Single Transformer pass
                     + 1 field token                   over K=4 sliding window
                     per turn)                                   │
                                                        ┌────────┴────────┐
                                                        ▼                 ▼
                                                   Value head       current_tokens
                                                   V(s) [B,1]       [B, 13, 256]
                                                                          │
                                          ┌───────────────────────────────┤
                                          ▼                               ▼
                                   ActionEncoder                   PredictionHeads
                                   [B, 13, 256]                 (opponent tokens only)
                                          │
                                          ▼
                                   Cross-attention
                                   action_logits [B, 13]
                                   masked + log_softmax
```

**One Transformer pass per step.** `backbone.encode()` runs the Transformer and returns `current_tokens` + V(s). The actor (`backbone.act()`) is a single cross-attention on top — no second forward pass.

---

## Key numbers

| Hyperparameter       | Value                          |
|----------------------|--------------------------------|
| Total parameters     | ~3.9 M                         |
| d_model              | 256                            |
| Transformer layers   | 3                              |
| Attention heads      | 4                              |
| FFN dim              | 512                            |
| K turns (window)     | 4                              |
| Tokens per turn      | 13 (12 Pokémon + 1 field)      |
| Full sequence length | 52                             |
| TOKEN_DIM (Pokémon)  | 438                            |
| FIELD_DIM            | 72                             |
| N_SCALARS            | 222                            |
| Action slots         | 13 (4 moves + 4 mechanic + 5 switches) |

---

## Project structure

```
CynthAI_v2/
├── env/
│   ├── state_encoder.py      # Rust state dict → PokemonFeatures / FieldFeatures
│   └── action_space.py       # ActionEncoder: move/mechanic/switch → [B, 13, D_MODEL]
│
├── model/
│   ├── embeddings.py         # PokemonEmbeddings: indices + scalars → TOKEN_DIM
│   ├── backbone.py           # BattleBackbone: Transformer + value head + actor cross-attn
│   ├── prediction_heads.py   # PredictionHeads: auxiliary opponent hidden-state prediction
│   └── agent.py              # CynthAIAgent: full forward pass wrapper (AgentOutput)
│
├── training/
│   ├── rollout.py            # collect_rollout: parallel episode collection + GAE
│   ├── losses.py             # compute_losses: PPO-clip + value + entropy + pred
│   └── self_play.py          # train(): main PPO self-play loop + opponent pool
│
├── data/
│   └── dicts/                # species_index.json, move_index.json, item_index.json, …
│
├── simulator/                # Rust PyBattle bindings
│
├── README.md                 # this file
├── ARCHITECTURE.md           # detailed forward-pass diagram
├── POKEMON_VECTOR.md         # Pokémon token design (sections 1–11 + scalars)
├── FIELD.md                  # field token design (FIELD_DIM=72 breakdown)
└── BELIEF_STATE.md           # v2 opponent belief injection — design + activation plan
```

---

## Modules

### `env/state_encoder.py`
Converts `PyBattle.get_state()` dicts to structured feature objects.
- `encode_pokemon(poke_dict)` → `PokemonFeatures` (indices + 222 scalars)
- `encode_field(state_dict)` → `FieldFeatures` (72 floats)
- Vocabulary tables: `SPECIES_INDEX`, `MOVE_INDEX`, `ITEM_INDEX`, `ABILITY_INDEX`, `TYPE_INDEX`
- Constants: `N_SPECIES=1381`, `N_MOVES=686`, `N_ITEMS=250`, `N_ABILITIES=311`, `N_TYPES=19`

### `env/action_space.py`
Universal mechanic support — same 13-slot layout for Tera, Mega, Z-move, Dynamax, None.
- `ActionEncoder(move_embed, type_embed)` — shares embedding tables with `PokemonEmbeddings`
- Slots: `[0:4]` base moves, `[4:8]` mechanic moves (additive modulation), `[8:13]` switches
- `MECH_NONE=0 / MECH_TERA=1 / MECH_MEGA=2 / MECH_ZMOVE=3 / MECH_DYNAMAX=4`

### `model/embeddings.py`
- `PokemonEmbeddings` — projects a `PokemonBatch [B, K*12]` to `[B, K*12, TOKEN_DIM=438]`
- `collate_features(list[list[PokemonFeatures]])` → batched `PokemonBatch`
- Embedding tables: species (32-d, init from base stats), types (8-d, SVD type-chart prior), items (16-d), abilities (16-d), moves (32-d, SVD from 46 known attributes)
- `collate_field_features(list[FieldFeatures])` → `FieldBatch [B, 72]`

### `model/backbone.py`
- `BattleBackbone` — core Transformer (Pre-LN, `enable_nested_tensor=False`)
- `encode(pokemon_tokens, field_tokens)` → `(current_tokens [B,13,256], value [B,1])`
- `act(action_embeds, current_tokens, action_mask)` → `action_logits [B,13]`
- `get_attention_maps(pokemon_tokens, field_tokens)` → dict with `attention_maps` (list of `[B,4,52,52]` per layer), `value`, `current_tokens`, `token_labels` — for interpretability (monkey-patches self-attention to capture weights)
- Positional embeddings: `temporal_emb(K=4)` + `slot_emb(13)`, both additive and learned

### `model/prediction_heads.py`
Auxiliary supervised heads on opponent tokens `current_tokens[:, 6:12, :]`.
- Predicts: item (250 cls), ability (311 cls), tera type (19 cls), 4 moves (686 cls each)
- `compute_loss(logits, *targets)` — masked cross-entropy (only revealed slots count)
- `build_targets(full_opp_batch)` — extracts targets + masks from ground-truth state
- No belief injection in v1. See `BELIEF_STATE.md` for v2 design.

### `model/agent.py`
`CynthAIAgent` — single `forward()` call returns `AgentOutput`:
```python
AgentOutput(current_tokens, value, action_logits, log_probs, pred_logits)
```
Shares `move_embed` and `type_embed` between `PokemonEmbeddings` and `ActionEncoder`.

### `training/rollout.py`
- `collect_rollout(agent_self, agent_opp, n_envs=16, min_steps=512)` — complete-episode collection
- `BattleWindow` — K=4 sliding window, zero-padded for early turns
- `RolloutBuffer` — stores `Transition` objects, computes GAE (γ=0.99, λ=0.95), yields minibatches
- `build_action_mask(state, side_idx)` — derives `[13] bool` mask from `get_state()` (True = illegal)
- `compute_step_reward(...)` — dense: ±0.05 per KO, 0.01×ΔHP advantage, ±1.0 terminal

### `training/losses.py`
`compute_losses(...)` → dict with keys `policy / value / entropy / pred / total`:
- PPO-clip with ε=0.2, advantage normalisation per batch
- MSE value loss (returns bounded in ≈ [−1, 1])
- Entropy over legal actions only (`masked_fill(action_mask, 0)` before sum)
- Coefficients: `c_value=0.5`, `c_entropy=0.01`, `c_pred=0.5`

### `training/self_play.py`
`train(cfg: TrainingConfig)` — full PPO self-play loop:
1. Sample opponent from `OpponentPool` (falls back to live agent when pool is empty)
2. `collect_rollout` under `torch.no_grad()`
3. 4 PPO epochs × minibatch updates (batch=128, `clip_grad_norm=0.5`)
4. Linear LR decay (`3e-4 → 1e-5` over `total_updates`)
5. Log every update: `update | win% | policy | value | entropy | pred | total | lr`
6. CSV logging: `checkpoints/metrics.csv` — persistent history of all metrics
7. Checkpoint `.pt` every 50 updates; keeps only last 5 regular checkpoints
8. Best checkpoint (`agent_best.pt`) saved when win rate improves
9. Snapshot to opponent pool every 10 updates

---

## Training

```bash
cd CynthAI_v2
python -m training.self_play
```

Or with custom config:
```python
from training.self_play import TrainingConfig, train

train(TrainingConfig(
    device        = "cuda",
    n_envs        = 32,
    min_steps     = 1024,
    total_updates = 5000,
    lr            = 3e-4,
    lr_min        = 1e-5,
))
```

Loading a checkpoint:
```python
import torch
from model.agent import CynthAIAgent

ckpt  = torch.load("checkpoints/agent_001000.pt")
agent = CynthAIAgent()
agent.load_state_dict(ckpt["model"])
```

---

## Reward design

| Event              | Reward           |
|--------------------|------------------|
| Win                | +1.0             |
| Loss               | −1.0             |
| Opponent KO        | +0.05            |
| Own KO             | −0.05            |
| HP advantage delta | +0.01 × Δadv     |

HP advantage = `(own_hp / own_maxhp) − (opp_hp / opp_maxhp)`. Dense at every step, sparse terminal reward. Zero-sum by construction in self-play.

---

## Hyperparameters (TrainingConfig defaults)

| Parameter          | Default | Notes                             |
|--------------------|---------|-----------------------------------|
| `lr`               | 3e-4    | Adam, eps=1e-5                    |
| `lr_min`           | 1e-5    | Linear decay endpoint             |
| `total_updates`    | 2000    |                                   |
| `n_envs`           | 16      | Parallel battles                  |
| `min_steps`        | 512     | Min transitions per rollout       |
| `n_epochs`         | 4       | PPO epochs per rollout            |
| `batch_size`       | 128     |                                   |
| `gamma`            | 0.99    | GAE discount                      |
| `lam`              | 0.95    | GAE lambda                        |
| `clip_eps`         | 0.2     | PPO clipping                      |
| `c_value`          | 0.5     | Value loss coefficient            |
| `c_entropy`        | 0.01    | Entropy bonus coefficient         |
| `c_pred`           | 0.5     | Predictor auxiliary coefficient   |
| `max_grad_norm`    | 0.5     | Gradient clipping                 |
| `pool_size`        | 5       | Max opponent snapshots            |
| `pool_update_freq` | 10      | Snapshot every N updates          |
| `checkpoint_freq`  | 50      | Save .pt every N updates          |
| `keep_last`        | 5       | Max regular checkpoints to keep   |
| `log_every`        | 1       | Stdout + CSV log frequency        |

---

## Action mask convention

`True = ILLEGAL` throughout. Derived from `get_state()` since the Rust sim has no `get_legal_actions()`. Force-switch, trapped, tera-already-used, and PP-disabled moves are all handled in `build_action_mask()`.

---

## v2 roadmap

| Feature                           | Status     |
|-----------------------------------|------------|
| Rust simulator integration        | Done       |
| State encoder (sections 1–11)     | Done       |
| Pokémon / field embeddings        | Done       |
| SVD embedding priors (type/move)  | Done       |
| Transformer backbone              | Done       |
| Action encoder (all mechanics)    | Done       |
| Prediction heads (auxiliary)      | Done       |
| Attention map extraction          | Done       |
| PPO rollout + GAE                 | Done       |
| PPO losses                        | Done       |
| Self-play training loop           | Done       |
| POMDP masking (opponent info)     | v2         |
| Belief state injection            | v2 — see `BELIEF_STATE.md` |
| Pretraining on team sets (162k)   | v2         |
| Elo-based opponent selection      | v2         |

---

## Documentation

| File                | Contents                                                    |
|---------------------|-------------------------------------------------------------|
| `ARCHITECTURE.md`   | Full forward-pass diagram with shapes                       |
| `POKEMON_VECTOR.md` | Pokémon token: all 11 sections, embedding dims, scalars     |
| `FIELD.md`          | Field token: all 72 dims explained                          |
| `BELIEF_STATE.md`   | v2 opponent belief injection: design, activation conditions |
