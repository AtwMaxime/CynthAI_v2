# CynthAI v2

**PPO self-play agent for Pokémon Showdown (Gen 9 Random Battle).**

CynthAI learns to play Pokémon battles from scratch via self-play reinforcement learning. It uses a Transformer backbone over a sliding window of past turns, a cross-attention actor head for action selection, and auxiliary prediction heads for opponent state inference.

The entire battle simulation runs in Rust (`pokemon-showdown-rs`) via PyO3 bindings for maximum throughput.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Why This Architecture?](#why-this-architecture)
- [Key Numbers](#key-numbers)
- [Project Structure](#project-structure)
- [Pokémon Token Design (TOKEN_DIM = 438)](#pokémon-token-design-token_dim--438)
- [Field Token Design (FIELD_DIM = 72)](#field-token-design-field_dim--72)
- [Action Space (13 slots)](#action-space-13-slots)
- [Reward Design](#reward-design)
- [Training Loop (PPO Self-Play)](#training-loop-ppo-self-play)
- [Module Details](#module-details)
- [Hyperparameters](#hyperparameters)
- [Installation](#installation)
- [Usage](#usage)
- [Training Results](#training-results)
- [v2 Roadmap](#v2-roadmap)
- [Known Issues](#known-issues)
- [Documentation](#documentation)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Rust Simulator (pokemon-showdown-rs)                     │
│  PyBattle(format, seed) ──► get_state() ──► {turn, field, sides, pokemon…}  │
│  PyBattle.get_new_log_entries() ──► [PS protocol log lines]                  │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         state_encoder.py + revealed_tracker.py                │
│  encode_pokemon(poke_dict) ──► PokemonFeatures (indices + 222 scalars)       │
│  encode_field(state_dict)   ──► FieldFeatures (72 floats)                   │
│  RevealedTracker.update(log_entries, state, side) ──► {species/item/...}     │
│  12 Pokémon + 1 field token per turn                                        │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      embeddings.py (PokemonEmbeddings)                        │
│  [B, K*12] indices + scalars ──► [B, K*12, TOKEN_DIM=438]                   │
│  apply_reveal_mask(batch, reveal, ratio) ──► masked PokemonBatch             │
│  Embeddings: species(32) + types(3×8) + item(16) + ability(16) + moves(4×32)│
│  + scalars(222) = 438                                                        │
│  SVD priors: type chart (19×8), move attributes (686×32)                    │
└───────────────────────────────────┬──────────────────────────────────────────┘
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      backbone.py (BattleBackbone)                             │
│                                                                              │
│  ┌─── _build_sequence ────┐                                                 │
│  │  Pokémon [B,48,438]    │  project → [B,48,256]                           │
│  │  Field   [B,4,72 ]     │  project → [B,4,256]                            │
│  │  Reshape → [B,4,13,256]│  + temporal_emb + slot_emb                     │
│  │  Flatten → [B,52,256]  │  (4 turns × 13 tokens = 52)                    │
│  └───────────┬────────────┘                                                  │
│              ▼                                                               │
│  ┌─── TransformerEncoder ──┐  Pre-LN, 3 layers, 4 heads, d_model=256       │
│  │  [B, 52, 256] → [B, 52, 256]   bidirectional, no causal mask            │
│  └───────────┬────────────┘                                                  │
│              ▼                                                               │
│  ┌─────────────────────────┐  Select last 13 tokens (current turn)         │
│  │  current_tokens [B,13,256]  ────► Value Head ──► V(s) [B,1]             │
│  └───────────┬─────────────┘                                                 │
│              │                                                               │
│  ┌───────────┴───────────────┐                                              │
│  ▼                           ▼                                               │
│  ActionEncoder          PredictionHeads                                      │
│  [B,13,256]              [B,6,250/311/19/686]                               │
│      │                   (item/ability/tera/moves)                          │
│      ▼                                                                       │
│  Cross-attention (Q=actions, K=V=current_tokens)                            │
│  action_logits [B,13] — masked + log_softmax                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Key design choice: one Transformer pass per step.** The backbone runs the Transformer once and returns enriched tokens for the current turn. The actor is a lightweight cross-attention on top — no second forward pass. The critic (value head) is an MLP on the flattened 13 current tokens.

### Why This Architecture?

| Choice | Rationale |
|--------|-----------|
| **Single Transformer pass** | The Pokémon Showdown action space is small (13 discrete actions). A full decoder-based autoregressive policy is overkill. Instead, we run the Transformer once for both value and state representation, then apply a lightweight cross-attention actor. |
| **Sliding window (K=4)** | Battles have ~20–100 turns. Encoding all past turns is wasteful and exceeds context. K=4 captures recent game state while keeping sequence length fixed at 52 tokens. |
| **Bidirectional attention** | Unlike language modelling, battle state has no causal direction — all past K turns are fully observable. No mask needed. |
| **Pre-LN Transformer** | More stable training than Post-LN, especially at small d_model. |
| **Auxiliary prediction heads** | Forces the model to encode hidden opponent information via masked cross-entropy — only gameplay-revealed attributes contribute to the loss. |
| **SVD embedding priors** | Species embeddings initialised from base stats; type embeddings from the type-effectiveness matrix SVD; move embeddings from 46 known attributes. Gives the model a warm-start rather than learning type matchups from scratch. |

---

## Key Numbers

| Hyperparameter        | Value                          |
|-----------------------|--------------------------------|
| Total parameters      | ~3.1 M                         |
| d_model               | 256                            |
| Transformer layers    | 3                              |
| Attention heads       | 4                              |
| FFN dim               | 512                            |
| Dropout               | 0.1                            |
| K turns (window)      | 4                              |
| Tokens per turn       | 13 (12 Pokémon + 1 field)      |
| Full sequence length  | 52                             |
| TOKEN_DIM (Pokémon)   | 438                            |
| FIELD_DIM             | 72                             |
| N_SCALARS             | 222                            |
| Action slots          | 13 (4 moves + 4 mechanic + 5 switches) |
| Vocabulary: species   | 1,381                          |
| Vocabulary: moves     | 686                            |
| Vocabulary: items     | 250                            |
| Vocabulary: abilities | 311                            |
| Vocabulary: types     | 19                             |
| Volatile flags        | 181                            |
| Side conditions       | 23                             |
| Weather types         | 9                              |
| Terrain types         | 5                              |

---

## Project Structure

```
CynthAI_v2/
│
├── env/                          # Environment encoding
│   ├── state_encoder.py          # Rust dict → PokemonFeatures / FieldFeatures
│   └── action_space.py           # ActionEncoder: 13 action embeddings
    │   └── revealed_tracker.py       # Tracks revealed opponent info from battle log
│
├── model/                        # Neural network modules
│   ├── embeddings.py             # PokemonEmbeddings + apply_reveal_mask()
│   ├── backbone.py               # BattleBackbone: Transformer + value + actor
│   ├── prediction_heads.py       # Auxiliary opponent predictors
│   └── agent.py                  # CynthAIAgent: full forward wrapper
│
├── training/                     # RL training pipeline
│   ├── rollout.py                # Parallel episode collection + GAE + RevealedTracker + POMDP mask
│   ├── losses.py                 # PPO-clip + value + entropy + pred losses
│   ├── self_play.py              # Main PPO self-play loop (entry point)
│   ├── evaluate.py               # Checkpoint evaluation vs random/checkpoint
│   └── visualize.py              # Training metrics plotting
│
├── data/
│   ├── build_dicts.py            # Vocabulary + SVD embedding generation
│   └── dicts/                    # Pre-built JSON index/embedding files
│       ├── species_index.json    # 1,381 species + base stats
│       ├── move_index.json       # 686 moves
│       ├── move_embeddings.json  # SVD prior: [686 × 32]
│       ├── item_index.json       # 250 items
│       ├── ability_index.json    # 311 abilities
│       ├── type_index.json       # 19 types
│       └── type_embeddings.json  # SVD prior: [19 × 8]
│
├── simulator/                    # Rust battle simulator (PyO3)
│   ├── Cargo.toml                # Rust package manifest
│   ├── src/lib.rs                # PyBattle: PyO3 bindings, get_state(), make_choices(), get_new_log_entries()
│   └── python/__init__.py        # Python import wrapper
│
├── checkpoints/                  # Saved model weights + metrics
│   ├── agent_best.pt
│   ├── agent_NNNNNN.pt
│   └── metrics.csv
│
├── tests/                        # Test files
│   ├── test_encoder.py           # Vocab sizes + state encoding
│   ├── test_full_pipeline.py     # End-to-end forward pass smoke test
│   └── test_attention_maps.py    # Attention map extraction + sanity checks
│
├── plots/                        # Generated training visualizations
│
├── .claude/                      # Claude Code configuration
├── requirements.txt
└── .gitignore
```

---

## Pokémon Token Design (TOKEN_DIM = 438)

Each Pokémon is encoded as a 438-dimensional vector with 11 active sections (Section 12 reserved for v2 belief injection).

| Section | Content | Dim | Details |
|---------|---------|-----|---------|
| 1 | Species index | 32 | Embedding initialised from 6 base stats (HP + 5 others) |
| 2 | Type 1 index | 8 | SVD type-chart prior |
| 3 | Type 2 index | 8 | SVD type-chart prior |
| 4 | Tera type index | 8 | SVD type-chart prior |
| 5 | Item index | 16 | Learned embedding |
| 6 | Ability index | 16 | Learned embedding |
| 7 | Moves (×4) | 128 | 4 × 32-d, SVD prior from 46 move attributes |
| 8 | Scalars | 222 | See below |
| 9 | *(v2 belief injection)* | — | Reserved for confidence scores |
| **Total** | | **438** | |

### Scalar Layout (222 dimensions)

| Index | Field | Values |
|-------|-------|--------|
| 0 | level | 0–100 (normalised) |
| 1 | hp_ratio | 0–1 |
| 2 | terastallized | 0/1 |
| 3–7 | base_stats | atk, def, spa, spd, spe |
| 8–12 | stats | atk, def, spa, spd, spe |
| 13 | is_predicted | 0/1 (v2) |
| 14–20 | boosts | atk, def, spa, spd, spe, accuracy, evasion |
| 21–24 | move_pp | 0–1 per slot |
| 25–28 | move_disabled | 0/1 per slot |
| 29–35 | status | 7-way one-hot: "", brn, frz, par, psn, tox, slp |
| 36–216 | volatiles | 181 flags (see below) |
| 217 | is_active | 0/1 |
| 218 | fainted | 0/1 |
| 219 | trapped | 0/1 |
| 220 | force_switch_flag | 0/1 |
| 221 | revealed | 0/1 |

**Note:** HP is NOT in base_stats/stats — the current HP ratio is at index 1 (hp_ratio). Base HP is baked into the species embedding initialisation.

### Volatile Conditions (181 flags)

Full Gen 9 catalogue including: confusion, leechseed, substitute, protect, endure, taunt, encore, disable, torment, imprison, ingrain, aquaring, charge, focusenergy, laserfocus, lockdown, magnetrise, telekinesis, nightmare, perishsong, curse, stockpile, powertrick, gastroacid, healblock, embargo, trapped, destinybond, grudge, flashfire, slowstart, powertrap, partiallytrapped, choice lock, gem, typechange, dynamax, terashell, command, commander, protosynthesis, quarkdrive, hadronengine, orichalcumpulse, and many more.

### Token Layout per Turn

Within each turn, the 13 tokens are arranged as:

| Slots | Pokémon | Count |
|-------|---------|-------|
| 0 | Your active | 1 |
| 1–5 | Your bench | 5 |
| 6 | Opponent active | 1 |
| 7–11 | Opponent bench | 5 |
| 12 | Field token | 1 |

### Sequence Structure (K=4 window)

Turns are stacked oldest-first: `[T0_OWN0…T0_FIELD, T1_OWN0…T1_FIELD, T2_OWN0…T2_FIELD, T3_OWN0…T3_FIELD]` for a total of 52 tokens per forward pass.

---

## Field Token Design (FIELD_DIM = 72)

| Offset | Content | Dim | Details |
|--------|---------|-----|---------|
| 0–8 | Weather | 9 | 9-way one-hot: none/sun/rain/sand/hail/snow/primordialsea/desolateland/deltastream |
| 9–13 | Terrain | 5 | 5-way one-hot: none/electric/grassy/misty/psychic |
| 14–21 | Pseudo-weather | 8 | gravity, magicroom, trickroom, wonderroom, echoedvoice, mudsport, watersport, fairylock |
| 22–46 | Side 0 conditions | 25 | 23 side conditions + pokemon_left/6 + total_fainted/6 |
| 47–71 | Side 1 conditions | 25 | Same as side 0 |

### Side Conditions (23 per side)

stealthrock, spikes (layers/3), toxicspikes (layers/2), stickyweb, reflect, lightscreen, auroraveil, tailwind, mist, safeguard, luckychant, craftyshield, matblock, quickguard, wideguard, gmaxsteelsurge, gmaxcannonade, gmaxvolcalith, gmaxvinelash, gmaxwildfire, grasspledge, waterpledge, firepledge

All normalised by their maximum layer count.

---

## Action Space (13 slots)

| Slot | Action | Description |
|------|--------|-------------|
| 0–3 | Regular moves | `move 1` to `move 4` |
| 4–7 | Mechanic moves | Moves with Tera/Mega/Z/Dmax modifier |
| 8–12 | Switches | Switch to bench Pokémon 0–4 |

### Action Embedding

The `ActionEncoder` builds each action embedding by combining:
- **Base moves (0–3):** `concat(move_embed[move_idx], active_token, pp_ratio, move_disabled)` → Linear → 256-d
- **Mechanic moves (4–7):** Base move embedding + additive modulation from `concat(type_embed[mech_type], one_hot(mech_id, 5))` → Linear → 256-d
- **Switches (8–12):** Direct Linear projection of bench tokens → 256-d

### Supported Mechanics

| Constant | Value |
|----------|-------|
| `MECH_NONE` | 0 |
| `MECH_TERA` | 1 |
| `MECH_MEGA` | 2 |
| `MECH_ZMOVE` | 3 |
| `MECH_DYNAMAX` | 4 |

### Action Masking (True = ILLEGAL)

The `build_action_mask()` function derives legal actions from the battle state:
- **Move slots:** Legal when the active Pokémon is not fainted, not force-switched, and the move is not disabled/out-of-PP
- **Mechanic slots:** Legal when Tera/Mega/Z/Dmax is available and base move is legal
- **Switch slots:** Legal when the active Pokémon is fainted, force-switched, or not trapped, AND bench Pokémon are available

**Action mask convention:** `True = ILLEGAL` throughout the codebase. Masked logits receive `-1e9` before softmax.

---

## Reward Design

| Event | Reward | Frequency |
|-------|--------|-----------|
| Win | +1.0 | Terminal (sparse) |
| Loss | -1.0 | Terminal (sparse) |
| Opponent KO | +0.05 | Per event |
| Own KO | -0.05 | Per event |
| HP advantage delta | 0.01 × Δadv | Every step (dense) |

**HP advantage** = `(own_total_hp / own_total_maxhp) - (opp_total_hp / opp_total_maxhp)`.

The HP-advantage delta (`adv_current - adv_previous`) is bounded approximately in [-1, 1] per turn, so the dense reward is at most ±0.01 per step — dominated by KO rewards (±0.05) and terminal reward (±1.0).

The reward is **zero-sum by design** in self-play: both sides experience symmetric rewards, so the expected sum of rewards across both players is zero.

---

## Training Loop (PPO Self-Play)

### Algorithm

1. **Sample opponent** from `OpponentPool` (deque of frozen past checkpoints). With 20% probability, override with a random policy (10% `RandomPolicy`, 10% `FullOffensePolicy`) for regularisation (P3). When the pool is empty (early training), the agent plays against itself (true self-play).
2. **Collect rollout**: Run N parallel battles (`n_envs=16` by default) under `torch.no_grad()`. After each `make_choices()`, the battle log is fetched via `get_new_log_entries()` and passed to `RevealedTracker.update()` to track which opponent attributes have been revealed through gameplay. The `apply_reveal_mask()` function then masks unrevealed opponent attributes in the model input — the agent makes decisions with partial information. The unmasked ground-truth data is stored separately for prediction targets. Store transitions (state, action, log_prob, value, reward, done, reveal_state) in a `RolloutBuffer`.
3. **Compute GAE**: γ=0.99, λ=0.95 over complete episodes. Episode boundaries detected via `Transition.done`.
4. **PPO update** (4 epochs, minibatch size 128):
   - Re-run forward pass for each transition — `apply_reveal_mask()` masks inputs again with the same reveal state and schedule ratio
   - Policy loss: clipped surrogate (ε=0.2)
   - Value loss: MSE between predicted V(s) and normalised GAE returns (P5: return normalisation, c_value=1.0)
   - Entropy bonus: negative entropy over legal actions only (coefficient 0.01)
   - Auxiliary prediction loss: masked cross-entropy on opponent item/ability/tera/moves — only revealed attributes contribute to the gradient (coefficient 0.5)
5. **Optimisation**: AdamW with `weight_decay=1e-4`; `clip_grad_norm_(max_norm=0.5)` before optim step. LR via warmup (20 steps) + cosine decay from 2.5e-4 (P4, P6).
6. **Log metrics**: win rate, policy/value/entropy/pred/total loss, learning rate, pool size, grad_norm, explained_variance, clip_frac (P4).
7. **Periodic evaluation**: Every `eval_freq` updates, run 10 games vs Random/FullOffense/pool opponents with the same masking ratio as training.
8. **Checkpoint**: Save model every 100 updates; keep best based on win rate.
9. **Snapshot**: Copy frozen agent to opponent pool when `win_rate > 0.55` over last 100 games, with 5-update cooldown (P3).

### Key Design Details

- **Complete episodes**: No truncated rollouts. The buffer collects entire battles, making GAE bootstrap straightforward (next_value=0 at terminal states).
- **Single Transform per step**: The encoder runs once; the actor is a separate cross-attention. During PPO training, only the encoder → value/policy path needs gradients.
- **Advantage normalisation**: Per-batch z-score normalisation before PPO clipping.
- **Gradient clipping**: Global norm clipped to 0.5 (P4).
- **Weight decay**: AdamW with `weight_decay=1e-4` for L2 regularisation (P4).
- **Learning rate schedule (P6)**: Warmup + cosine decay from 2.5e-4 to 1e-5:
  1. **Warmup** (first 20 updates): `lr = base_lr * update / warmup_steps` — ramps up linearly to avoid early instability.
  2. **Cosine decay** (remaining updates): `lr = base_lr * 0.5 * (1 + cos(π * (update - warmup) / (total - warmup)))` — smooth annealing.
- **Return normalisation (P5)**: `returns = (returns - mean) / (std + 1e-8)` before value loss computation for stable critic training.
- **POMDP masking curriculum (P1)**: Controls what fraction of opponent Pokémon have their unrevealed attributes masked. Four schedule modes:
  - `linear` (default): ratio ramps from 0 → `max_ratio` linearly after `warmup` updates
  - `exp`: exponential ramp, faster initial rise (`ratio = max_ratio * (1 - exp(-k * progress))`)
  - `step`: jumps from 0 to `max_ratio` at `step_update`
  - `phase`: manual breakpoints — e.g., `(600, 2500)` with values `(0.0, 0.5, 1.0)` gives 3 distinct phases
  Default: P1 starts after 200 warmup updates, ramps linearly to 1.0. Masking is applied consistently during rollout, training, and evaluation.
- **Reward curriculum (P2)**: Gradually transitions the agent from dense rewards (HP advantage, KOs) to near-sparse (only win/loss ±1). Same 4 schedule modes as P1. `dense_scale` multiplies non-terminal rewards (KO, HP advantage) and decays from 1.0 to a configurable minimum (default 0.25). Terminal rewards (±1.0 for win/loss) are never scaled. Default: linear decay starting at update 0, reaching 0.25 at the end of training.
- **Monitoring (P4)**: In addition to standard losses, `metrics.csv` tracks `grad_norm`, `explained_variance` (quality of V(s) fit), and `clip_frac` (fraction of PPO-clipped samples) for early detection of training instability.
- **CSV logging**: Every update appends to `metrics.csv` (losses, win rate, LR, grad_norm, clip_frac, explained_variance, mask_ratio, dense_scale). Evaluation results go to `eval.csv` (win rate per opponent type). Both files are written to the run's checkpoint directory.

### Opponent Pool & Mixing (P3)

The `OpponentPool` stores frozen copies of past checkpoints. Pool management has been enhanced:
- **Adaptive snapshots**: The agent is snapshotted to the pool only when `win_rate > 0.55` over the last 100 games, with a cooldown of 5 updates between snapshots — prevents flooding the pool with near-identical copies.
- **Increased pool size**: `pool_size=20` (up from 5) for greater opponent diversity.
- **Opponent mixing**: During rollout, the opponent is sampled as follows:
  - 80% from the opponent pool (past checkpoints)
  - 10% `RandomPolicy` (uniformly random legal actions)
  - 10% `FullOffensePolicy` (random legal move, no switches)
  
  This regularisation prevents overfitting to the pool and forces robustness against unpredictable or simplistic opponents.

---

## Module Details

### `env/state_encoder.py`

Converts raw `PyBattle.get_state()` dictionaries into structured feature objects.

- `encode_pokemon(poke_dict)` → `PokemonFeatures` with species/type/item/ability/move indices + 222 scalars
- `encode_side(side_dict)` → `SideFeatures` with 23 normalised conditions + counts
- `encode_field(state_dict)` → `FieldFeatures` with weather + terrain + pseudo-weather + 2× side conditions
- Vocabulary size constants for all categorical fields
- SVD embedding priors loaded from JSON
- 181 volatile condition flags indexed by name

### `env/revealed_tracker.py`

`RevealedTracker` tracks what information about opponent Pokémon has been revealed through natural gameplay events in the battle log. One tracker per parallel environment, reset on episode end.

**Log event detection:**
- `|switch|` / `|drag|` → species revealed (Pokémon switches in)
- `|-ability|` / `|-activate|...ability:` / `|-endability|` → ability revealed
- `|-enditem|` → item consumed/knocked off
- `|-heal|` / `|-damage|` with `[from] <item_name>` → item revealed (checked against `ITEM_INDEX`)
- `terastallized` flag in state → tera type revealed
- PP < maxPP in state → move revealed

**Per-slot tracking:** Maintains boolean arrays per env: `species[6]`, `item[6]`, `ability[6]`, `tera[6]`, `moves[6][4]`.

**Integration:** Called in `collect_rollout()` after each `make_choices()`. The reveal state is stored in each `Transition` and used during training to mask prediction loss targets and to mask model inputs via `apply_reveal_mask()`.

### `env/action_space.py`

`ActionEncoder` builds 13 action embeddings from the current turn's active token, move indices, bench tokens, and mechanic descriptor. Shares embedding tables with `PokemonEmbeddings` (same weight objects for moves and types).

### `model/embeddings.py`

`PokemonEmbeddings` projects categorical indices + scalar features into a flat 438-d token per Pokémon.

- **Species embeddings (32-d):** Initialised from 6 normalised base stats (HP + atk/def/spa/spd/spe). The first 6 PCA-like dimensions are set to the stats; the remaining 26 are learned from scratch.
- **Type embeddings (8-d):** Initialised from the 19×19 type-effectiveness matrix → SVD → 8 principal components.
- **Move embeddings (32-d):** Initialised from 46 move attributes (type, category, power, accuracy, PP, priority, etc.) → SVD → 32 components.
- **Item/Ability embeddings (16-d each):** Learned from scratch.
- **POMDP masking:** `apply_reveal_mask()` zero-masks unrevealed opponent categorical attributes during both rollout and training. Per-Pokémon Bernoulli sampling controlled by `mask_ratio`. When species is unknown, all attributes + scalars are masked. When species is known, types and base_stats remain visible (deterministic) while item/ability/tera/moves/stats/move_pp are masked per reveal state.

### `model/backbone.py`

`BattleBackbone` is the core Transformer stack:

- **Input projections:** `pokemon_proj` (Linear 438→256), `field_proj` (Linear 72→256)
- **Positional embeddings:** `temporal_emb` (Embedding 4→256, one per turn) + `slot_emb` (Embedding 13→256, one per slot)
- **Transformer:** 3 Pre-LN layers, 4 heads, FFN 512, dropout 0.1, batch_first, `enable_nested_tensor=False`
- **Value head:** MLP: Linear(13*256, 256) → ReLU → Linear(256, 1)
- **Actor:** Cross-attention (Q=action_embeds, K=V=current_tokens) → Linear(256, 1) per slot → [B, 13] logits
- **Attention maps:** `get_attention_maps()` iterates each transformer layer manually to capture attention weights with `need_weights=True`, bypassing PyTorch's MHA fastpath

### `model/prediction_heads.py`

Four independent Linear heads applied to the 6 opponent tokens (`current_tokens[:, 6:12]`):

| Head | Classes | Weight Shape |
|------|---------|-------------|
| item | 250 | Linear(256, 250) |
| ability | 311 | Linear(256, 311) |
| tera | 19 | Linear(256, 19) |
| moves | 686 | Linear(256, 2744) → reshape [B, 6, 4, 686] |

The `moves` head uses a single Linear layer to predict all 4 move slots jointly. Loss is masked cross-entropy — only revealed Pokémon (non-zero indices) contribute to the gradient. With P1 POMDP masking, the loss masks come from the `RevealedTracker` (actual gameplay reveals) rather than from UNK indices. No belief injection in v1 (see `BELIEF_STATE.md` for v2).

### `model/agent.py`

`CynthAIAgent(nn.Module)` orchestrates the full forward pass:

```python
pokemon_tokens = poke_emb(poke_batch)                      # [B, 48, 438]
current_tokens, value = backbone.encode(pokemon_tokens,     # [B, 13, 256], [B, 1]
                                         field_tensor)
action_embeds = action_enc(active_token, move_idx, ...,     # [B, 13, 256]
                            bench_tokens, mechanic_id)
action_logits = backbone.act(action_embeds,                  # [B, 13]
                              current_tokens, action_mask)
pred_logits = predictor(current_tokens[:, 6:12])             # 4 prediction heads
```

### `training/rollout.py`

- `collect_rollout()`: Core collection function. Runs N parallel `PyBattle` instances, builds batched inputs for both agents, samples actions, calls `make_choices()`, computes rewards, and stores transitions. Runs under `torch.no_grad()`. After each `make_choices()`, fetches log entries via `get_new_log_entries()` and updates a `RevealedTracker` per env.
- **Switch sub-turn handling:** Pokémon Showdown enters a sub-turn after a KO or forced switch (U-turn, Roar, etc.). One side receives `request_state="Switch"` while the other receives `"None"`. These are detected per-env during the main collection loop and deferred to a dedicated forward pass:
  1. **Detection:** After re-reading the fresh state, envs with `req_self="None"` or `req_opp="None"` are collected into `subt_envs`. If both sides are `"None"` (battle doesn't expect input), the step is skipped. If one side has `"Switch"`, the env is queued.
  2. **Sub-turn forward pass:** After the main per-env loop, queued sub-turn envs are grouped by which side needs to switch. For `side_self` sub-turns, `_build_agent_inputs()` builds a mini-batch from the envs' `BattleWindow` histories, `agent_self()` produces log_probs and value estimates, and `_sample_action()` samples a switch action (slots 8–12).
  3. **Execution:** The sampled switch choice is sent via `make_choices()`. A full `Transition` is recorded (species_idx, type1_idx, ..., action, log_prob_old, action_mask, reward, done, value_old) — same fields as a normal turn.
  4. **No fallbacks:** If the model's chosen switch is rejected by the Rust sim (e.g., position validation bug in `choose_switch.rs`), the environment is reset. No random default or placeholder is used — the model must learn valid switches through masking.
  5. **Opponent sub-turns:** For `side_opp` sub-turns, `agent_opp` (neural or bot) handles the switch. Bot policies use `.act()` directly; neural opponents get the same forward pass treatment.
- After sub-turn processing, `skip_buffer` marks all sub-turn envs so the main transition recording loop skips them (their transitions were already recorded by the sub-turn handler).
- `BattleWindow`: Per-environment sliding window (K=4). Zero-padded for early turns.
- `RolloutBuffer`: Stores transitions on CPU. `compute_gae()` computes GAE (γ=0.99, λ=0.95) backwards over complete episodes. `minibatches()` yields shuffled batches. `_gather()` includes reveal state tensors (`reveal_species`, `reveal_item`, `reveal_ability`, `reveal_tera`, `reveal_moves` — all bool).
- `build_action_mask()`: Derives legal action mask from state dict. Returns `all-True` (all illegal) when `request_state == "None"`, preventing the model from acting in inconsistent sim states. Switch-only mask when `request_state == "Switch"`.
- `action_to_choice()`: Converts action index (0–12) to PS choice string (`"move 1"`, `"switch 3"`, `"move 1 terastallize"`).
- `RandomPolicy`: Lightweight policy for evaluation against random opponents. Samples uniformly from legal actions.
- `compute_step_reward()`: Computes dense reward from prev/curr state delta.

### `training/losses.py`

`compute_losses()` returns a dict of loss components:

- **policy_loss:** `-E[min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)]` where `ratio = exp(logπ_new - logπ_old)`
- **value_loss:** `MSE(V(s), returns)` where returns = GAE + V_old(s)
- **entropy_loss:** `-Σ p(a)·log p(a)` over legal actions only (maximising entropy = minimising this)
- **pred_loss:** Sum of masked cross-entropy over all 4 prediction heads
- **total:** `policy + 0.5·value + 0.01·entropy + 0.5·pred`

### `training/self_play.py`

Main entry point. `train(TrainingConfig)` implements the full self-play loop with checkpointing, opponent pool management, CSV logging, POMDP masking curriculum (P1), reward curriculum (P2), and optional resume from checkpoint. Every update appends to `metrics.csv`; evaluation results go to `eval.csv`.

Both P1 and P2 support 4 schedule modes via `compute_mask_ratio()` and `compute_dense_scale()`:

| Mode | Description | When to use |
|------|-------------|-------------|
| `linear` | Ramp/decay linearly over `[warmup, total]` | Simple, predictable curriculum |
| `exp` | Exponential curve — fast initial change | When you want a quick transition |
| `step` | Abrupt switch at `step_update` | When you want distinct regimes |
| `phase` | Manual breakpoints with custom values | Full control (e.g. Fondations → Transition → Maîtrise) |

The `phase` mode takes `(breakpoints, values)` tuples. Example:
```
mask_phase_breakpoints=(600, 2500), mask_phase_values=(0.0, 0.5, 1.0)
```
→ updates [0, 600] mask=0.0, (600, 2500] mask=0.5, (2500, ∞) mask=1.0

### `training/evaluate.py`

Evaluates a checkpoint against a random or checkpoint opponent. Reports win rate with Wilson 95% confidence interval. Uses `collect_rollout` with sufficient min_steps to collect complete battles. The evaluation masking ratio matches the training masking ratio at the time of evaluation.

### `training/visualize.py`

Reads `metrics.csv` and generates 2×3 grid plots (win rate, policy/value/entropy/pred loss, learning rate) or a single overlay figure with twin y-axes.

---

## Battle Log & Revealed Information

The Rust simulator maintains a battle log in Pokémon Showdown protocol format (`self.log: Vec<String>`), recording all game events (switches, ability activations, item consumption, weather changes, etc.). Each log entry follows PS format: `|switch|p2a: Zacian|...`, `|-ability|p2a|intrepidsword|...`, `|-enditem|p2a|Leftovers|[from] move: Knock Off`, etc.

### Python Access

The `PyBattle` binding exposes `get_new_log_entries()` which returns all log entries since the last call and advances the read position (`sent_log_pos`). This is called in `collect_rollout()` after each `make_choices()`.

### Revealed State Tracking

`RevealedTracker` (`env/revealed_tracker.py`) parses the battle log to detect reveals:

| Event | Log Pattern | Attribute Revealed |
|-------|-------------|-------------------|
| Switch-in | `\|switch\|` / `\|drag\|` | Species |
| Ability activation | `\|-ability\|` / `\|-activate\|...ability:` | Ability |
| Ability suppression | `\|-endability\|` | Ability (was active, now known) |
| Item consumed/removed | `\|-enditem\|` | Item |
| Item-based healing | `\|-heal\|...\|[from] leftovers\|` | Item (checked vs ITEM_INDEX) |
| Item-based damage | `\|-damage\|...\|[from] lifeorb\|` | Item (checked vs ITEM_INDEX) |
| Terastallization | `terastallized` flag in state | Tera type |
| Move used | PP < maxPP in state | Move |

### Masking Logic

`apply_reveal_mask()` in `model/embeddings.py` applies the reveal state to model inputs:

- **Species unknown** → all categoricals + all scalars zeroed (Pokémon is completely unknown)
- **Species known** → types and base_stats kept (deterministic from species); item/ability/tera/moves masked if not yet revealed; scalars `stats[8:13]` always masked (actual stats are non-observable); `move_pp[21:24]` masked per-slot when the move is unrevealed (PP leaks move identity via maxPP)
- **Per-Pokémon Bernoulli** with `mask_ratio` controls whether masking is applied at all (curriculum schedule)

The unmasked ground-truth is stored separately in the Transition for prediction head targets. The prediction loss masks are overridden with the actual reveal state so only gameplay-revealed attributes contribute to the gradient.

### `training/visualize.py`

Reads `metrics.csv` and generates 2×3 grid plots (win rate, policy/value/entropy/pred loss, learning rate) or a single overlay figure with twin y-axes.

---

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr` | 2.5e-4 | AdamW, weight_decay=1e-4, eps=1e-5 |
| `lr_min` | 1e-5 | Cosine decay endpoint |
| `warmup_steps` | 20 | Linear warmup before cosine decay |
| `total_updates` | 5000 | Total PPO updates |
| `n_envs` | 32 | Parallel battles |
| `min_steps` | 1024 | Min transitions per rollout |
| `n_epochs` | 2 | PPO epochs per rollout |
| `batch_size` | 128 | Minibatch size |
| `gamma` | 0.99 | GAE discount |
| `lam` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clipping ε |
| `c_value` | 1.0 | Value loss coefficient |
| `c_entropy` | 0.02 | Entropy bonus coefficient |
| `c_pred` | 0.6 | Predictor auxiliary coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `weight_decay` | 1e-4 | AdamW L2 regularisation |
| `pool_size` | 30 | Max opponent snapshots |
| `pool_snapshot_threshold` | 0.55 | Win-rate threshold for snapshot |
| `pool_cooldown` | 5 | Min updates between snapshots |
| `eval_freq` | 100 | Run evaluation every N updates |
| `eval_n_games` | 10 | Games per opponent in evaluation |
| `checkpoint_freq` | 100 | Save .pt every N updates |
| `log_every` | 1 | Log every update |
| `mask_schedule` | `"phase"` | P1 schedule mode: linear, exp, step, phase |
| `mask_phase_breakpoints` | `(600, 2500)` | Update breakpoints for phase mode |
| `mask_phase_values` | `(0.0, 0.5, 1.0)` | Mask ratios at each phase |
| `dense_schedule` | `"phase"` | P2 schedule mode |
| `dense_phase_breakpoints` | `(600, 2500)` | Update breakpoints for phase mode |
| `dense_phase_values` | `(1.0, 0.5, 0.1)` | Dense_scale at each phase |

---

## Installation

### Prerequisites

- Python 3.10+
- Rust toolchain (for simulator compilation)
- PyTorch 2.0+ (CUDA optional but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/AtwMaxime/CynthAI_v2.git
cd CynthAI_v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows

# Install Python dependencies
pip install torch numpy matplotlib

# Build and install the Rust simulator
cd simulator
pip install maturin
maturin develop --release
cd ..

# Generate vocabulary dictionaries
python data/build_dicts.py
```

### Verify Installation

```bash
python test_encoder.py
python test_full_pipeline.py
python test_attention_maps.py
```

---

## Usage

### Training

```bash
# Default settings (2000 updates)
python -m training.self_play

# Custom configuration
python -m training.self_play \
    --device cuda \
    --n-envs 32 \
    --total-updates 5000 \
    --lr 3e-4
```

Or programmatically:

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

### Evaluation

```bash
# Against random opponent
python -m training.evaluate \
    --checkpoint checkpoints/agent_best.pt \
    --opponent random \
    --n-battles 200

# Against another checkpoint
python -m training.evaluate \
    --checkpoint checkpoints/agent_000600.pt \
    --opponent checkpoints/agent_best.pt \
    --n-battles 200
```

### Visualization

```bash
python -m training.visualize
```

### Loading a Checkpoint

```python
import torch
from model.agent import CynthAIAgent

ckpt  = torch.load("checkpoints/agent_001000.pt")
agent = CynthAIAgent()
agent.load_state_dict(ckpt["model"])
agent.eval()
```

### Attention Map Extraction

```python
from simulator import PyBattle
from training.rollout import BattleWindow, encode_state
from model.agent import CynthAIAgent
from model.backbone import K_TURNS

b = PyBattle("gen9randombattle", 42)
window = BattleWindow()
for _ in range(K_TURNS):
    pf, ff = encode_state(b.get_state(), side_idx=0)
    window.push(pf, ff)

poke_batch, field_tensor = build_inputs(window)  # see test_attention_maps.py
agent = CynthAIAgent()
result = agent.backbone.get_attention_maps(poke_batch, field_tensor)

for i, attn in enumerate(result["attention_maps"]):
    print(f"Layer {i}: {attn.shape}")  # [1, 4, 52, 52]
```

---

## Training Results

### Run 1 (updates 1–500)

- **Peak win rate:** ~70% against past self (update 103)
- **Final win rate:** ~56% against past self (update 500)
- **Pool size:** 5 (full)
- **Learning rate:** Decayed from 3e-4 to ~2.27e-4

**Pattern:** Rapid initial improvement (win rate spikes to 0.70 by update 103), followed by partial policy collapse (down to 0.44 around update 200), then gradual recovery and stabilisation around 0.55–0.65.

### Run 2 (updates 501–621)

- **Starting win rate:** 33% (pool reset to 0, fresh self-play)
- **Peak win rate:** 68% (update 575)
- **Final win rate:** ~49% (update 621)

**Pattern:** Faster learning than Run 1 but similar peak-then-decline oscillation, suggesting an underlying instability (possibly the side-conditions encoder bug, now fixed).

### Analysis

The peak-then-decline pattern suggests:
1. The agent initially learns basic combat (type matchups, strong moves)
2. Over time, it exploits weaknesses in its past self that don't generalise
3. The critic (value head) struggles to stabilise, oscillating between 0.01 and 0.10 loss
4. The side-conditions encoder bug (now fixed) prevented learning of hazards/screens, limiting strategic depth

---

## v2 Roadmap

| Feature | Status | Description |
|---------|--------|-------------|
| Rust simulator integration | ✅ Done | PyO3 bindings for fast battle simulation |
| State encoder (sections 1–11) | ✅ Done | Full Gen 9 state → feature pipeline |
| Pokémon / field embeddings | ✅ Done | SVD-prior initialised embeddings |
| Transformer backbone | ✅ Done | 3-layer Pre-LN, d_model=256 |
| Action encoder (all mechanics) | ✅ Done | Universal Tera/Mega/Z/Dmax support |
| Prediction heads | ✅ Done | Item/ability/tera/move predictors |
| Attention map extraction | ✅ Done | Per-layer attention visualisation |
| PPO rollout + GAE | ✅ Done | Complete-episode collection |
| PPO losses | ✅ Done | Clip + value + entropy + pred |
| Self-play training loop | ✅ Done | Opponent pool, checkpointing, logging |
| **P3 — Pool size + opponent mixing** | ✅ Done | Pool=20, adaptive snapshots, 80/10/10 mixing |
| **P4 — Regularisation & monitoring** | ✅ Done | AdamW, grad clipping, grad_norm/explained_variance/clip_frac logging |
| **P5 — Critic optimisation** | ✅ Done | c_value=1.0, return normalisation |
| **P6 — Warmup + cosine LR** | ✅ Done | 20-step warmup, cosine decay from 2.5e-4 |
| **P1 — POMDP masking (opponent info)** | ✅ Done | RevealedTracker + apply_reveal_mask + 4-mode curriculum |
| **P2 — Reward curriculum** | ✅ Done | 4-mode dense_scale scheduler (linear/exp/step/phase) |
| **Belief state injection** | 🔜 v2 | Argmax + softmax confidence → state encoder |
| **Pre-training on team sets** | 🔜 v2 | 162k team compositions |
| **Elo-based opponent selection** | 🔜 v2 | Adaptive opponent difficulty |

---

## Known Issues

- **Simulator stability:** The Rust simulator (`pokemon-showdown-rs`) can panic with `"Not all choices done"` when the choice string doesn't match the game state. Four fixes were applied to the Rust engine (request_state guards, terastallize suffix parsing, forced_switches_left handling) and the action mask now uses `request_state` from the state dict as source of truth. Residual panics are caught with try/except in `collect_rollout` and are rare (< 0.03% of steps).
- **Rust simulator fixes (local):** Three bugs in the unmaintained `pokemon-showdown-rs-master` were fixed locally:
  1. `get_choice_index`: auto-pass conditionnel en mode Switch (évite "You sent more switches than needed")
  2. `choose_switch`: validation de Pokémon actif corrigée (utilise `active.contains()` au lieu de `position < active.len()`)
  3. Exposition de `slot_conditions` à Python pour détecter Revival Blessing
  Détails : voir [`memo_rust.md`](memo_rust.md)
- **Fastpath incompatibility:** PyTorch 2.11+ MHA fastpath bypasses Python-level `_sa_block`. The `get_attention_maps()` method uses a manual layer-by-layer forward pass instead, calling `self_attn()` directly to capture weights.

---

## Documentation

| File | Contents |
|------|----------|
| `ARCHITECTURE.md` | Detailed forward-pass diagram with tensor shapes |
| `POKEMON_VECTOR.md` | Complete Pokémon token: 11 sections + scalar layout |
| `FIELD.md` | Field token: all 72 dimensions explained |
| `BELIEF_STATE.md` | v2 opponent belief injection design + activation plan |
| `README.md` | This file — full project overview |