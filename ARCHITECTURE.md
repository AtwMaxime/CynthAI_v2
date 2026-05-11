# CynthAI_v2 — Architecture Document

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CynthAIAgent (agent.py)                         │
│                                                                         │
│   State Dict ──▶ PokemonEmbeddings ──▶ BattleBackbone ──┬──▶ Value      │
│   (PyBattle)      (embeddings.py)     (backbone.py)     │               │
│                                                         ├──▶ Logits     │
│                    ActionEncoder ◀── current_tokens ────┤               │
│                    (action_space.py)                    │               │
│                                                         └──▶ Preds      │
│                    PredictionHeads ◀── opp_tokens ──────┘               │
│                    (prediction_heads.py)                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: State → Token → Transformer → Action

```
Step 1: Encode            Step 2: Embed              Step 3: Transformer
┌──────────────┐         ┌──────────────┐           ┌──────────────────────┐
│ PyBattle     │         │ PokemonEmb   │           │ BattleBackbone       │
│ .get_state() │────────▶│ .forward()   │──────────▶│ .encode()            │
│              │         │              │           │                      │
│ dict with:   │         │ species(32)  │           │ _build_sequence()    │
│  sides[0,1]  │         │ type1(8)     │           │  ├─ project poke,    │
│  pokemon[6]  │         │ type2(8)     │           │  │  field tokens     │
│  field state │         │ tera_type(8) │           │  ├─ reshape to       │
│  weather     │         │ item(16)     │           │  │  [B,K,13,D_MODEL] │
│  terrain     │         │ ability(16)  │           │  ├─ add temporal_emb │
│  ...         │         │ moves[4](32) │           │  └─ add slot_emb     │
└──────────────┘         │ scalars(222) │           │                      │
                         │ = 438 dims   │           │ TransformerEncoder   │
                         └──────────────┘           │  3 layers × 4 heads  │
                                                    │  Pre-LN, d_model=256 │
                                                    │  FFN=512, dropout=0.1│
                                                    │                      │
                                                    │ Output:              │
                                                    │  current_tokens      │
                                                    │  [B, 13, 256]        │
                                                    └──────┬───────────────┘
                                                           │
       ┌──────────────────────┌────────────────────────────┤
       ▼                      ▼                            ▼
Step 4a: Value Head          Step 4b: Actor Head           Step 4c: Prediction Heads
┌──────────────────┐   ┌──────────────────────────┐   ┌──────────────────────┐
│ value_head       │   │ action_cross_attn +      │   │ PredictionHeads      │
│                  │   │ action_score             │   │                      │
│ mean_pool(13)    │   │                          │   │ opp_tokens[:,6:12,:] │
│   ↓              │   │ Query: action_embeds     │   │        ↓             │
│ [B, 256]         │   │   [B, 13, 256]           │   │ item_head    (Linear)│
│   ↓              │   │ Key/Value: current_tokens│   │ ability_head (Linear)│
│ Linear→ReLU→     │   │   [B, 13, 256]           │   │ tera_head    (Linear)│
│ Linear→1         │   │        ↓                 │   │ move_head    (Linear)│
│   ↓              │   │ MultiheadAttention(4)    │   │        ↓             │
│ V(s) [B, 1]      │   │        ↓                 │   │ item [B,6,250]       │
└──────────────────┘   │ Linear → scalar          │   │ abil [B,6,311]       │
                       │        ↓                 │   │ tera [B,6,19]        │
                       │ logits [B, 13]           │   │ moves [B,6,4,686]    │
                       │ masked_fill(illegal,-1e9)│   └──────────────────────┘
                       └──────────────────────────┘
```

## Token Sequence Layout

```
K=4 turns, 13 tokens per turn, 52 tokens total

Turn 0 (oldest)              Turn K-1 (current)
├──────────┼──────────┤     ├──────────┼──────────┤
│ OWN 0-5  │ OPP 6-11 │ ... │ OWN 0-5  │ OPP 6-11 │
│          │          │     │          │          │
│ [active, │ [active, │     │ [active, │ [active, │
│  bench×5]│  bench×5]│     │  bench×5]│  bench×5]│
└──────────┴──────────┘     └──────────┴──────────┘
    12 pokemon tokens     +      1 field token     = 13 per turn

Positional Embeddings (additive):
  temporal_emb: Embedding(K=4, d_model=256)  — which turn
  slot_emb:     Embedding(13, d_model=256)   — which position
```

## DETR-Style Query-Key Cross-Attention (Actor Head)

The actor head follows the same pattern as DETR (DEtection TRansformer):

```
DETR Analogy:
  Object Queries ──▶ Cross-Attention ◀── Encoder Output
       │                    │                   │
  "what object?"      matching          image features
       
CynthAI Actor Head:
  Action Queries ──▶ Cross-Attention ◀── Current Tokens
       │                    │                   │
  "what action?"       matching          battle state tokens
  [B, 13, 256]                           [B, 13, 256]
```

### How Action Queries Are Built

Each of the 13 action slots gets its own query embedding, constructed from live battle context:

```
ActionEncoder.forward():

  Move slots (0-3):                    Mechanic slots (4-7):
  ┌────────────────────┐              ┌────────────────────────┐
  │ move_emb(move_idx) │   [B,4,32]   │ base_moves             │ [B,4,256]
  │ active_token       │   [B,4,256]  │   +                    │
  │ [pp_ratio, disab]  │   [B,4,2]    │ mechanic_proj(         │
  │        ↓           │              │   type_emb(tera_type)  │ [B,8]
  │ move_proj(Linear)  │              │   +                    │
  │        ↓           │              │   one_hot(mechanic_id) │ [B,5]
  │ base_moves [B,4,256]│             │ )                      │ [B,256]
  └────────────────────┘              │        ↓               │
                                      │ mech_moves [B,4,256]   │
  Switch slots (8-12):                └────────────────────────┘
  ┌────────────────────┐
  │ bench_tokens[i]    │   [B,5,256]
  │        ↓           │
  │ switch_proj(Linear)│
  │        ↓           │
  │ switch_acts[B,5,256]│
  └────────────────────┘

  Concatenate: [moves(4) | mechanic(4) | switch(5)] → [B, 13, D_MODEL]
```

### Cross-Attention: Action Embeddings (Query) over Current Tokens (Key/Value)

```
     Action Embeddings (Q)         Current Tokens (K,V)
     ┌─────────────────┐          ┌─────────────────┐
     │ Move 1 ─────────┼─────────▶│ OWN_active      │
     │ Move 2 ─────────┼─────────▶│ OWN_bench_1     │
     │ Move 3 ─────────┼─────────▶│ OWN_bench_2     │
     │ Move 4 ─────────┼─────────▶│ OWN_bench_3     │
     │ Tera 1 ─────────┼─────────▶│ OWN_bench_4     │
     │ Tera 2 ─────────┼─────────▶│ OWN_bench_5     │
     │ Tera 3 ─────────┼─────────▶│ OPP_active      │
     │ Tera 4 ─────────┼─────────▶│ OPP_bench_1     │
     │ Switch 1 ───────┼─────────▶│ OPP_bench_2     │
     │ Switch 2 ───────┼─────────▶│ OPP_bench_3     │
     │ Switch 3 ───────┼─────────▶│ OPP_bench_4     │
     │ Switch 4 ───────┼─────────▶│ OPP_bench_5     │
     │ Switch 5 ───────┼─────────▶│ FIELD           │
     └─────────────────┘          └─────────────────┘
          [B, 13, 256]                 [B, 13, 256]

     │                                              │
     └────────── MultiheadAttention(4) ─────────────┘
                         │
                         ▼
              Linear → scalar per slot
                         │
                         ▼
              action_logits [B, 13]
         (masked_fill with -1e9 for illegal)
```

### Why DETR-style?

- **Single Transformer pass**: The backbone runs once to produce rich token representations. The actor head does NOT re-encode anything — it just cross-attends action queries over the already-computed tokens. This is the same efficiency pattern as DETR.
- **Contextualized actions**: Each action embedding can attend to any token (own Pokémon, opponent Pokémon, field), learning which parts of the state matter for each action type.
- **Shared representation**: Value head and prediction heads also consume the same token representations, encouraging the Transformer to learn generally useful features.

## Prediction Heads

```
Opponent tokens [B, 6, D_MODEL]
       │
       ├──▶ item_head    Linear(256 → 250)   → item_logits    [B, 6, 250]
       ├──▶ ability_head Linear(256 → 311)   → ability_logits [B, 6, 311]
       ├──▶ tera_head    Linear(256 → 19)    → tera_logits    [B, 6, 19]
       └──▶ move_head    Linear(256 → 686×4) → move_logits    [B, 6, 4, 686]
```

Linear heads only (no MLPs) — the Transformer tokens already encode rich contextual information. The auxiliary loss provides a supervised signal that helps the backbone learn useful opponent representations, but gradients from these heads do NOT flow into the policy directly (only through shared backbone representations).

## POMDP Masking System

```
RevealedTracker (per env)
  │
  │  Parses PS protocol battle log:
  │    |switch|p2a: Garchomp     → species revealed
  │    |-ability|p2a|Intimidate   → ability revealed
  │    |-enditem|p2a|Leftovers   → item revealed
  │    terastallized flag         → tera revealed
  │    PP < maxPP                 → move revealed
  │
  ▼
Reveal state [6 slots]:
  species: [True, False, False, False, False, True]
  item:    [True, False, False, False, False, False]
  ability: [True, False, False, False, False, False]
  tera:    [False, ...]
  moves:   [[True, True, False, False], [False, ...], ...]

apply_reveal_mask(PokemonBatch, reveal_state, mask_ratio):
  │
  │  For each opponent slot:
  │    Bernoulli(mask_ratio) → should_mask
  │    if should_mask AND NOT revealed → zero the attribute
  │
  ▼
Masked PokemonBatch (zeros for unrevealed indices + stats)
```

## PPO Training Loop

```
train() in self_play.py

for update in 1..total_updates:

  ┌─ 1a. Opponent Selection ─────────────────────────┐
  │   80% pool.sample(agent)                          │
  │   10% RandomPolicy                                │
  │   10% FullOffensePolicy                           │
  └───────────────────────────────────────────────────┘
       │
  ┌─ 1b. Rollout Collection (rollout.py) ─────────────┐
  │   collect_rollout(agent_self, agent_opp, n_envs)  │
  │                                                    │
  │   For each env in parallel:                        │
  │     encode_state() → push to BattleWindow (K=4)   │
  │     _build_agent_inputs() → model inputs           │
  │     POMDP masking (apply_reveal_mask)              │
  │     agent.forward() → action sample                │
  │     action_to_choice() → PS protocol               │
  │     PyBattle.make_choices()                        │
  │     compute_step_reward()                          │
  │     tracker.update() from new log entries          │
  │     → Transition stored in RolloutBuffer           │
  │                                                    │
  │   RolloutBuffer.compute_gae(γ=0.99, λ=0.95)       │
  └───────────────────────────────────────────────────┘
       │
  ┌─ 2. PPO Updates (n_epochs × minibatches) ────────┐
  │   For each minibatch:                             │
  │     Re-apply POMDP mask (training augmentation)    │
  │     agent.forward(masked_poke, ...)                │
  │     PredictionHeads.compute_loss()                 │
  │     compute_losses() → PPO + value + entropy + pred│
  │     loss.backward()                                │
  │     clip_grad_norm_(0.5)                           │
  │     optimizer.step()                               │
  └───────────────────────────────────────────────────┘
       │
  ┌─ 3. LR Schedule ──────────────────────────────────┐
  │   Warmup (linear 0→lr, 20 steps)                  │
  │   Cosine decay (lr → lr_min)                       │
  └───────────────────────────────────────────────────┘
       │
  ┌─ 4. Logging (stdout + metrics.csv) ───────────────┐
  └───────────────────────────────────────────────────┘
       │
  ┌─ 5. Evaluation (every eval_freq) ─────────────────┐
  │   500 games vs Random, FullOffense, Pool           │
  │   Snapshot agent if pool_wr > 0.55                 │
  │   Save diagnostic plots + eval.csv                 │
  └───────────────────────────────────────────────────┘
       │
  ┌─ 6. Checkpoint (every checkpoint_freq) ───────────┐
  │   agent_NNNNNN.pt + attention maps                 │
  └───────────────────────────────────────────────────┘
```

## Token Embedding Detail (438-dim)

```
┌──────────┬──────┬──────────────────────────────────────┐
│ Field    │ Dims │ Source                                │
├──────────┼──────┼──────────────────────────────────────┤
│ Species  │ 32   │ Embedding(1381, 32), seeded from BST  │
│ Type 1   │ 8    │ Embedding(19, 8), seeded from log2    │
│ Type 2   │ 8    │ (shared weight)                       │
│ Tera     │ 8    │ (shared weight)                       │
│ Item     │ 16   │ Embedding(250, 16)                    │
│ Ability  │ 16   │ Embedding(311, 16)                    │
│ Move 1   │ 32   │ Embedding(686, 32), type+cat prior    │
│ Move 2   │ 32   │ (shared weight)                       │
│ Move 3   │ 32   │ (shared weight)                       │
│ Move 4   │ 32   │ (shared weight)                       │
│ Scalars  │ 222  │ Raw float32 values (no embedding)     │
├──────────┼──────┼──────────────────────────────────────┤
│ TOTAL    │ 438  │                                       │
└──────────┴──────┴──────────────────────────────────────┘

Scalars breakdown (222 values):
  level(1) + hp_ratio(1) + terastallized(1) + base_stats(5) + stats(5)
  + is_predicted(1) + boosts(7) + move_pp(4) + move_disabled(4)
  + status(7) + volatiles(181) + is_active(1) + fainted(1) + trapped(1)
  + force_switch(1) + revealed(1)
```

## Field Token (72-dim)

```
┌───────────────┬──────┬──────────────────────────┐
│ Field         │ Dims │ Description              │
├───────────────┼──────┼──────────────────────────┤
│ Weather       │ 9    │ One-hot (sun, rain, etc) │
│ Terrain       │ 5    │ One-hot (grassy, etc)    │
│ Pseudo-weather │ 8    │ Multi-hot (trick room)  │
│ Side 0 conds  │ 23   │ Side conditions one-hot  │
│ Side 0 misc   │ 2    │ pokemon_left, fainted    │
│ Side 1 conds  │ 23   │ Side conditions one-hot  │
│ Side 1 misc   │ 2    │ pokemon_left, fainted    │
├───────────────┼──────┼──────────────────────────┤
│ TOTAL         │ 72   │                          │
└───────────────┴──────┴──────────────────────────┘
```

## Model Size Breakdown

| Module | Parameters |
|--------|-----------|
| PokemonEmbeddings (7 tables) | ~184K |
| BattleBackbone (Transformer + projections) | ~3.1M |
| Value Head (Linear-ReLU-Linear) | ~65K |
| Action Cross-Attention + score | ~330K |
| ActionEncoder (move/mech/switch proj) | ~200K |
| PredictionHeads (4 Linear heads) | ~2.8M |
| **Total** | **~6.7M** |

## Key Design Decisions

1. **Single Transformer pass**: `encode()` runs the Transformer once; `act()` only does lightweight cross-attention. This keeps inference fast (important for 32 parallel envs).

2. **Mean-pool for value**: Simple averaging over 13 current-turn tokens instead of flattening (3328→256 would add parameters). Pooling is order-invariant and costs zero parameters.

3. **Shared embedding weights**: `ActionEncoder` shares `move_embed` and `type_embed` with `PokemonEmbeddings`, ensuring consistent representations and fewer parameters.

4. **Pre-LN Transformer**: Layer norm before attention/FFN sublayers (not after), which is more stable for training and is standard in modern architectures.

5. **Linear prediction heads**: No MLPs on prediction heads — the Transformer tokens are already rich enough. MLPs would overfit on the sparse reveal signal.

6. **PP on state, not PP remaining**: Encoded as ratio (pp/maxpp) for meaningful comparison across different moves (a move at 1/5 is very different from a move at 1/40).