# CynthAI_v2 — Architecture Document

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CynthAIAgent (agent.py)                         │
│                                                                         │
│   State Dict ──▶ PokemonEmbeddings ──▶ BattleBackbone ──┬──▶ Value      │
│   (PyBattle)      (embeddings.py)     (backbone.py)     │               │
│                                                         ├──▶ Logits     │
│                    ActionEncoder ◀── pre_tokens ─────────┤               │
│                    (action_space.py)  (pre-Transformer)  │               │
│                                                         └──▶ Preds      │
│                    PredictionHeads ◀── post_tokens ──────┘               │
│                    (prediction_heads.py)  (post-Transformer)            │
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
└──────────────┘         │ scalars(223) │           │                      │
                         │ = 439 dims   │           │ TransformerEncoder   │
                         └──────────────┘           │  3 layers × 4 heads  │
                                                    │  Pre-LN, d_model=256 │
                                                    │  FFN=512, dropout=0.15│
                                                    │                      │
                                                    │ Output:              │
                                                    │  pre_tokens  [B,13,256]  ← before Transformer
                                                    │  post_tokens [B,13,256]  ← after Transformer
                                                    └──────┬───────────────┘
                                                           │
       ┌──────────────────────┌────────────────────────────┤
       ▼                      ▼                            ▼
Step 4a: Value Head    Step 4b: Actor Head          Step 4c: Prediction Heads
┌────────────────┐   ┌─────────────────────────┐   ┌──────────────────────┐
│ value_head     │   │ ActionEncoder(pre_tokens)│   │ PredictionHeads      │
│                │   │  + action_cross_attn     │   │                      │
│ mean_pool over │   │  + action_score          │   │ post_tokens[:,6:12,:]│
│ post_tokens(13)│   │                          │   │        ↓             │
│   ↓            │   │ Query: action_embeds     │   │ item_head    (Linear)│
│ [B, 256]       │   │   [B, 13, 256]           │   │ ability_head (Linear)│
│   ↓            │   │ Key/Value: post_tokens   │   │ tera_head    (Linear)│
│ Linear(256,256)│   │   [B, 13, 256]           │   │ move_head    (Linear)│
│ ReLU           │   │        ↓                 │   │ stats_head   (Linear)│
│ Linear(256,1)  │   │ MultiheadAttention(4)    │   │        ↓             │
│   ↓            │   │        ↓                 │   │ item [B,6,250]       │
│ V(s) [B, 1]   │   │ Linear → scalar          │   │ abil [B,6,311]       │
└────────────────┘   │        ↓                 │   │ tera [B,6,19]        │
                     │ logits [B, 13]           │   │ moves[B,6,686] (BCE) │
                     │ masked_fill(illegal,-1e9)│   │ stats[B,6,6]  (MSE)  │
                     └─────────────────────────┘   └──────────────────────┘
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

Positional Embeddings (additive, learned):
  temporal_emb: Embedding(K=4, d_model=256)  — which turn (0=oldest, 3=current)
  slot_emb:     Embedding(13, d_model=256)   — which position within turn
```

## DETR-Style Query-Key Cross-Attention (Actor Head)

The actor head follows the same pattern as DETR (DEtection TRansformer):

```
DETR Analogy:
  Object Queries ──▶ Cross-Attention ◀── Encoder Output
       │                    │                   │
  "what object?"      matching          image features

CynthAI Actor Head:
  Action Queries ──▶ Cross-Attention ◀── Current Tokens (post-Transformer)
       │                    │                   │
  "what action?"       matching          battle state tokens
  [B, 13, 256]                           [B, 13, 256]
```

### How Action Queries Are Built

Action queries are built from **pre-Transformer** tokens (raw projected features, no self-attention)
to prevent the active Pokémon token from dominating queries via self-match shortcuts (P13e).

```
ActionEncoder.forward():

  Move slots (0-3):                    Mechanic slots (4-7):
  ┌─────────────────────┐             ┌────────────────────────┐
  │ move_emb(move_idx)  │  [B,4,32]   │ base_moves             │ [B,4,256]
  │ [pp_ratio, disabled]│  [B,4,2]    │   +                    │
  │        ↓            │             │ mechanic_proj(         │
  │ move_proj(Linear    │             │   type_emb(tera_type)  │ [B,8]
  │   D_MOVE+2→D_MODEL) │             │   + one_hot(mech_id)   │ [B,5]
  │        ↓            │             │ )                      │ [B,256]
  │ base_moves [B,4,256]│             │        ↓               │
  └─────────────────────┘             │ mech_moves [B,4,256]   │
                                      └────────────────────────┘
  Switch slots (8-12):
  ┌─────────────────────┐
  │ bench pre_tokens[i] │  [B,5,256]
  │        ↓            │
  │ switch_proj(Linear) │
  │        ↓            │
  │ switch_acts[B,5,256]│
  └─────────────────────┘

  Concatenate: [regular(4) | mechanic(4) | switch(5)] → [B, 13, D_MODEL]
```

Note: `active_token` was removed from move_proj input (P22) — it caused all move queries to
collapse towards the same representation, losing move identity.

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

     └────────── MultiheadAttention(4 heads) ─────────┘
                         │
                         ▼
              Linear(256→1) → scalar per slot
                         │
                         ▼
              action_logits [B, 13]
         (masked_fill with -1e9 for illegal)
```

Two attention regularisation terms are computed on the cross-attention weights `attn_w [B,H,13,13]`:
- **P14 — per-query entropy**: `-(attn_w × log attn_w).sum(dim=-1).mean()` — maximised to spread attention mass across keys.
- **P18 — von Neumann rank**: SVD of the attention matrix; `-(p × log p).sum()` on normalised singular values — maximised to keep different queries attending to different keys.

## Prediction Heads

```
Opponent tokens [B, 6, D_MODEL]  (post_tokens[:, 6:12, :])
       │
       ├──▶ item_head    Linear(256 → 250)   → item_logits    [B, 6, 250]   CE loss
       ├──▶ ability_head Linear(256 → 311)   → ability_logits [B, 6, 311]   CE loss
       ├──▶ tera_head    Linear(256 → 19)    → tera_logits    [B, 6, 19]    CE loss
       ├──▶ move_head    Linear(256 → 686)   → move_logits    [B, 6, 686]   BCE multi-label
       └──▶ stats_head   Linear(256 → 6)     → stats_logits   [B, 6, 6]     MSE × 0.001
                                               (HP + atk, def, spa, spd, spe)
```

All heads are single Linear layers — the Transformer tokens already encode rich context.
Each head is masked to revealed slots only; unrevealed slots contribute zero gradient.
Move head uses BCE (multi-label) rather than CE: a Pokémon can use any of its 4 known moves,
and the label is a binary vector with 1s at each known move. Top-4 recall is reported at eval.
Stats head loss is scaled by `c_stats=0.001` to prevent raw stat values (~1-700) from dominating.

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
Reveal state [6 opponent slots]:
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
Masked PokemonBatch (zeros for unrevealed indices + scalars)
```

Applied twice: during rollout (decision-making) and again during training (data augmentation).
At mask_ratio=0.0 (Phase I), all opponent info is visible — "cheater" mode for bootstrapping.
At mask_ratio=1.0 (Phase III), only revealed info is visible — realistic POMDP setting.

## PPO Training Loop

```
train() in self_play.py

for update in 1..total_updates:

  ┌─ 1a. Opponent Selection ─────────────────────────────────┐
  │   10% RandomPolicy                                        │
  │   10% FullOffensePolicy                                   │
  │   60% EMAOpponent  (polyak-averaged online weights)       │
  │   20% pool.sample()  (periodic snapshots, or EMA if empty)│
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 1b. Rollout Collection (rollout.py) ─────────────────────┐
  │   collect_rollout(agent_self, agent_opp, n_envs=32)       │
  │                                                            │
  │   For each env in parallel:                                │
  │     encode_state() → push to BattleWindow (K=4)           │
  │     _build_agent_inputs() → model inputs                   │
  │     POMDP masking (apply_reveal_mask, mask_ratio)          │
  │     agent.forward() → action sample                        │
  │     action_to_choice() → PS protocol string                │
  │     PyBattle.make_choices()                                │
  │     compute_step_reward() → dense + terminal rewards       │
  │     RevealedTracker.update() from new log entries          │
  │     → Transition stored in RolloutBuffer                   │
  │                                                            │
  │   RolloutBuffer.compute_gae(γ=0.99, λ=0.95)               │
  │     → returns pre-normalised globally (z-score per rollout)│
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 2. PPO Updates (n_epochs=4 × minibatches) ──────────────┐
  │   For each minibatch:                                     │
  │     Re-apply POMDP mask (training augmentation)           │
  │     agent.forward(masked_poke, ...)                        │
  │     PredictionHeads.compute_loss() → pred_loss            │
  │     compute_losses() → policy + value + entropy + pred    │
  │     attn reg: - c_attn_entropy × attn_entropy             │
  │               + c_attn_rank × (ln(13) − attn_rank)        │
  │     total.backward()                                       │
  │     clip_grad_norm_(0.5)                                   │
  │     optimizer.step()                                       │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 2b. EMA weight update ───────────────────────────────────┐
  │   ema_params = 0.995 × ema_params + 0.005 × online_params │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 3. LR Schedule ──────────────────────────────────────────┐
  │   Warmup (linear 0→lr over 20 steps)                      │
  │   Cosine decay (lr → lr_min=1e-5)                         │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 4. Logging (stdout + metrics.csv) ───────────────────────┐
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 5. Evaluation (every eval_freq updates) ─────────────────┐
  │   100 games vs Random, FullOffense, EMA, Pool             │
  │   Periodic pool snapshot every pool_snapshot_freq=100      │
  │   (no win-rate gate — periodic regardless of performance)  │
  │   Save diagnostic plots + eval.csv + attention maps        │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 6. Checkpoint (every checkpoint_freq updates) ───────────┐
  │   agent_NNNNNN.pt (model + optimizer + config + update)    │
  └───────────────────────────────────────────────────────────┘
```

## Token Embedding Detail (439-dim)

```
┌──────────┬──────┬──────────────────────────────────────┐
│ Field    │ Dims │ Source                                │
├──────────┼──────┼──────────────────────────────────────┤
│ Species  │ 32   │ Embedding(1381, 32), seeded from BST  │
│ Type 1   │ 8    │ Embedding(19, 8), seeded from log2    │
│ Type 2   │ 8    │ (shared weight with Type 1)           │
│ Tera     │ 8    │ (shared weight with Type 1)           │
│ Item     │ 16   │ Embedding(250, 16)                    │
│ Ability  │ 16   │ Embedding(311, 16)                    │
│ Move 1   │ 32   │ Embedding(686, 32), type+cat prior    │
│ Move 2   │ 32   │ (shared weight with Move 1)           │
│ Move 3   │ 32   │ (shared weight with Move 1)           │
│ Move 4   │ 32   │ (shared weight with Move 1)           │
│ Scalars  │ 223  │ Raw float32 values (no embedding)     │
├──────────┼──────┼──────────────────────────────────────┤
│ TOTAL    │ 439  │                                       │
└──────────┴──────┴──────────────────────────────────────┘

Scalars breakdown (223 values):
  level(1) + hp_ratio(1) + hp_raw(1) + terastallized(1)
  + base_stats(5: atk,def,spa,spd,spe)
  + stats(5: atk,def,spa,spd,spe)
  + is_predicted(1)
  + boosts(7: atk,def,spa,spd,spe,accuracy,evasion)
  + move_pp(4) + move_disabled(4)
  + status(7: one-hot — none/brn/frz/par/psn/tox/slp)
  + volatiles(181)
  + is_active(1) + fainted(1) + trapped(1) + force_switch(1) + revealed(1)
  = 223
```

## Field Token (72-dim)

```
┌────────────────┬──────┬──────────────────────────────────┐
│ Field          │ Dims │ Description                      │
├────────────────┼──────┼──────────────────────────────────┤
│ Weather        │ 9    │ One-hot (none + 8 weathers)      │
│ Terrain        │ 5    │ One-hot (none + 4 terrains)      │
│ Pseudo-weather │ 8    │ Multi-hot (gravity, trickroom…)  │
│ Side 0 conds   │ 23   │ Side conditions (scaled layers)  │
│ Side 0 misc    │ 2    │ pokemon_left/6, total_fainted/6  │
│ Side 1 conds   │ 23   │ (same structure)                 │
│ Side 1 misc    │ 2    │ (same structure)                 │
├────────────────┼──────┼──────────────────────────────────┤
│ TOTAL          │ 72   │                                  │
└────────────────┴──────┴──────────────────────────────────┘
```

## Model Size Breakdown

| Module | Parameters |
|--------|-----------|
| PokemonEmbeddings (species + type + item + ability + move tables) | ~75K |
| BattleBackbone — input projections (pokemon_proj + field_proj) | ~132K |
| BattleBackbone — positional embeddings (temporal + slot) | ~4K |
| BattleBackbone — TransformerEncoder (3 layers × 4 heads, d=256, ffn=512) | ~1.58M |
| BattleBackbone — value head (Linear-ReLU-Linear) | ~66K |
| BattleBackbone — action cross-attention + score | ~263K |
| ActionEncoder (move_proj + mechanic_proj + switch_proj) | ~79K |
| PredictionHeads (item + ability + tera + move + stats heads) | ~327K |
| **Total** | **~2.53M** |

## Key Design Decisions

1. **Single Transformer pass**: `encode()` runs the Transformer once; `act()` only does lightweight cross-attention over already-computed tokens. This keeps inference fast across 32 parallel envs.

2. **Pre- vs post-Transformer tokens split**: Action queries use pre-Transformer tokens (raw projected features + positional embeddings, before self-attention). This avoids the active Pokémon self-matching itself in cross-attention (P13e). The backbone keys/values and prediction heads use post-Transformer (enriched) tokens.

3. **Mean-pool for value**: Simple averaging over 13 current-turn post-Transformer tokens. Order-invariant, costs zero parameters, avoids the 3328→256 flattening bottleneck.

4. **Shared embedding weights**: `ActionEncoder` shares `move_embed` and `type_embed` with `PokemonEmbeddings`, ensuring consistent representations and fewer parameters.

5. **Pre-LN Transformer**: Layer norm before each sub-layer (not after), standard in modern architectures for training stability.

6. **Linear prediction heads only**: No MLPs — the Transformer tokens encode rich enough context. MLPs would overfit on the sparse reveal signal and add unnecessary parameters.

7. **Global return normalisation**: Returns are z-scored once per rollout inside `compute_gae()` before minibatch splits. This gives the value head a stable target across all minibatches in a PPO update — per-minibatch normalisation would create a moving target that prevents the critic from learning.

8. **BCE multi-label for move head**: Move prediction uses binary cross-entropy over all 686 moves rather than 4 independent cross-entropies. This naturally handles the unordered set of a Pokémon's moves and avoids the positional ambiguity of slot-wise CE.

## Patch History

Chronological log of significant architectural and training changes. Details and motivation in `TODO.md`.

| Patch | Area | Change |
|-------|------|--------|
| P10a | Training | POMDP masking symétrised — `agent_opp` now also receives masked observations (same `RevealedTracker` info from its side). Fixes the systematic information asymmetry that caused win rates to collapse. |
| P10b | Training | EMA Opponent added — polyak-averaged copy of online weights (decay=0.995) replaces pool as the primary opponent. Always available, no snapshot threshold. |
| P10c | Training | Residual pool kept for diversity — periodic snapshots every 100 updates regardless of win rate. Size capped at 10. WR-gated snapshot system removed. |
| P11a | Training | Advantage normalisation per-batch (z-score) in `compute_losses()`. |
| P11b | Training | `c_value` 1.0 → 2.0 for stronger critic regularisation. |
| P11c | Training | `min_steps` 1024 → 2048 for denser rollouts (~3-4 battles per env per update). |
| P13b | Backbone | Padding mask added — zero-filled turns (early game, K=4 window not full) are excluded from self-attention via `src_key_padding_mask`. |
| P13c | Reward | `HP_ADV_SCALE` ×5 + `COUNT_ADV_SCALE=0.03` added — improves signal density between sparse terminal rewards. |
| P13d | Prediction | Move head factorised with 128-dim bottleneck (705K → 387K params). Later superseded by P15c move head redesign. |
| P13e | ActionEncoder | `active_token` removed from move query input — prevents self-match shortcut where move queries trivially attend to the active Pokémon token. |
| P14 | Backbone | Cross-attention entropy regularisation — `c_attn_entropy × H(attn_w)` added to loss to prevent all 13 action queries from collapsing to the FIELD token. Dropout 0.1 → 0.15. |
| P15b | Training | Prediction head masks corrected — auxiliary loss masks now use simulator ground truth (`species_idx != UNK`) instead of `RevealedTracker` booleans. Fixes `item_acc=0%` pathology. |
| P15c | Prediction | Stats head added (`Linear(256→6)`, MSE ×0.001) predicting opponent HP + atk/def/spa/spd/spe. Move head redesigned as single BCE multi-label head (686 outputs) replacing slot-wise CE. |
| P16a | Training | `c_entropy` reduced to prevent entropy bonus from dominating the policy loss. |
| P16b | Training | `n_epochs` 2 → 4 — clip_frac was very low, model can support more update passes per rollout. |
| P16d | Training | Advantage diagnostics added to logs: `adv_mean`, `adv_std`, `ratio_dev`. |
| P17 | Reward | Dense rewards scaled ×10 (KO ±0.5, HP ±0.5, count ±0.3). Status inflicted (+0.1), hazard set (+0.1/layer), hazard removed (+0.1/layer) added. |
| P17b | Training | `c_stats=0.001` added — scales stats MSE loss to prevent raw stat values (~1-700) from dominating the auxiliary loss. |
| P18 | Backbone | Cross-attention rank regularisation — von Neumann entropy of attention singular values added to loss to penalise low-rank attention (all queries attending to same pattern). |
| P22 | ActionEncoder | `active_token` removal confirmed stable (was P13e); move_proj input is now `D_MOVE+2=34` dims only. |
| —   | Training | **Return normalisation moved to `compute_gae()`** — z-score computed once per rollout before minibatch splits, replacing per-minibatch normalisation in `compute_losses()`. Fixes moving-target pathology that prevented critic from learning (EV stuck at ~0.05). |
