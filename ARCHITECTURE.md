# CynthAI_v2 — Architecture Document

## High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          CynthAIAgent (agent.py)                          │
│                                                                           │
│   State Dict ──▶ PokemonEmbeddings ──▶ BattleBackbone ──┬──▶ Logits      │
│   (PyBattle)      (embeddings.py)      (backbone.py)    │                │
│                    + ScalarRunningNorm                   ├──▶ Preds       │
│                                                          │                │
│                    ActionEncoder ◀── pre_tokens ─────────┘                │
│                    (action_space.py)  (pre-Transformer)                   │
│                                                                           │
│                    PredictionHeads ◀── post_tokens                        │
│                    (prediction_heads.py)  (post-Transformer)              │
│                                                                           │
│   IndependentCritic ◀── pokemon_tokens, field_tensor, action_embeds      │
│   (critic.py)           Own Transformer + CLS token + value head         │
│                         + victory head + action-aware cross-attention     │
└───────────────────────────────────────────────────────────────────────────┘
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
└──────────────┘         │              │           │                      │
                         │ scalars(223) │           │ CLS token prepended  │
                         │  ↑           │           │  [B, 1+52, D_MODEL]  │
                         │ ScalarRun-   │           │                      │
                         │ ningNorm     │           │ TransformerEncoder   │
                         │ (EMA norm,   │           │  3 layers × 4 heads  │
                         │  per-feature)│           │  Pre-LN, d_model=256 │
                         │ = 439 dims   │           │  FFN=512, dropout=0.15│
                         └──────────────┘           │                      │
                                                    │ Output:              │
                                                    │  pre_tokens  [B,13,256]  ← before Transformer
                                                    │  post_tokens [B,13,256]  ← after Transformer
                                                    │  cls_token   [B, 256]    ← aggregation token
                                                    └──────┬───────────────┘
                                                           │
       ┌──────────────────────┌────────────────────────────┤
       ▼                      ▼                            ▼
Step 4a: Actor Head     Step 4b: IndependentCritic   Step 4c: Prediction Heads
┌─────────────────────┐ ┌────────────────────────┐   ┌──────────────────────┐
│ ActionEncoder       │ │ IndependentCritic      │   │ PredictionHeads      │
│  (pre_tokens)       │ │ (critic.py)            │   │                      │
│  + action_cross_attn│ │                        │   │ post_tokens[:,6:12,:]│
│  + action_score     │ │ Own CLS + Transformer  │   │        ↓             │
│                     │ │ + victory_head (BCE)   │   │ item_head    (Linear)│
│ Query: action_embeds│ │ + action cross-attn    │   │ ability_head (Linear)│
│   [B, 13, 256]      │ │ + tanh value_bound     │   │ tera_head    (Linear)│
│ Key/Value:post_toks │ │        ↓               │   │ move_head    (Linear)│
│   [B, 13, 256]      │ │ V(s,a) [B, 1]         │   │ stats_head   (Linear)│
│        ↓            │ │ win_logit [B, 1]       │   │        ↓             │
│ MHA(4 heads)        │ └────────────────────────┘   │ item [B,6,250]       │
│        ↓            │                               │ abil [B,6,311]       │
│ Linear → scalar     │                               │ tera [B,6,19]        │
│        ↓            │                               │ moves[B,6,686] (BCE) │
│ logits [B, 13]      │                               │ stats[B,6,6]  (MSE)  │
│ masked_fill(-1e9)   │                               └──────────────────────┘
└─────────────────────┘
```

## Token Sequence Layout

```
CLS + K=4 turns × 13 tokens per turn = 1 + 52 = 53 tokens

[CLS]  Turn 0 (oldest)              Turn K-1 (current)
       ├──────────┼──────────┤     ├──────────┼──────────┤
       │ OWN 0-5  │ OPP 6-11 │ ... │ OWN 0-5  │ OPP 6-11 │
       │          │          │     │          │          │
       │ [active, │ [active, │     │ [active, │ [active, │
       │  bench×5]│  bench×5]│     │  bench×5]│  bench×5]│
       └──────────┴──────────┘     └──────────┴──────────┘
           12 pokemon tokens     +      1 field token     = 13 per turn

CLS token: learned parameter, prepended to the sequence, used by critic for
           aggregation and value estimation (cls_backbone_grad controls whether
           its gradient flows through the backbone Transformer or is detached).

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
to prevent the active Pokemon token from dominating queries via self-match shortcuts (P13e).

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
- **P14 — per-query entropy**: `-(attn_w * log attn_w).sum(dim=-1).mean()` — maximised to spread attention mass across keys.
- **P18 — von Neumann rank**: SVD of the attention matrix; `-(p * log p).sum()` on normalised singular values — maximised to keep different queries attending to different keys.

## Independent Critic (critic.py)

```
IndependentCritic
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Inputs: pokemon_tokens [B, K*12, TOKEN_DIM]                         │
│          field_tensor   [B, K, FIELD_DIM]                            │
│          action_embeds  [B, 13, D_MODEL]  (from ActionEncoder)       │
│          action_mask    [B, 13]                                       │
│                                                                      │
│  1. Project pokemon + field → [B, K*13, D_MODEL]                     │
│  2. Add temporal_emb + slot_emb                                      │
│  3. Prepend CLS token → [B, 1+52, D_MODEL]                          │
│  4. TransformerEncoder (2 layers, 4 heads, d=256)                    │
│  5. Extract CLS output → [B, D_MODEL]                               │
│                                                                      │
│  Optional action-aware cross-attention (critic_action_aware=True):   │
│  ┌───────────────────────────────────────────────────────────┐       │
│  │ CLS as query, action_embeds as key/value                  │       │
│  │ n_cross_layers=1 (MHA + LayerNorm)                        │       │
│  │ action_mask applied (illegal actions excluded)             │       │
│  │ → CLS enriched with action context → V(s,a) not just V(s) │       │
│  │ Diagnostics: critic_action_attn_entropy, attn_max          │       │
│  └───────────────────────────────────────────────────────────┘       │
│                                                                      │
│  Value head: Linear(D_MODEL→D_MODEL) → ReLU → Linear(D_MODEL→1)    │
│  value_bound: tanh(output) × bound  (default bound=10.0)            │
│  → V(s,a) [B, 1]                                                    │
│                                                                      │
│  Victory head (use_victory_head=True):                               │
│  Linear(D_MODEL→1) → win_logit [B, 1]                               │
│  BCE loss on episode outcome (1=win, 0=loss)                         │
│  Bootstraps CLS to encode win-relevant information                   │
│                                                                      │
│  Own optimizer: AdamW(lr=5e-4, wd=1e-4)                              │
│  Own grad clip: 1.0 (vs 0.5 for actor)                               │
│  Detached forward: no gradient flows back to backbone                │
└──────────────────────────────────────────────────────────────────────┘
```

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
Move head uses BCE (multi-label) with pos_weight correction for class imbalance (4/686).
Stats head loss scaled by `c_stats=0.001`.

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
At mask_ratio=0.0 ("cheater" mode), all opponent info is visible — used for bootstrapping.
At mask_ratio=1.0 (Phase 2), only revealed info is visible — realistic POMDP setting.

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
  │   collect_rollout(agent_self, agent_opp, n_envs=64)       │
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
  │   For each minibatch (batch_size=256):                     │
  │     Re-apply POMDP mask (training augmentation)           │
  │     agent.forward(masked_poke, ...)                        │
  │     critic.forward(poke_tokens, field, action_embeds, ...) │
  │     PredictionHeads.compute_loss() → pred_loss            │
  │     compute_losses() → policy + value + entropy + pred    │
  │     Victory head loss (BCE on win label, c_victory=0.1)    │
  │     Attn reg: - c_attn_entropy × attn_entropy             │
  │               + c_attn_rank × (ln(13) − attn_rank)        │
  │     total.backward()                                       │
  │     clip_grad_norm_(actor=0.5, critic=1.0)                 │
  │     actor_optimizer.step()                                 │
  │     critic_optimizer.step()                                │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 2b. EMA weight update ───────────────────────────────────┐
  │   ema_params = 0.995 × ema_params + 0.005 × online_params │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 3. LR Schedule ──────────────────────────────────────────┐
  │   Warmup (linear 0→lr over 20 steps)                      │
  │   Cosine decay (lr → lr_min=1e-5)                         │
  │   Separate schedule for critic                             │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 4. Logging (stdout + wandb) ────────────────────────────┐
  │   Per-update metrics: policy_loss, value_loss, entropy,   │
  │     clip_frac, explained_variance, lr, attn_entropy,      │
  │     attn_rank, critic_action_attn_entropy/max, grad_norms │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 5. Evaluation (every eval_freq updates) ─────────────────┐
  │   500 games vs Random, FullOffense, EMA, Pool              │
  │   Win rates + Wilson 95% CI                                │
  │   Plots: action dists, battle length, reward decomposition,│
  │     value calibration, cross-attention heatmaps,           │
  │     cosine similarity matrices                             │
  │   Save: eval.csv, eval_data/*.json, plots/                 │
  │   Attention maps: Transformer per-head + per-layer grids   │
  │   Wandb: all metrics + glob for PNG plots                  │
  │                                                            │
  │   ┌─ Probing (every probe_freq evals) ──────────────────┐ │
  │   │ Fresh rollout (min_steps=2048) for token caching     │ │
  │   │ 4 probe suites (see Probing System below)            │ │
  │   │ Results logged as probe/* metrics to wandb           │ │
  │   └─────────────────────────────────────────────────────┘ │
  └───────────────────────────────────────────────────────────┘
       │
  ┌─ 6. Checkpoint (every checkpoint_freq updates) ───────────┐
  │   agent_NNNNNN.pt (model + optimizer(s) + config + update) │
  │   Periodic pool snapshot every pool_snapshot_freq updates   │
  └───────────────────────────────────────────────────────────┘
```

## Probing System (diag/probing/)

During evaluation, a probing suite runs periodically (`probe_freq` evals) to diagnose learned representations. A fresh rollout is collected, tokens are cached at all model levels, and 4 probe modules run independently.

### Token Caching (`cache_tokens_full`)

Extracts from model forward passes:
- **Actor tokens**: token embeddings (pre-projection), current state tokens (post-Transformer), full 52-token sequence
- **DETR queries**: 13 action embeddings after cross-attention
- **Critic tokens**: CLS token, full 53-token sequence
- **Labels**: return, win, per-pokemon type/item/ability/HP/stats, next-turn HP

### Probe Modules

```
┌─ Actor Probes (actor_probes.py) ──────────────────────────────────────────┐
│ Token-level linear probes on post-Transformer representations:            │
│   - Return R²: per-token regression predicting episode return             │
│   - Win AUC: per-token logistic regression predicting win/loss            │
│   - Type/Item/Ability accuracy: per-token classification (12 pokemon)     │
│   - HP R²: per-token regression on HP ratio                               │
│   - Base stats R²: per-token regression on 5 base stats                   │
│   - Mean-pool probes: same metrics on mean-pooled 12 tokens               │
│   - Cross-token 12×12 matrix: cosine similarity between pokemon tokens    │
│   - CLS analysis: backbone CLS vs mean-pool comparison                    │
│   - PCA scatter: colored by return, win, HP, etc.                         │
└───────────────────────────────────────────────────────────────────────────┘

┌─ Critic Probes (critic_probes.py) ────────────────────────────────────────┐
│ CLS token analysis from independent critic:                               │
│   - Value prediction R² and Pearson correlation                           │
│   - Win classification AUC                                                │
│   - PCA scatter: CLS colored by value and win label                       │
│   - Effective rank: SVD on CLS activations (representation diversity)     │
│   - Per-dimension correlation: which CLS dims track value/win             │
│   - 52-token probes: same analyses on full critic sequence                │
└───────────────────────────────────────────────────────────────────────────┘

┌─ DETR Probes (detr_probes.py) ────────────────────────────────────────────┐
│ Action query analysis (after cross-attention):                            │
│   - Action chosen AUC: per-query logistic regression on chosen/not-chosen │
│   - Mean-pool 13-class accuracy: classify which action was chosen         │
│   - Win probability: predict win from DETR queries                        │
│   - Delta HP (active slot): predict HP change for next turn               │
│   - KO prediction: predict whether active pokemon gets KO'd next turn     │
│   All filtered to no-switch steps only (action < 8)                       │
└───────────────────────────────────────────────────────────────────────────┘

┌─ SVD Probes (svd_probes.py) ──────────────────────────────────────────────┐
│ Representation geometry analysis at 3 levels:                             │
│   - token_emb: raw embeddings (before Transformer)                        │
│   - state_cur: current-turn 13 tokens (post-Transformer)                  │
│   - state_all: full 52-token sequence                                     │
│ For each level:                                                           │
│   - TruncatedSVD PCA (2D visualization)                                   │
│   - Energy spectrum: cumulative variance explained by top components      │
│   - Effective rank: intrinsic dimensionality of representations           │
│   - Per-state SVD: singular value distribution                            │
└───────────────────────────────────────────────────────────────────────────┘
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
│ Scalars  │ 223  │ Raw floats, normalised by             │
│          │      │ ScalarRunningNorm (EMA per-feature)   │
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

ScalarRunningNorm (model/embeddings.py):
  EMA update during training:  mean ← lerp(mean, batch_mean, 0.05)
                                var  ← lerp(var,  batch_var,  0.05)
  Frozen at eval (model.eval()): uses accumulated stats read-only.
  Buffers saved in checkpoint → no warm-up needed on resume.
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

### Actor (backbone + heads)

| Module | Parameters |
|--------|-----------|
| PokemonEmbeddings (species + type + item + ability + move tables) | ~75K |
| BattleBackbone — input projections (pokemon_proj + field_proj) | ~132K |
| BattleBackbone — positional embeddings (temporal + slot) + CLS token | ~4K |
| BattleBackbone — TransformerEncoder (3 layers × 4 heads, d=256, ffn=512) | ~1.58M |
| BattleBackbone — action cross-attention + score | ~263K |
| ActionEncoder (move_proj + mechanic_proj + switch_proj) | ~79K |
| PredictionHeads (item + ability + tera + move + stats heads) | ~327K |
| **Actor total** | **~2.46M** |

### IndependentCritic (critic_n_layers=2, action_aware=True)

| Module | Parameters |
|--------|-----------|
| IndependentCritic — input projections + positional emb + CLS | ~136K |
| IndependentCritic — TransformerEncoder (2 layers) | ~1.05M |
| IndependentCritic — action cross-attention (1 layer) | ~263K |
| IndependentCritic — value head + victory head | ~66K |
| **Critic total** | **~1.52M** |

| **Grand total** | **~3.98M** |

## Reward Design

| Component | Scale | Description |
|-----------|-------|-------------|
| Terminal win/loss | ±1.0 | Sparse, never scaled |
| Opponent KO | +0.5 | Scaled by `dense_scale` |
| Own KO | -0.5 | Scaled by `dense_scale` |
| Delta HP advantage | 0.5 × delta | Normalized HP difference ratio |
| Delta count advantage | 0.3 × delta/6 | Alive Pokemon difference |
| Status inflicted | +0.1 | Opponent gains a status condition |
| Hazard set | +0.1 × layers | Entry hazard placed on opponent side |
| Hazard removed | +0.1 × layers | Rapid Spin / Defog clears our side |

## Eval Diagnostics & Visualization

During evaluation (every `eval_freq` updates), the following are generated:

| Diagnostic | Output | Description |
|------------|--------|-------------|
| Win rates | eval.csv, wandb | Per-opponent win rate + Wilson 95% CI |
| Action distributions | plots/actions/ | Histograms of action choices per opponent |
| Battle length | plots/battle_len/ | Distribution of episode lengths |
| Reward decomposition | plots/reward/ | Stacked bar chart of reward components |
| Value calibration | plots/value_calib/ | Predicted V(s) vs actual return scatter |
| Cross-attention | plots/cross_attn/ | DETR attention heatmaps over keys |
| Cosine similarity | plots/cos_sim/ | 5 matrices: A_AT, B_BT, C_CT, B_CT, A_CT |
| Attention maps | plots/attn_maps/ | Transformer per-head per-layer grids |
| Probing (periodic) | wandb probe/* | Actor/critic/DETR/SVD probes (see above) |

Cosine similarity matrices measure representation alignment:
- **A**: current own tokens, **B**: current opponent tokens, **C**: DETR action queries
- Self-similarity (A_AT, B_BT, C_CT) and cross-similarity (B_CT, A_CT)

## Key Design Decisions

1. **Single Transformer pass**: `encode()` runs the Transformer once; `act()` only does lightweight cross-attention over already-computed tokens. This keeps inference fast across 64 parallel envs.

2. **Pre- vs post-Transformer tokens split**: Action queries use pre-Transformer tokens (raw projected features + positional embeddings, before self-attention). This avoids the active Pokemon self-matching itself in cross-attention (P13e). The backbone keys/values and prediction heads use post-Transformer (enriched) tokens.

3. **CLS token**: A learned aggregation token prepended to the sequence. Used by the critic for global state representation. `cls_backbone_grad` controls whether gradient from critic flows through backbone.

4. **Independent Critic**: Separate Transformer for V(s,a), with its own learning rate (5e-4 vs 2.5e-4 for actor) and grad clip (1.0 vs 0.5). Decoupling eliminates the gradient tension between actor and critic when they share a backbone. Critic can train faster without destabilising the actor's representations.

5. **Action-aware critic**: CLS token cross-attends to action embeddings (`n_cross_layers=1`), computing V(s,a) instead of V(s). This gives the critic richer information about the chosen action for more accurate value estimation.

6. **Victory head**: Auxiliary BCE head on critic CLS predicting episode outcome (win/loss). Bootstraps the CLS representation to encode win-relevant information early in training.

7. **Value bound (tanh squashing)**: Critic output is `tanh(raw) × bound` (default 10.0), preventing unbounded value estimates that can destabilize training.

8. **ScalarRunningNorm**: Per-feature EMA normalisation of the 223 raw scalars before token projection. Avoids arbitrary hard-coded constants and adapts to the actual distribution of in-battle values. Frozen at eval, saved in checkpoints.

9. **Shared embedding weights**: `ActionEncoder` shares `move_embed` and `type_embed` with `PokemonEmbeddings`, ensuring consistent representations and fewer parameters.

10. **Pre-LN Transformer**: Layer norm before each sub-layer (not after), standard in modern architectures for training stability.

11. **Linear prediction heads only**: No MLPs — the Transformer tokens encode rich enough context. MLPs would overfit on the sparse reveal signal and add unnecessary parameters.

12. **Global return normalisation**: Returns are z-scored once per rollout inside `compute_gae()` before minibatch splits. This gives the value head a stable target across all minibatches in a PPO update — per-minibatch normalisation would create a moving target.

13. **BCE multi-label for move head**: Move prediction uses binary cross-entropy over all 686 moves (with pos_weight) rather than 4 independent cross-entropies. This naturally handles the unordered set of a Pokemon's moves and avoids the positional ambiguity of slot-wise CE.

14. **Probing system**: Periodic linear probes on frozen representations diagnose what each model component has learned, without affecting training.

## Patch History

Chronological log of significant architectural and training changes. Details and motivation in `TODO.md`.

| Patch | Area | Change |
|-------|------|--------|
| P10a | Training | POMDP masking symmetrised — `agent_opp` now also receives masked observations. |
| P10b | Training | EMA Opponent added — polyak-averaged copy (decay=0.995). Always available. |
| P10c | Training | Residual pool kept for diversity — periodic snapshots every 100 updates. |
| P11a | Training | Advantage normalisation per-batch (z-score) in `compute_losses()`. |
| P11b | Training | `c_value` 1.0 → 2.0. |
| P11c | Training | `min_steps` 1024 → 2048 → 4096. |
| P13b | Backbone | Padding mask added — zero-filled turns excluded from self-attention. |
| P13c | Reward | HP advantage and count advantage rewards added. |
| P13e | ActionEncoder | `active_token` removed from move query input — prevents self-match shortcut. |
| P14 | Backbone | Cross-attention entropy regularisation. Dropout 0.1 → 0.15. |
| P15b | Training | Prediction head masks corrected — auxiliary loss uses simulator ground truth. |
| P15c | Prediction | Stats head added. Move head redesigned as BCE multi-label. |
| P16b | Training | `n_epochs` 2 → 4. |
| P17 | Reward | Dense rewards rescaled (KO ×10, HP ×50). Status/hazard rewards added. |
| P18 | Backbone | Cross-attention rank regularisation (von Neumann entropy on SVD). |
| P22 | ActionEncoder | `active_token` removal confirmed. move_proj input is `D_MOVE+2=34` dims. |
| — | Training | Return normalisation moved to `compute_gae()`. |
| — | Backbone | CLS token added for aggregation. Value head moved to critic. |
| — | Model | **IndependentCritic** with own Transformer, own optimizer, detached forward. |
| — | Critic | **Action-aware cross-attention**: CLS queries action embeddings → V(s,a). |
| — | Critic | **Victory head**: BCE on win/loss, bootstraps CLS representation. |
| — | Critic | **Value bound**: tanh squashing (default ±10.0) for stability. |
| — | Embeddings | **ScalarRunningNorm** — EMA per-feature normalisation of 223 raw scalars. |
| — | Eval | **Probing system**: 4 probe modules (actor/critic/DETR/SVD) run periodically. |
| — | Eval | **Cosine similarity matrices**: 5 representation alignment diagnostics. |
| — | Logging | **Wandb integration**: all metrics, plots, probing results logged to wandb. |
