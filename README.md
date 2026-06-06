# CynthAI_v2 — Pokemon AI with PPO Self-Play and Transformer Backbone

CynthAI_v2 is a deep reinforcement learning agent that plays Pokemon Showdown Gen 9 Random Battles. It uses a **Transformer backbone** (DETR-style cross-attention actor), **PPO self-play** with an EMA + pool opponent system, an **independent action-aware critic** with victory head, and a **Rust simulator** (PyO3 binding over `pokemon-showdown-rs`) for battle simulation.

## Project Structure

```
CynthAI_v2/
├── run_cheater_v10.py              # Latest launcher — cheater mode + independent critic
├── run_cheater_indep_critic.py     # Launcher — cheater mode + independent critic (base)
├── TODO.md                         # Known issues, roadmap, and proposed fixes
├── requirements.txt                # Python dependencies
├── simulator.pyd                   # Compiled Rust simulator (PyO3 wheel)
│
├── model/                          # Neural network modules
│   ├── agent.py                    # CynthAIAgent — top-level nn.Module
│   ├── backbone.py                 # BattleBackbone — Transformer + CLS token + actor head
│   ├── critic.py                   # IndependentCritic — separate Transformer, action-aware, victory head
│   ├── embeddings.py               # PokemonEmbeddings — token encoding + ScalarRunningNorm
│   └── prediction_heads.py         # PredictionHeads — auxiliary supervised heads (item/ability/tera/moves/stats)
│
├── env/                            # Environment interface
│   ├── state_encoder.py            # State → feature vectors (PokemonFeatures, FieldFeatures)
│   ├── action_space.py             # ActionEncoder — 13 action embeddings (DETR queries)
│   ├── revealed_tracker.py         # POMDP tracking — parses battle logs for opponent info
│   └── bots.py                     # Rule-based opponents (RandomPolicy, FullOffensePolicy)
│
├── training/                       # Training loop and utilities
│   ├── self_play.py                # PPO training loop, opponent pool, EMA, curriculum, wandb
│   ├── rollout.py                  # Episode collection, GAE, BattleWindow, reward computation
│   ├── losses.py                   # PPO-clip loss + entropy + prediction + attention reg losses
│   ├── evaluate.py                 # Evaluation vs fixed opponents, cosine similarity diagnostics
│   ├── probing_eval.py             # Probing orchestration — runs 4 probe suites on fresh rollouts
│   ├── monitor.py                  # Eval plots (action dists, battle length, value calib, cos sim)
│   ├── attention_viz.py            # Transformer + cross-attention weight visualization
│   ├── visualize.py                # Training curve visualization
│   └── live_viz.py                 # Real-time battle visualization (Pygame)
│
├── diag/probing/                   # Representation probing system
│   ├── __main__.py                 # CLI entry point for standalone probing
│   ├── _common.py                  # Token caching, label extraction, shared utilities
│   ├── actor_probes.py             # Per-token return/win/type/item/HP probes on backbone
│   ├── critic_probes.py            # CLS analysis, effective rank, PCA, per-dim correlation
│   ├── detr_probes.py              # Action query probes: chosen action, win, delta HP, KO
│   └── svd_probes.py               # SVD/PCA at 3 levels, energy spectrum, effective rank
│
├── data/dicts/                     # Vocabulary index files (JSON)
│   ├── species_index.json          # 1381 Pokemon species
│   ├── move_index.json             # 686 moves
│   ├── item_index.json             # 250 items
│   ├── ability_index.json          # 311 abilities
│   ├── type_index.json             # 19 types (incl. ???)
│   ├── type_embeddings.json        # Pre-computed type embeddings (log2 type chart, 8-dim)
│   └── move_embeddings.json        # Pre-computed move embeddings (type+category, 32-dim)
│
├── scripts/                        # Diagnostic and analysis scripts
│   ├── pool_check.py               # Inspect opponent pool snapshots
│   ├── read_attn_maps.py           # Load and visualise saved attention maps
│   └── viz_attention.py            # Standalone attention visualization
├── diag/                           # Additional diagnostics (cosine sim, offline critic, etc.)
├── debug/                          # Debug scripts (NaN diagnosis, battle tests, etc.)
├── experiments/                    # One-off experiments (EV estimation, critic blowup diag)
├── simulator/                      # Rust simulator source (PyO3)
│   └── src/lib.rs                  # PyBattle — PS protocol battle engine
├── tests/                          # Test suite
│   ├── run_all_tests.py            # Unified test runner
│   ├── test_full_pipeline.py       # Model pipeline tests
│   ├── test_simulator.py           # Simulator + encoder tests
│   ├── test_encoder.py             # Vocabulary and encoding tests
│   └── test_attention_maps.py      # Attention map capture tests
└── checkpoints/                    # Saved training runs
    └── <run_name>/                 # Per-run: agent_*.pt, metrics.csv, eval.csv, plots/
```

## Architecture Overview

**~2.46M parameters** (actor) / **~3.98M** with `IndependentCritic` (action-aware + victory head).

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture document with diagrams.

### Model Pipeline

```
State dict (PyBattle) → State Encoder → ScalarRunningNorm → Token Embedding (439-dim)
     → Transformer Backbone (3 layers, 4 heads, d=256, CLS token)
          ├→ Actor Head (DETR cross-attention) → 13 action logits
          ├→ IndependentCritic (own 2-layer Transformer + action cross-attention) → V(s,a) + win_logit
          └→ Prediction Heads → opponent item/ability/tera/moves/stats
```

1. **State Encoding** (`env/state_encoder.py`): Converts `PyBattle.get_state()` dicts into `PokemonFeatures` and `FieldFeatures`. Each Pokemon token encodes species (32), 3x types (8 each), item (16), ability (16), 4x moves (32 each), and 223 scalar features (HP, stats, boosts, volatiles, status).

2. **Token Embedding** (`model/embeddings.py`): `PokemonEmbeddings` looks up 7 embedding tables, applies `ScalarRunningNorm` (per-feature EMA normalisation) to the 223 raw scalars, then concatenates → 439-dim per Pokemon. Type embeddings seeded from log2 type chart; move embeddings from type+category priors.

3. **Transformer Backbone** (`model/backbone.py`): 3 layers, 4 heads, d_model=256, FFN=512, Pre-LN, dropout=0.15. Processes K=4 turns of 13 tokens each (6 own + 6 opponent + 1 field) = 52 tokens + CLS. Learned temporal + slot positional embeddings. Padding mask zeroes out empty turns.

4. **Actor Head (DETR-style)**: 13 action queries (built from pre-Transformer tokens) cross-attend over the 13 current-turn post-Transformer tokens. Output: 13 logits (4 moves + 4 mechanic variants + 5 switches), masked for illegal actions.

5. **Independent Critic** (`model/critic.py`): Separate 2-layer Transformer with its own CLS token. **Action-aware**: CLS cross-attends to action embeddings to compute V(s,a). **Victory head**: auxiliary BCE predicting episode outcome, bootstraps CLS to encode win-relevant features. Output bounded by tanh × 10.0. Own optimizer (lr=5e-4), own grad clip (1.0), detached from backbone.

6. **Prediction Heads** (`model/prediction_heads.py`): 5 independent Linear heads on opponent tokens predicting item (250 CE), ability (311 CE), tera type (19 CE), moves (686 BCE multi-label), and stats (6 MSE). Auxiliary supervised loss, masked to revealed slots.

### Action Space (13 slots)

| Slots | Description |
|-------|-------------|
| 0-3   | Regular moves (move 1-4) |
| 4-7   | Mechanic moves (same moves + Tera modifier) |
| 8-12  | Switch to bench Pokemon 0-4 |

### POMDP Masking

`RevealedTracker` parses PS protocol battle logs to track revealed opponent attributes. `apply_reveal_mask()` zeros unrevealed categoricals/scalars with configurable `mask_ratio`. At `mask_ratio=0.0` ("cheater" mode), all opponent info is visible — used in Phase 1 training.

## Training Setup

### Two-Phase Strategy

- **Phase 1 (current)**: Full-information PPO self-play (`mask_ratio=0.0`). Learn a high-quality critic that sees all information (perfect value function).
- **Phase 2 (planned)**: POMDP actor training. Actor sees masked observations, critic remains frozen from Phase 1 (asymmetric actor-critic).

### PPO Configuration

| Parameter | Value |
|-----------|-------|
| gamma (discount) | 0.99 |
| lambda (GAE) | 0.95 |
| Clip epsilon | 0.2 |
| Epochs per update | 4 |
| Batch size | 256 |
| Parallel envs | 64 |
| Min steps per rollout | 4096 |
| Actor optimizer | AdamW (lr=2.5e-4, wd=1e-4) |
| Actor LR schedule | Linear warmup (20 steps) + cosine decay to 1e-5 |
| Actor max grad norm | 0.5 |
| Critic optimizer | AdamW (lr=5e-4, wd=1e-4) |
| Critic max grad norm | 1.0 |

### Loss Function

```
total = policy_loss
      + c_value × value_loss
      + c_entropy × entropy_loss
      + c_pred × pred_loss
      + c_victory × victory_loss
      - c_attn_entropy × attn_entropy
      + c_attn_rank × (ln(13) - attn_rank)
```

| Component | Coefficient | Description |
|-----------|-------------|-------------|
| Policy | 1.0 | PPO-clip surrogate with per-batch advantage z-score |
| Value | c_value=2.0 | MSE against globally z-scored returns |
| Entropy | c_entropy=0.01 | Negative entropy over legal actions (exploration) |
| Prediction | c_pred=0.6 | Masked CE/BCE/MSE on opponent hidden state |
| Victory | c_victory=0.1 | BCE on episode outcome (critic victory head) |
| Attn entropy | c_attn_entropy=0.01 | Maximise per-query cross-attention entropy |
| Attn rank | c_attn_rank=0.005 | Maximise von Neumann entropy of attention SVs |

### Reward Design

| Component | Scale | Description |
|-----------|-------|-------------|
| Terminal win/loss | +/-1.0 | Sparse, never scaled |
| Opponent KO | +0.5 | Scaled by `dense_scale` |
| Own KO | -0.5 | Scaled by `dense_scale` |
| Delta HP advantage | 0.5 × delta | Normalized HP difference ratio |
| Delta count advantage | 0.3 × delta/6 | Alive Pokemon difference |
| Status inflicted | +0.1 | Opponent gains a status condition |
| Hazard set | +0.1 × layers | Entry hazard placed on opponent side |
| Hazard removed | +0.1 × layers | Clears own side hazards |

### Opponent Mixing

10% RandomPolicy / 10% FullOffensePolicy / 60% EMA opponent / 20% pool (past snapshots).

- **EMA Opponent**: Polyak-averaged online weights (decay=0.995, warmup=5 updates). Smooths policy oscillations.
- **Pool**: Periodic snapshots (every `pool_snapshot_freq` updates, threshold=0.55). Size capped at 10.

## Evaluation & Diagnostics

### Evaluation (every `eval_freq` updates)

500 games vs Random, FullOffense, EMA, and Pool opponents. Results saved to `eval.csv` and `eval_data/` (per-game JSON). Logged to wandb.

**Generated plots:**
- Action distribution histograms per opponent type
- Battle length distribution
- Reward decomposition (stacked bar chart per opponent)
- Value calibration (predicted V(s) vs actual return)
- Cross-attention heatmaps (DETR queries over state tokens)
- Cosine similarity matrices (5 matrices: own-own, opp-opp, DETR-DETR, opp-DETR, own-DETR)
- Transformer attention maps (per-head, per-layer grids)

### Probing System (every `probe_freq` evals)

Periodic linear probes on frozen representations to diagnose what each model component has learned. A fresh rollout is collected, tokens are cached at all model levels, then 4 probe suites run:

| Suite | What it probes | Key metrics |
|-------|---------------|-------------|
| **Actor** | Per-token backbone representations | Return R², Win AUC, type/item/ability accuracy, HP R², stats R², cross-token cosine sim |
| **Critic** | CLS token from independent critic | Value R², Win AUC, effective rank, PCA scatter, per-dim correlation |
| **DETR** | Action queries after cross-attention | Action chosen AUC, win probability, delta HP, KO prediction |
| **SVD** | Representation geometry at 3 levels | PCA visualization, energy spectrum, effective rank |

All probing results logged as `probe/*` metrics to wandb.

### Monitored Training Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| explained_variance | Critic quality: 1 - Var(return-value)/Var(return) | >0.3 by update 300-400 |
| clip_frac | Fraction of clipped policy ratios | 0.05-0.20 |
| attn_entropy | Cross-attention spread (higher = more distributed) | Should stabilize |
| attn_rank | Attention head diversity (higher = less collapsed) | Should stay high |
| move_recall | Prediction head quality for moves | >0.7 |
| critic_action_attn_entropy | Critic cross-attention over actions | Monitor for collapse |

## Key Files Reference

| File | Purpose |
|------|---------|
| `model/backbone.py` | Transformer, CLS token, actor cross-attention, attention map capture |
| `model/critic.py` | IndependentCritic — own Transformer, action-aware V(s,a), victory head |
| `model/agent.py` | Top-level forward pass, wires backbone + critic + heads |
| `model/embeddings.py` | Token embeddings (439-dim), ScalarRunningNorm, POMDP masking |
| `model/prediction_heads.py` | 5 prediction heads + masked losses |
| `env/state_encoder.py` | Battle state → feature vectors, vocabulary constants |
| `env/action_space.py` | 13 action embeddings (moves + mechanic + switch, DETR queries) |
| `env/revealed_tracker.py` | Parses PS protocol logs to track revealed opponent info |
| `training/self_play.py` | Main training loop, EMA, opponent pool, wandb logging |
| `training/rollout.py` | Episode collection, GAE, reward computation |
| `training/losses.py` | PPO-clip + value + entropy + prediction + attention reg losses |
| `training/evaluate.py` | Evaluation against opponents, cosine similarity diagnostics |
| `training/probing_eval.py` | Probing orchestration (runs 4 probe suites on fresh rollouts) |
| `training/monitor.py` | Eval plots generation |
| `training/attention_viz.py` | Transformer + cross-attention visualization |
| `diag/probing/` | 4 probe modules: actor, critic, DETR, SVD |
| `tests/run_all_tests.py` | Unified test runner |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Build Rust simulator
cd simulator && maturin develop --release && cd ..

# Train — cheater mode (full info, independent critic)
python run_cheater_v10.py

# Resume from checkpoint
python run_cheater_v10.py --resume checkpoints/<run_name>/agent_000100.pt

# Run test suite
python tests/run_all_tests.py

# Run probing standalone on a checkpoint
python -m diag.probing --checkpoint checkpoints/<run_name>/agent_000500.pt
```
