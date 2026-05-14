# CynthAI_v2 — Pokémon AI with PPO Self-Play and Transformer Backbone

CynthAI_v2 is a deep reinforcement learning agent that plays Pokémon Showdown Gen 9 Random Battles. It uses a **Transformer backbone** (DETR-style cross-attention actor), **PPO self-play** with an EMA + pool opponent system, and a **Rust simulator** (PyO3 binding over `pokemon-showdown-rs`) for battle simulation.

## Project Structure

```
CynthAI_v2/
├── run_cheater_indep_critic.py  # Launcher — cheater mode (full info) + independent critic
├── run_cheater.py               # Launcher — cheater mode, shared critic
├── run_curriculum_max.py        # Launcher — 3-phase curriculum training
├── TODO.md                      # Known issues, roadmap, and proposed fixes
├── TODO2.md                     # Medium/long-term improvement ideas
├── requirements.txt             # Python dependencies
├── simulator.pyd                # Compiled Rust simulator (PyO3 wheel)
│
├── model/                       # Neural network modules
│   ├── agent.py                 # CynthAIAgent — top-level nn.Module
│   ├── backbone.py              # BattleBackbone — Transformer + value head + actor head
│   ├── critic.py                # IndependentCritic — separate Transformer for V(s)
│   ├── embeddings.py            # PokemonEmbeddings — token encoding + ScalarRunningNorm
│   └── prediction_heads.py      # PredictionHeads — auxiliary supervised heads
│
├── env/                         # Environment interface
│   ├── state_encoder.py         # State → feature vectors (PokemonFeatures, FieldFeatures)
│   ├── action_space.py          # ActionEncoder — 13 action embeddings
│   ├── revealed_tracker.py      # POMDP tracking — parses battle logs for opponent info
│   └── bots.py                  # Rule-based opponents (FullOffensePolicy)
│
├── training/                    # Training loop and utilities
│   ├── self_play.py             # PPO training loop + opponent pool + curriculum schedules
│   ├── rollout.py               # Episode collection, GAE, BattleWindow, action masking
│   ├── losses.py                # PPO-clip loss + entropy + auxiliary prediction loss
│   ├── evaluate.py              # Evaluation vs fixed opponents
│   ├── monitor.py               # Diagnostic plots (actions, battle length, value calibration)
│   ├── visualize.py             # Training curve visualization
│   ├── live_viz.py              # Real-time battle visualization (Pygame)
│   └── attention_viz.py         # Cross-attention weight visualization
│
├── data/dicts/                  # Vocabulary index files (JSON)
│   ├── species_index.json       # 1381 Pokémon species
│   ├── move_index.json          # 686 moves
│   ├── item_index.json          # 250 items
│   ├── ability_index.json       # 311 abilities
│   ├── type_index.json          # 19 types (incl. ???)
│   ├── type_embeddings.json     # Pre-computed type embeddings (log2 type chart, 8-dim)
│   └── move_embeddings.json     # Pre-computed move embeddings (type+category, 32-dim)
│
├── scripts/                     # Diagnostic and analysis scripts
│   ├── pool_check.py            # Inspect opponent pool snapshots
│   └── read_attn_maps.py        # Load and visualise saved attention maps
├── diag/                        # Cosine similarity and representation diagnostics
├── debug/                       # Debug scripts
├── simulator/                   # Rust simulator source (PyO3)
│   └── src/lib.rs               # PyBattle — PS protocol battle engine
├── tests/                       # Test suite
│   ├── run_all_tests.py         # Unified test runner (4 suites)
│   ├── test_full_pipeline.py    # Model pipeline tests (67 assertions)
│   ├── test_simulator.py        # Simulator + encoder tests
│   ├── test_encoder.py          # Vocabulary and encoding tests
│   └── test_attention_maps.py   # Attention map capture tests
└── checkpoints/                 # Saved training runs
    └── <run_name>/              # Per-run: agent_*.pt, metrics.csv, eval.csv, plots/
```

## Architecture Overview

**~2.53M parameters** (actor-only) / **~3.78M** with `IndependentCritic`.

### Model Pipeline

```
State dict (PyBattle) → State Encoder → ScalarRunningNorm → Token Embedding → Transformer Backbone
                                                                                  ├→ Value Head → V(s)  [backbone or IndependentCritic]
                                                                                  ├→ Actor Head (cross-attn) → 13 action logits
                                                                                  └→ Prediction Heads → opponent item/ability/tera/moves/stats
```

1. **State Encoding** (`env/state_encoder.py`): Converts `PyBattle.get_state()` dicts into `PokemonFeatures` and `FieldFeatures`. Each Pokémon token encodes species (32), 3× types (8 each), item (16), ability (16), 4× moves (32 each), and 223 scalar features (HP, stats, boosts, volatiles, status).

2. **Token Embedding** (`model/embeddings.py`): `PokemonEmbeddings` looks up 7 embedding tables, applies `ScalarRunningNorm` (per-feature EMA normalisation, updated during training and frozen at eval) to the 223 raw scalars, then concatenates everything → 439-dim vector per Pokémon. Type embeddings are seeded from log2 type chart; move embeddings from type+category priors.

3. **Transformer Backbone** (`model/backbone.py`): 3 layers, 4 heads, d_model=256, FFN=512, Pre-LN. Processes K=4 turns of 13 tokens each (6 own + 6 opponent + 1 field) = 52 tokens. Uses learned temporal + slot positional embeddings. Padding mask zeroes out empty/future turns.

4. **Actor Head (DETR-style)**: Action embeddings (13 queries) cross-attend over the 13 current-turn tokens (keys/values). Action queries use **pre-Transformer** tokens to avoid self-match shortcuts (P13e). Output: 13 logits (4 moves + 4 mechanic variants + 5 switches).

5. **Value Head**: Learned query scores each of the 13 current-turn tokens, softmax weights produce a weighted sum → Linear(256,256)-ReLU-Linear(256,1) → V(s). When `use_independent_critic=True`, a separate `IndependentCritic` (own Transformer, lr, weight decay, grad clip) replaces the backbone's value head entirely.

6. **Prediction Heads**: 5 independent Linear heads on opponent tokens predicting item (250 CE), ability (311 CE), tera type (19 CE), moves (686 BCE multi-label, top-4 recall at inference), and stats (6 MSE). Auxiliary supervised loss, masked to revealed slots.

### Action Space (13 slots)

| Slots | Description |
|-------|-------------|
| 0-3   | Regular moves (move 1-4) |
| 4-7   | Mechanic moves (same moves + Tera modifier) |
| 8-12  | Switch to bench Pokémon 0-4 |

Action embeddings are built dynamically from the active Pokémon's pre-transformer token, move embeddings, PP/disabled scalars, bench tokens, and mechanic type/ID.

### POMDP Masking

`RevealedTracker` parses PS protocol battle logs to track which opponent attributes have been revealed. `apply_reveal_mask()` zeros unrevealed categoricals and scalars (species, item, ability, tera, moves, PP, stats) with configurable probability `mask_ratio`.

## Training Setup

### PPO Configuration

| Parameter | Value |
|-----------|-------|
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Clip ε | 0.2 |
| Epochs | 4 |
| Batch size | 128 |
| Actor optimizer | AdamW (lr=2.5e-4, wd=1e-4) |
| Actor LR schedule | Linear warmup (20 steps) + cosine decay to 1e-5 |
| Actor max grad norm | 0.5 |
| Critic optimizer (indep.) | AdamW (lr=5e-4, wd=1e-4) |
| Critic max grad norm | 1.0 |
| Parallel envs | 32 |

### Loss Function

```
total = policy_loss + c_value × value_loss + c_entropy × entropy_loss + c_pred × pred_loss
      - c_attn_entropy × attn_entropy + c_attn_rank × (ln(13) − attn_rank)
```

- **Policy**: PPO-clip surrogate with advantage normalisation (per-batch z-score)
- **Value**: MSE against returns pre-normalised globally per rollout (z-score once in `compute_gae`, before minibatch splits). With independent critic, value loss uses the critic's own estimate.
- **Entropy**: Negative entropy over legal actions (encourages exploration)
- **Prediction**: Masked CE (item/ability/tera) + BCE multi-label (moves) + MSE×0.001 (stats) on opponent hidden state
- **Attn entropy** (P14): maximise per-query cross-attention entropy to prevent attention collapse
- **Attn rank** (P18): maximise von Neumann entropy of cross-attention singular values to prevent rank collapse

### Reward Design

| Component | Scale | Description |
|-----------|-------|-------------|
| Terminal win/loss | ±1.0 | Sparse, never scaled |
| Opponent KO | +0.05 | Scaled by `dense_scale` |
| Own KO | -0.05 | Scaled by `dense_scale` |
| Δ HP advantage | 0.01 × Δadv | Scaled by `dense_scale` |
| Δ count advantage | 0.01 × Δcount/6 | Alive Pokémon difference, scaled by `dense_scale` |
| Status inflicted | +0.01 | Opponent Pokémon gains a status condition |
| Hazard set | +0.01 × layers | Entry hazard placed on opponent side |
| Hazard removed | +0.01 × layers | Rapid Spin / Defog clears our side |

### Opponent Mixing

10% RandomPolicy / 10% FullOffensePolicy / 60% EMA opponent / 20% pool (past snapshots).
When the pool is empty, EMA fills the pool share.

- **EMA Opponent**: Polyak-averaged online weights (decay=0.995, warmup=5 updates). Always available, smooths policy oscillations.
- **Pool**: Periodic snapshots (every `pool_snapshot_freq` updates). Size capped at 10. Optional win-rate gate via `pool_snapshot_threshold`.

### Evaluation

Every `eval_freq` updates: `eval_n_games` games vs RandomPolicy, FullOffensePolicy, EMA, and pool. Results saved to `eval.csv` and `eval_data/eval_data_update*.json` (per-game detail).

## Key Files Reference

| File | Purpose |
|------|---------|
| `model/backbone.py` | Transformer, attention-pooling value head, actor cross-attention, attention map capture |
| `model/critic.py` | IndependentCritic — separate Transformer for V(s), decoupled from actor |
| `model/agent.py` | Top-level forward pass, wires backbone + critic + heads |
| `model/embeddings.py` | Token embeddings (439-dim), ScalarRunningNorm, POMDP masking, collation |
| `model/prediction_heads.py` | 5 prediction heads (item/ability/tera/moves/stats) + masked losses |
| `env/state_encoder.py` | Battle state → feature vectors, all vocabulary constants |
| `env/action_space.py` | 13 action embeddings (moves + mechanic + switch) |
| `env/revealed_tracker.py` | Parses PS protocol logs to track revealed opponent info |
| `training/self_play.py` | Main training loop, EMA opponent, opponent pool, curriculum schedules |
| `training/rollout.py` | Episode collection, GAE computation, action masking, reward components |
| `training/losses.py` | PPO-clip + value + entropy + prediction + attention reg losses |
| `training/evaluate.py` | Evaluation against fixed opponents with diagnostics |
| `tests/run_all_tests.py` | Unified test runner (4 suites, ~100 assertions) |
| `run_cheater_indep_critic.py` | Launcher for cheater_v4: full info + independent critic |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Build Rust simulator
cd simulator && maturin develop --release && cd ..

# Train — cheater mode (full info, independent critic) [recommended starting point]
python run_cheater_indep_critic.py

# Train — 3-phase curriculum
python run_curriculum_max.py

# Resume from checkpoint
python run_cheater_indep_critic.py --resume checkpoints/cheater_v4/agent_000100.pt

# Run test suite
python tests/run_all_tests.py
```
