# CynthAI_v2 — Pokémon AI with PPO Self-Play and Transformer Backbone

CynthAI_v2 is a deep reinforcement learning agent that plays Pokémon Showdown Gen 9 Random Battles. It uses a **Transformer backbone** (DETR-style cross-attention actor), **PPO self-play** with curriculum learning, and a **Rust simulator** (PyO3 binding over `pokemon-showdown-rs`) for battle simulation.

## Project Structure

```
CynthAI_v2/
├── run_curriculum_max.py      # Main launcher — 3-phase curriculum training
├── TODO.md                    # Known issues, roadmap, and proposed fixes
├── memo_rust.md               # Rust simulator fixes documentation
├── requirements.txt           # Python dependencies
├── live_viz.py                # Real-time battle visualization (Pygame)
├── simulator.pyd              # Compiled Rust simulator (PyO3 wheel)
│
├── model/                     # Neural network modules
│   ├── agent.py               # CynthAIAgent — top-level nn.Module
│   ├── backbone.py            # BattleBackbone — Transformer + value head + actor head
│   ├── embeddings.py          # PokemonEmbeddings — categorical + scalar token encoding
│   └── prediction_heads.py    # PredictionHeads — auxiliary supervised heads
│
├── env/                       # Environment interface
│   ├── state_encoder.py       # State → feature vectors (PokemonFeatures, FieldFeatures)
│   ├── action_space.py        # ActionEncoder — 13 action embeddings
│   ├── revealed_tracker.py    # POMDP tracking — parses battle logs for opponent info
│   └── bots.py                # Rule-based opponents (FullOffensePolicy)
│
├── training/                  # Training loop and utilities
│   ├── self_play.py           # PPO training loop + opponent pool + curriculum schedules
│   ├── rollout.py             # Episode collection, GAE, BattleWindow, action masking
│   ├── losses.py              # PPO-clip loss + entropy + auxiliary prediction loss
│   ├── evaluate.py            # Evaluation vs fixed opponents
│   ├── monitor.py             # Diagnostic plots (actions, battle length, value calibration)
│   ├── visualize.py           # Training curve visualization
│   └── attention_viz.py       # Cross-attention weight visualization
│
├── data/dicts/                # Vocabulary index files (JSON)
│   ├── species_index.json     # 1381 Pokémon species
│   ├── move_index.json        # 686 moves
│   ├── item_index.json        # 250 items
│   ├── ability_index.json     # 311 abilities
│   ├── type_index.json        # 19 types (incl. ???)
│   ├── type_embeddings.json   # Pre-computed type embeddings (log2 type chart, 8-dim)
│   └── move_embeddings.json   # Pre-computed move embeddings (type+category, 32-dim)
│
├── scripts/                   # Data generation scripts
├── simulator/                 # Rust simulator source (PyO3)
│   └── src/lib.rs             # PyBattle — PS protocol battle engine
├── tests/                     # Test files
├── debug/                     # Debug/diagnostic scripts
└── checkpoints/               # Saved training runs
    └── curriculum_max_*/      # Per-run: agent_*.pt, metrics.csv, eval.csv, plots/
```

## Architecture Overview

**~2.5M parameters** total across all modules.

### Model Pipeline

```
State dict (PyBattle) → State Encoder → Token Embedding → Transformer Backbone
                                                              ├→ Value Head → V(s)
                                                              ├→ Actor Head (cross-attn) → 13 action logits
                                                              └→ Prediction Heads → opponent item/ability/tera/moves/stats
```

1. **State Encoding** (`env/state_encoder.py`): Converts `PyBattle.get_state()` dicts into `PokemonFeatures` (439-dim per Pokémon) and `FieldFeatures` (72-dim). Each Pokémon token encodes species (32), 3× types (8 each), item (16), ability (16), 4× moves (32 each), and 223 scalar features (HP, stats, boosts, volatiles, status).

2. **Token Embedding** (`model/embeddings.py`): `PokemonEmbeddings` looks up 7 embedding tables and concatenates them with scalars → 439-dim vector per Pokémon. Type embeddings are seeded from log2 type chart; move embeddings from type+category priors.

3. **Transformer Backbone** (`model/backbone.py`): 3 layers, 4 heads, d_model=256, FFN=512, Pre-LN. Processes K=4 turns of 13 tokens each (6 own + 6 opponent + 1 field) = 52 tokens. Uses learned temporal + slot positional embeddings. Padding mask zeroes out empty/future turns (all-zero field features) in the self-attention computation.

4. **Actor Head (DETR-style)**: Action embeddings (13 queries) cross-attend over the 13 current-turn tokens (keys/values). This is the same pattern as DETR's object queries — learnable queries attend over encoder output. Output: 13 logits (4 moves + 4 mechanic variants + 5 switches). Action queries are built from **pre-transformer** tokens to avoid self-match shortcuts (P13e).

5. **Value Head**: Mean-pool over 13 current-turn tokens → Linear(256,256)-ReLU-Linear(256,1) → scalar V(s).

6. **Prediction Heads**: 5 independent Linear heads on opponent tokens predicting item (250 CE), ability (311 CE), tera type (19 CE), moves (686 BCE multi-label, top-4 recall at inference), and stats (6 MSE: HP + atk/def/spa/spd/spe). Auxiliary supervised loss, masked to revealed slots.

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
| Optimizer | AdamW (lr=2.5e-4, wd=1e-4) |
| LR schedule | Linear warmup (20 steps) + cosine decay to 1e-5 |
| Max grad norm | 0.5 |
| Parallel envs | 32 |

### Loss Function

```
total = policy_loss + c_value × value_loss + c_entropy × entropy_loss + c_pred × pred_loss
      - c_attn_entropy × attn_entropy + c_attn_rank × (ln(13) − attn_rank)
```

- **Policy**: PPO-clip surrogate with advantage normalisation (per-batch z-score)
- **Value**: MSE against returns pre-normalised globally per rollout (z-score once in `compute_gae`, before minibatch splits)
- **Entropy**: Negative entropy over legal actions (encourages exploration)
- **Prediction**: Masked CE (item/ability/tera) + BCE multi-label (moves) + MSE×0.001 (stats) on opponent hidden state
- **Attn entropy** (P14): maximise per-query cross-attention entropy to prevent attention collapse
- **Attn rank** (P18): maximise von Neumann entropy of cross-attention singular values to prevent rank collapse

### Reward Design

| Component | Scale | Description |
|-----------|-------|-------------|
| Terminal win/loss | ±1.0 | Sparse, never scaled |
| Opponent KO | +0.5 | Scaled by `dense_scale` |
| Own KO | -0.5 | Scaled by `dense_scale` |
| Δ HP advantage | 0.5 × Δadv | Scaled by `dense_scale` |
| Δ count advantage | 0.3 × Δcount/6 | Alive Pokémon difference, scaled by `dense_scale` |
| Status inflicted | +0.1 | Opponent Pokémon gains a status condition, scaled by `dense_scale` |
| Hazard set | +0.1 × layers | Entry hazard placed on opponent side, scaled by `dense_scale` |
| Hazard removed | +0.1 × layers | Rapid Spin / Defog clears our side, scaled by `dense_scale` |

### Curriculum (3 phases)

| Phase | Updates | Mask Ratio | Dense Scale | Description |
|-------|---------|------------|-------------|-------------|
| I — Foundations | 0-600 | 0.0 | 1.0 | Full info, full dense rewards |
| II — Transition | 601-2500 | 0.5 | 0.5 | Medium masking, medium sparse |
| III — Mastery | 2501+ | 1.0 | 0.1 | Full masking, near-sparse |

### Opponent Mixing

10% RandomPolicy / 10% FullOffensePolicy / 60% EMA opponent / 20% pool (past snapshots).
When the pool is empty (early training), EMA is used for the pool share as well.

- **EMA Opponent**: Exponential Moving Average of the online agent weights (decay=0.995, warmup=5 updates). Provides a stable, always-available self-play opponent that smooths over the agent's oscillations during training. Updated after each PPO step.
- **Pool**: Periodic snapshots (every 100 updates) of the online agent. Size limited to 10 most recent snapshots.

### Evaluation

Every `eval_freq` updates: 100 games vs RandomPolicy, FullOffensePolicy, EMA opponent, and pool.
The agent is snapshotted into the pool every `pool_snapshot_freq=100` updates (regardless of win rate).

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `model/backbone.py` | 321 | Transformer, value head, actor cross-attention, attention map capture, padding mask |
| `model/agent.py` | 91 | Top-level forward pass combining all sub-modules |
| `model/embeddings.py` | 274 | Token embeddings (439-dim), POMDP masking, collation |
| `model/prediction_heads.py` | 238 | 5 prediction heads (item/ability/tera/moves/stats) + masked losses |
| `env/state_encoder.py` | 254 | Battle state → feature vectors, all vocabulary constants |
| `env/action_space.py` | 66 | 13 action embeddings (moves + mechanic + switch) |
| `env/revealed_tracker.py` | 313 | Parses PS protocol logs to track revealed opponent info |
| `training/self_play.py` | 756 | Main training loop, EMA opponent, opponent pool, curriculum schedules |
| `training/rollout.py` | 817 | Episode collection, GAE computation, action masking, reward components |
| `training/losses.py` | 99 | PPO-clip + value + entropy + prediction + attention reg losses |
| `training/evaluate.py` | 345 | Evaluation against fixed opponents with diagnostics |
| `training/monitor.py` | 435 | Diagnostic plots from eval data |
| `env/bots.py` | 211 | FullOffensePolicy — rule-based damage maximizer |
| `run_curriculum_max.py` | 69 | Launcher with 3-phase curriculum config |
| `simulator/src/lib.rs` | — | Rust PyBattle (PyO3 binding over pokemon-showdown-rs) |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Build Rust simulator
cd simulator && maturin develop --release && cd ..

# Train (fresh run)
python run_curriculum_max.py

# Resume from checkpoint
python run_curriculum_max.py --resume checkpoints/curriculum_max_20260510_2141/agent_001000.pt

# Evaluate a checkpoint
python -m training.evaluate --checkpoint checkpoints/agent_001000.pt --n-battles 200
```
