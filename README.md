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
│   └── monitor.py             # Diagnostic plots (actions, battle length, value calibration)
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

**~6.7M parameters** total across all modules.

### Model Pipeline

```
State dict (PyBattle) → State Encoder → Token Embedding → Transformer Backbone
                                                              ├→ Value Head → V(s)
                                                              ├→ Actor Head (cross-attn) → 13 action logits
                                                              └→ Prediction Heads → opponent item/ability/tera/moves
```

1. **State Encoding** (`env/state_encoder.py`): Converts `PyBattle.get_state()` dicts into `PokemonFeatures` (438-dim per Pokémon) and `FieldFeatures` (72-dim). Each Pokémon token encodes species (32), 3× types (8 each), item (16), ability (16), 4× moves (32 each), and 222 scalar features (HP, stats, boosts, volatiles, status).

2. **Token Embedding** (`model/embeddings.py`): `PokemonEmbeddings` looks up 7 embedding tables and concatenates them with scalars → 438-dim vector per Pokémon. Type embeddings are seeded from log2 type chart; move embeddings from type+category priors.

3. **Transformer Backbone** (`model/backbone.py`): 3 layers, 4 heads, d_model=256, FFN=512, Pre-LN. Processes K=4 turns of 13 tokens each (6 own + 6 opponent + 1 field) = 52 tokens. Uses learned temporal + slot positional embeddings.

4. **Actor Head (DETR-style)**: Action embeddings (13 queries) cross-attend over the 13 current-turn tokens (keys/values). This is the same pattern as DETR's object queries — learnable queries attend over encoder output. Output: 13 logits (4 moves + 4 mechanic variants + 5 switches).

5. **Value Head**: Mean-pool over 13 current-turn tokens → Linear-ReLU-Linear → scalar V(s).

6. **Prediction Heads**: 4 independent Linear heads on opponent tokens predicting item (250), ability (311), tera type (19), and 4× moves (686). Auxiliary supervised loss, masked to revealed slots.

### Action Space (13 slots)

| Slots | Description |
|-------|-------------|
| 0-3   | Regular moves (move 1-4) |
| 4-7   | Mechanic moves (same moves + Tera modifier) |
| 8-12  | Switch to bench Pokémon 0-4 |

Action embeddings are built dynamically from the active Pokémon's backbone token, move embeddings, PP/disabled scalars, bench tokens, and mechanic type/ID.

### POMDP Masking

`RevealedTracker` parses PS protocol battle logs to track which opponent attributes have been revealed. `apply_reveal_mask()` zeros unrevealed categoricals and scalars (species, item, ability, tera, moves, PP, stats) with configurable probability `mask_ratio`.

## Training Setup

### PPO Configuration

| Parameter | Value |
|-----------|-------|
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| Clip ε | 0.2 |
| Epochs | 2 |
| Batch size | 128 |
| Optimizer | AdamW (lr=2.5e-4, wd=1e-4) |
| LR schedule | Linear warmup (20 steps) + cosine decay to 1e-5 |
| Max grad norm | 0.5 |
| Parallel envs | 32 |

### Loss Function

```
total = policy_loss + c_value × value_loss + c_entropy × entropy_loss + c_pred × pred_loss
```

- **Policy**: PPO-clip surrogate with advantage normalization
- **Value**: MSE with return normalization (per-batch z-score)
- **Entropy**: Negative entropy over legal actions (encourages exploration)
- **Prediction**: Masked cross-entropy on opponent hidden state

### Reward Design

| Component | Scale | Description |
|-----------|-------|-------------|
| Terminal win/loss | ±1.0 | Sparse, never scaled |
| Opponent KO | +0.05 | Scaled by `dense_scale` |
| Own KO | -0.05 | Scaled by `dense_scale` |
| Δ HP advantage | 0.01 × Δadv | Scaled by `dense_scale` |

### Curriculum (3 phases)

| Phase | Updates | Mask Ratio | Dense Scale | Description |
|-------|---------|------------|-------------|-------------|
| I — Foundations | 0-600 | 0.0 | 1.0 | Full info, full dense rewards |
| II — Transition | 601-2500 | 0.5 | 0.5 | Medium masking, medium sparse |
| III — Mastery | 2501+ | 1.0 | 0.1 | Full masking, near-sparse |

### Opponent Mixing

80% pool (past snapshots) / 10% RandomPolicy / 10% FullOffensePolicy. When pool is empty (no snapshots yet), pool.sample() returns the current agent (pure self-play).

### Evaluation

Every `eval_freq` updates: 500 games vs RandomPolicy, FullOffensePolicy, and pool. Agent is snapshotted into the pool when pool eval win rate exceeds 0.55.

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `model/backbone.py` | 316 | Transformer, value head, actor cross-attention, attention map capture |
| `model/agent.py` | 106 | Top-level forward pass combining all sub-modules |
| `model/embeddings.py` | 327 | Token embeddings (438-dim), POMDP masking, collation |
| `model/prediction_heads.py` | 159 | 4 Linear prediction heads + masked CE loss |
| `env/state_encoder.py` | ~400 | Battle state → feature vectors, all vocabulary constants |
| `env/action_space.py` | 82 | 13 action embeddings (moves + mechanic + switch) |
| `env/revealed_tracker.py` | 372 | Parses PS protocol logs to track revealed opponent info |
| `training/self_play.py` | 748 | Main training loop, opponent pool, curriculum schedules |
| `training/rollout.py` | 893 | Episode collection, GAE computation, action masking |
| `training/losses.py` | 110 | PPO-clip + value + entropy + prediction loss |
| `training/evaluate.py` | 222 | Evaluation against fixed opponents |
| `training/monitor.py` | 324 | Diagnostic plots from eval data |
| `env/bots.py` | 265 | FullOffensePolicy — rule-based damage maximizer |
| `run_curriculum_max.py` | 81 | Launcher with 3-phase curriculum config |
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