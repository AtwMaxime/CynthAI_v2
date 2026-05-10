# Architecture — CynthAI v2

Forward pass complet, shapes réelles.

---

## Vue d'ensemble

```
PyBattle.get_state()
        │
        ▼
[ state_encoder.py ]
  encode_pokemon() × 12  →  PokemonFeatures  (indices + 222 scalaires)
  encode_field()         →  FieldFeatures    (72 floats)
        │
        ▼
[ model/embeddings.py ]
  PokemonEmbeddings(PokemonBatch [B, K*12])
    species_embed   [K*12, 32]    ← initialisé depuis les base stats (hp/atk/def/spa/spd/spe)
    type1/2/tera    [K*12,  8] × 3  ← prior SVD de la matrice d'efficacité des types
    item_embed      [K*12, 16]
    ability_embed   [K*12, 16]
    move_embed×4    [K*12, 32] × 4  ← prior SVD de 46 attributs (type, power, catégorie, …)
    scalaires       [K*12, 222]
    ─────────────────────────────
    token           [K*12, TOKEN_DIM=438]    (K=4 → [B, 48, 438])

  field_proj(FieldBatch [B, K, 72])          (non embedé, projecté par backbone)
        │
        ▼
[ model/backbone.py — BattleBackbone ]

  ① _build_sequence()
       pokemon_proj  [B, 48, 438] → [B, 48, 256]
       field_proj    [B, 4,  72]  → [B, 4,  256]

       Reshape + concat par tour → [B, 4, 13, 256]

       + temporal_emb(4)   Embedding(4, 256)   [4, 256]  (0=oldest, 3=current)
       + slot_emb(13)      Embedding(13, 256)  [13, 256] (0-5=own, 6-11=opp, 12=field)

       flatten → [B, 52, 256]   (K*N_SLOTS = 4*13 = 52 tokens)

  ② TransformerEncoder  (3 layers, 4 heads, FFN=512, Pre-LN)
       [B, 52, 256] → [B, 52, 256]

  ③ current_tokens = output[:, -13:, :]   [B, 13, 256]

  ④ value_head  MLP(mean-pooled 256 → 256 → 1)  [B, 1]       ← critique V(s)

        │
        ▼
[ env/action_space.py — ActionEncoder ]
  Inputs: active_token [B, 256], move_idx [B, 4], pp_ratio [B, 4],
          move_disabled [B, 4], bench_tokens [B, 5, 256],
          mechanic_id [B], mechanic_type_idx [B]

  base_moves  = move_proj(cat[move_emb, active_exp, scalaires])   [B, 4, 256]
  mech_mod    = mechanic_proj(cat[type_emb, mech_onehot])         [B, 1, 256]
  mech_moves  = base_moves + mech_mod                             [B, 4, 256]
  switch_acts = switch_proj(bench_tokens)                         [B, 5, 256]

  action_embeds = cat[base_moves, mech_moves, switch_acts, dim=1] [B, 13, 256]

        │
        ▼
[ model/backbone.py — backbone.act() ]
  Cross-attention :
    query = action_embeds    [B, 13, 256]
    key   = current_tokens   [B, 13, 256]
    value = current_tokens   [B, 13, 256]
  → attn_out                 [B, 13, 256]
  → action_score(Linear 256→1).squeeze(-1)   [B, 13]
  → masked_fill(action_mask, -1e9)           [B, 13]   ← True = ILLEGAL

        │
        ▼
[ model/prediction_heads.py — PredictionHeads ]
  Input : current_tokens[:, 6:12, :]   [B, 6, 256]   (tokens adverses)

  item_head    Linear(256, 250)   → [B, 6, 250]
  ability_head Linear(256, 311)   → [B, 6, 311]
  tera_head    Linear(256, 19)    → [B, 6, 19]
  move_head    Linear(256, 686*4) → [B, 6, 4, 686]

  Loss : cross-entropy masquée sur les slots révélés (mask = idx != UNK)
         Aucune injection en v1. Voir BELIEF_STATE.md pour v2.
```

---

## Séquence complète dans `CynthAIAgent.forward()`

```python
# 1. Embeddings Pokémon
pokemon_tokens = poke_emb(poke_batch)              # [B, K*12, 438]

# 2. Transformer unique — encode état + valeur
current_tokens, value = backbone.encode(           # [B, 13, 256], [B, 1]
    pokemon_tokens, field_tensor
)

# 3. Embeddings d'actions
action_embeds = action_enc(                        # [B, 13, 256]
    active_token      = current_tokens[:, 0, :],
    move_idx          = move_idx,
    pp_ratio          = pp_ratio,
    move_disabled     = move_disabled,
    bench_tokens      = current_tokens[:, 1:6, :],
    mechanic_id       = mechanic_id,
    mechanic_type_idx = mechanic_type_idx,
)

# 4. Actor — cross-attention uniquement (pas de 2e Transformer)
action_logits = backbone.act(                      # [B, 13]
    action_embeds, current_tokens, action_mask
)
log_probs = F.log_softmax(action_logits, dim=-1)   # [B, 13]

# 5. Têtes auxiliaires sur tokens adverses
pred_logits = predictor(current_tokens[:, 6:12, :])
```

---

## Token slots (par tour)

```
Slot  0     : propre actif
Slots 1–5   : propre banc  (ordre équipe, positions actives exclues)
Slots 6–11  : adversaire   (même convention)
Slot  12    : field token

Séquence complète K=4 : [tour_0 | tour_1 | tour_2 | tour_3]  → 52 tokens
                          ^^^^^^ oldest              ^^^^^^ current
```

---

## Convention masque d'actions

`action_mask [B, 13] bool` — **True = ILLÉGAL** dans tout le code.

```
Slots 0–3  : moves de base
Slots 4–7  : moves avec mécanique (Tera/Mega/Z/Dynamax)
Slots 8–12 : switchs (banc, dans l'ordre team_pos croissant)
```

Dérivé depuis `get_state()` dans `build_action_mask()` — le simulateur Rust n'expose pas de `get_legal_actions()`.

---

## Boucle PPO

```
collect_rollout()
  │  n_envs=32 combats parallèles
  │  épisodes complets (pas de troncature à longueur fixe)
  │  both agents sous torch.no_grad()
  │  transitions : champs int du PokemonBatch stockés (pas les float embeddings)
  │    → re-run de poke_emb() avec gradients à l'entraînement
  │  GAE en fin de buffer  γ=0.99, λ=0.95
  ▼
training loop (n_epochs=2)
  │  minibatches de taille 128 (shufflés)
  │  forward agent (avec grad)
  │  pred_loss  = PredictionHeads.compute_loss(pred_logits, *targets)
  │  losses     = compute_losses(logits_new, log_prob_old, actions, …)
  │  total.backward()
  │  clip_grad_norm_(agent, 0.5)
  │  Adam.step()
  ▼
scheduler.step()   # Warmup (20) + Cosine : 2.5e-4 → 1e-5
```

---

## Récompenses

| Événement          | Valeur            |
|--------------------|-------------------|
| Victoire           | +1.0              |
| Défaite            | −1.0              |
| KO adverse         | +0.05             |
| KO propre          | −0.05             |
| Δ avantage HP      | +0.01 × Δadv      |

---

## SVD Embedding Priors

Deux tables d'embedding sont initialisées avec des priorit SVD au lieu d'un tirage aléatoire, pour donner un sens sémantique dès le début de l'entraînement.

### Types (`D_TYPE=8`)

1. Matrice d'efficacité `M[18×18]` : `M[i][j]` = multiplicateur de dégâts quand le type `i` attaque le type `j` (2×, 1×, ½×, 0×).
2. Concaténation profil offensif + défensif → `features[18×36]`.
3. SVD → 8 dimensions → `type_prior[19×8]` (row 0 = __unk__ zeros).

Stocké dans `data/dicts/type_embeddings.json`, chargé via `TYPE_EMBEDDING_PRIOR`.

### Moves (`D_MOVE=32`)

1. Vecteur de 46 attributs par move :
   - `basePower/250`, `accuracy/100`, `pp/40`, `priority`, `critRatio`
   - Type one-hot (19), catégorie one-hot (3)
   - 20 flags binaires : recoil, drain, multihit, status, boosts, selfSwitch, forceSwitch, breaksProtect, heal, contact, protect, mirror, snatch, bypasssub, isZ, isMax, …
2. SVD pondéré par valeurs singulières → 32 dimensions.
3. Stocké dans `data/dicts/move_embeddings.json`, chargé via `MOVE_EMBEDDING_PRIOR`.

### Species

Initialisées depuis les base stats (`hp/atk/def/spa/spd/spe` normalisés par 255) dans `PokemonEmbeddings._init_species_from_base_stats()`. Les espèces avec des stats proches commencent proches dans l'espace d'embedding.

---

## Attention Maps (Interprétabilité)

`BattleBackbone.get_attention_maps()` permet de capturer les poids d'auto-attention de chaque couche du Transformer pour un état donné.

```python
result = agent.backbone.get_attention_maps(pokemon_tokens, field_tokens)
# result["attention_maps"]  →  list of N_LAYERS × [B, 4, 52, 52]
# result["value"]           →  [B, 1]
# result["current_tokens"]  →  [B, 13, 256]
# result["token_labels"]    →  list of 52 str labels
```

### Labels des tokens (format `_token_labels()`)

```
T0_own0  …  T0_own5  T0_opp0  …  T0_opp5  T0_field   (tour 0, le plus vieux)
T1_own0  …  T1_own5  T1_opp0  …  T1_opp5  T1_field   (tour 1)
T2_own0  …  T2_own5  T2_opp0  …  T2_opp5  T2_field   (tour 2)
T3_own0  …  T3_own5  T3_opp0  …  T3_opp5  T3_field   (tour 3, actuel)
```

### Implémentation

La méthode itère manuellement sur chaque couche du Transformer en appelant `self_attn()` avec `need_weights=True` pour capturer les matrices `[B, H, 52, 52]`, en contournant le `_sa_block` par défaut (qui utilise la fastpath PyTorch et ne retourne pas les poids d'attention).
