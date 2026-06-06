# CynthAI_v2 — TODO

Mis à jour : 2026-06-06

---

## Déjà implémenté ✅

| ID | Description | Détails |
|----|-------------|---------|
| P10a | Symétrisation du POMDP masking | `ins_opp` masqué symétriquement dans `rollout.py` |
| P10b | EMA Opponent | `EMAOpponent` dans `self_play.py`, decay=0.995 |
| P10c | Pool résiduel + snapshots périodiques | `pool_snapshot_threshold=0.55`, `pool_snapshot_freq=100` |
| P11a | Normalisation per-batch (z-score) | `compute_losses()` normalise advantages/returns |
| P11b | c_value 1.0 → 2.0 | Dans tous les launchers v7+ |
| P11c | min_steps 1024 → 4096 | `min_steps=4096` dans v9+ |
| P11e | Reward design (HP + count) | `HP_ADV_SCALE=0.05`, `COUNT_ADV_SCALE=0.03` |
| P13b | Padding mask (tours 1-3) | `src_key_padding_mask` dans `backbone.py` |
| P13c | Reward signal renforcé | HP ×5 + count advantage |
| P13d | Factorisation move head | Bottleneck 128, -318K params |
| P13e | Queries pre-transformer | Suppression du self-match `active_token` |
| P14 | Attention collapse fix | `c_attn_entropy` + `c_attn_rank` régularisation |
| P16a | Entropy tuning | Testé : `c_entropy=0.003` (v9b) et `0.01` (v10) en cours |
| P16b | PPO epochs 2 → 4 | `n_epochs=4` dans v9+ |
| P16d | Diagnostics advantages | `adv_mean`, `adv_std` loggés dans `losses.py` |
| P19 | Mix adversaires ajusté | `0% rand / 10% FO / 60% EMA / 30% pool` dans `self_play.py` |
| R3 | Générateur de teams | `generate_team_pool.js` → `team_pool.json` via vrai Showdown |
| — | Normalisation scalaires bruts | `ScalarRunningNorm` dans `embeddings.py` (EMA per-feature) |
| — | eval_n_games 500 | `eval_n_games=500` dans v9+ |
| — | Probing inline | `probing_eval.py` (SVD, actor, critic, DETR probes) |
| — | Victory head | `use_victory_head=True`, BCE loss dans `self_play.py` |
| — | Action-aware critic | Cross-attention sur action embeddings (v9+) |
| — | Tera masking | `tera_used` check dans `rollout.py`, slots 4-7 masqués si déjà tera |

---

## À faire — Court terme (pertinent maintenant)

### PFSP — Prioritized Fictitious Self-Play

Pondérer le sampling du pool par inverse du WR pour forcer l'agent à jouer contre ses adversaires les plus durs. Simple à greffer : remplacer le sampling uniforme par un sampling pondéré.

```python
w_i = 1 - win_rate_i   # ou softmax(-win_rate_i / T)
```

WR estimé sur une fenêtre glissante des derniers matchs contre chaque adversaire du pool.

**Effort** : 2-3h
**Fichiers** : `training/self_play.py` (opponent selection + pool tracking)

---

### Policy collapse — Investigation en cours

**Constat** (v9b) : WR vs EMA atteint 100% dès update 16, mais WR vs frozen opponent stagne à 26-35%. La policy devient trop déterministe et overfit l'EMA.

**Expériences en cours** :
- **v9b** : `c_entropy=0.003` — baseline, policy collapse confirmé
- **v10** : `c_entropy=0.01` (3x plus) — en cours, WR vs EMA monte plus lentement (48% à u12)

**Pistes si v10 ne résout pas** :
- Entropy annealing : `c_entropy * (1 - update/max_updates)` — commence haut, décroit
- Augmenter la proportion de frozen opponents dans le mix (actuellement 10% FO)
- Ajouter un mécanisme de diversity bonus (population-based)

---

## À faire — Moyen terme

### Rollout multiprocess workers (speedup ×6-8)

Le bottleneck est CPU : ~29s/update dont ~90% en rollout single-process. Avec 2 runs en parallèle, ça monte à ~30s+.

Architecture : N workers multiprocess, chacun avec n_envs/N envs et une copie no_grad du modèle. Workers poussent des `RolloutBuffer` via `mp.Queue`, main process agrège et fait le PPO update.

**Effort** : 2-3 jours
**Fichiers** : `training/rollout.py`, `training/self_play.py`

---

### P13a — Asymmetric Actor-Critic (Phase 2 POMDP)

Donner au critique l'état complet (unmasked) pendant que l'acteur reçoit l'état masqué. Technique éprouvée (Pinto et al. 2017). Déjà partiellement en place avec le critic indépendant — il suffit de lui passer les features non-masquées.

**Prérequis** : cheater convergé (Phase 1 terminée)
**Fichiers** : `model/backbone.py`, `training/rollout.py`, `training/self_play.py`

---

### P15b — Bug pred mask en mode POMDP

En mode cheater (mask_ratio=0), le bug est inactif — `build_targets()` fournit les masks ground truth. Mais quand `mask_ratio > 0` (Phase 2), les masks sont overridés par le `RevealedTracker` : les items ne sont presque jamais "révélés" → la tête item ne reçoit aucun gradient.

**Fix** : ne pas overrider les masks de `build_targets()` avec le `RevealedTracker` pour la loss auxiliaire. Le POMDP masking s'applique à l'état d'entrée, pas aux targets de supervision.

**Impact** : bloquant pour Phase 2
**Fichiers** : `training/self_play.py` (lignes 734-742)

---

### P15c — Tête de prédiction stats adverses (régression)

Ajouter un head `Linear(D_MODEL, 5)` pour prédire atk/def/spa/spd/spe des Pokémon adverses. Utile en POMDP car les natures sont aléatoires en Random Battle → stats varient pour une même espèce.

**Fichiers** : `model/prediction_heads.py`, `training/self_play.py`

---

### CLS token statique (value head)

Ajouter un token CLS appris en tête de la séquence (52 → 53 tokens). Il participe au self-attention et agrège l'info sur les K=4 turns. `post_tokens[:, 0, :]` alimente le value MLP au lieu du mean-pooling actuel.

Avantage : la value head voit l'historique complet au lieu du turn courant seul.

**Effort** : 2-3h
**Fichiers** : `model/backbone.py`

---

### Pool snapshot win-rate gate amélioré

Le gate actuel (`pool_snapshot_threshold=0.55`) est simple. Proposition : double critère — WR vs Random ≥ 0.45 (exclut les effondrements) OR snapshot forcé tous les 300 updates (diversité temporelle garantie).

**Effort** : 1h
**Fichiers** : `training/self_play.py`

---

### Tera dans le simulateur Rust (fix long terme)

Le simulateur accepte `move N terastallize` et émet `|-terastallize|`, mais les effets de type (STAB boost, type change) ne sont pas appliqués. Le masking des slots 4-7 fonctionne comme workaround.

**Effort** : 1 semaine
**Fichiers** : `pokemon-showdown-rs/src/`

---

## À faire — Long terme

### R1 — Distillation cheater → POMDP

Knowledge Distillation : le cheater figé fournit des soft targets (distributions de policy) à l'élève POMDP.

```
loss = α * PPO_loss(élève) + (1-α) * KL(π_élève || π_prof)
```

**Prérequis** : cheater convergé
**Fichiers** : `training/losses.py`, `training/self_play.py`

---

### R2 — SFT sur replays Showdown

Pré-entraînement sur replays humains (top-ladder ELO > 1800). Surtout pertinent pour OU où le méta est trop complexe pour bootstrapper en self-play.

Pipeline : télécharger replays → reconstruire état via simulateur Rust → SFT BCE sur actions humaines.

**Fichiers à créer** : `scripts/parse_replays.py`, `training/sft.py`, `run_sft.py`

---

### P8 — Opponent Action Query System

Prédire l'action adverse via un module `OpponentActionHead` (cross-attention symétrique à l'actor head) et injecter la prédiction dans la décision.

**Fichiers** : `model/opponent_head.py`, `model/agent.py`, `training/rollout.py`

---

### P9 — Analyse des matchups

Dashboard de WR par type adverse, catégorie de move, variables d'état. Heatmap forces/faiblesses.

**Fichiers** : `training/matchup_analysis.py`, `training/evaluate.py`

---

### Architecture — Mémoire et capacité

- **K=4 → K=8** : plus de contexte historique (~1h)
- **CLS + LSTM** : mémoire persistante inter-turns, détachée entre turns (1-2 jours)
- **GTrXL** : Transformer récurrent, référence DeepMind pour RL (complexe)

---

### Scaling

Pour référence : OpenAI Five ~140M, AlphaStar ~55M, CynthAI_v2 actuel ~4M.
Axes : d_model 256→512, n_layers 3→6, n_heads 4→8. Ne fait sens qu'une fois l'architecture validée.

---

### IMPALA / V-trace

Variante off-policy à actors asynchrones. Le learner tourne en continu sans attendre les rollouts. Compatible avec rollout multiprocess. Plus complexe à déboguer.

---

### League Training / PFSP avancé (AlphaStar)

Main agents + main exploiters + league exploiters. Plus complexe à opérer mais force la robustesse.

---

## Notes (non-bloquantes)

### N1 — Persistance `sent_log_pos` dans PyBattle
Si un jour on sérialise des `PyBattle` en cours de partie, il faudra exposer `sent_log_pos` côté Python. Pas un problème actuellement (envs recréés à chaque reset).

### N2 — Cas edge Zoroark / Illusion / Transform
Le `RevealedTracker` ne gère pas le désaveu d'espèce. Très rare en random battle. Fix : détecter `|-illusion|` / `|-transform|` dans les logs.

### N3 — Masque POMDP uniforme sur K=4 tours
`apply_reveal_mask()` applique le même masque aux K=4 turns. Fix si nécessaire : stocker `reveal_state` par turn dans `BattleWindow`.

### P12 — Curriculum lié à la progression
Les phases POMDP/dense sont à breakpoints fixes. Proposition : conditionner sur WR vs EMA. Non pertinent pour le cheater (pas de curriculum).
