# CynthAI_v2 — Pistes moyen terme

## Corriger le générateur de teams

Le générateur actuel ne reproduit pas fidèlement le format Gen 9 Random Battle officiel.
Parser `randombattlesets.json` (source Pokémon Showdown) pour remplacer l'approximation actuelle
par une lookup table exacte (moves, items, EVs/natures par rôle, abilities).

## Bug critique : value_preds hors échelle (EV ≈ 0)

**Constat** (cheater_v5, u100) : `value_preds` mean=3.96, std=49, max=711 alors que `value_returns` sont z-scorés (mean=0, std=1). Le critic prédit dans un espace complètement différent des targets → EV ≈ 0, value loss ne converge pas, fonction de valeur quasi-constante.

**Cause** : dans `compute_gae` (rollout.py), les returns sont z-scorés (`ret = (ret - mean) / std`) mais `value_old` (les prédictions brutes du critic) n't pas normalisées. La value loss compare des prédictions brutes contre des targets normalisées.

**Fix** : soit normaliser `value_old` au même moment que les returns, soit stocker les returns bruts pour le logging et ne z-scorer que pour la loss. À investiguer dans `training/losses.py` et `training/rollout.py`.

**Fichiers** : `training/rollout.py` (compute_gae), `training/losses.py` (value loss), `training/evaluate.py` (logging value_preds/value_returns).

---

## Tera non implémenté dans le simulateur Rust

**Constat** (cheater_v5) : `p.get("terastallized")` retourne toujours `None` même après térastallisation. Le simulateur accepte la commande `move N terastallize` mais n'applique aucun effet et n'émet pas l'événement `|-terastallize|`. Résultat : `tera_used` est toujours `False`, les slots 4-7 sont légaux à chaque tour, et l'agent choisit des tera moves qui ne font rien.

**Fix court terme** : masquer définitivement les slots 4-7 dans `build_action_mask` (`tera_used = True` forcé) jusqu'à implémentation complète du tera dans le simulateur Rust.

**Fix long terme** : implémenter la térastallisation dans `pokemon-showdown-rs` (type boost, changement de type, émission `|-terastallize|`, mise à jour du champ `terastallized` dans le state).

**Fichiers** : `training/rollout.py` (`build_action_mask`), `simulator/src/` (Rust).

---

## Entraîner un bon cheater

Agent en information complète (mask_ratio=0.0, dense_scale=1.0) sur 2000-3000 updates.
Métriques cibles : win rate vs Random ≥ 70%, explained_variance ≥ 0.3, move recall ≥ 0.7.
Ce modèle servira de teacher pour la distillation vers le modèle POMDP (voir R1 dans TODO.md).

## SFT sur replays top ladder (pour OU)

Pas nécessaire pour Random Battle — devient critique pour OU où le méta est trop complexe
pour bootstrapper depuis zéro en self-play.

**Volume recommandé** : 5 000-10 000 replays filtrés ELO ≥ 1700 sur le ladder Smogon OU.
Chaque replay donne ~50-80 décisions par joueur, soit ~500k décisions au total — suffisant
pour une bonne couverture du méta sans trop de bruit.

Source : API Showdown (`replay.pokemonshowdown.com`) + archives tournois Smogon pour la
qualité maximale (quantité limitée mais décisions quasi-optimales).

Pipeline : télécharger les replays → reconstruire l'état step-by-step via le simulateur Rust
(rejouer les inputs → `get_state()`) → SFT avec BCE sur les actions humaines
(label_smoothing=0.1 pour absorber les erreurs humaines) → fine-tune PPO self-play OU.

---

## Algorithme d'entraînement : IMPALA / V-trace

Variante off-policy à actors asynchrones. Chaque actor collecte des trajectoires avec sa
propre copie des poids (légèrement en retard sur le learner) et les envoie dans un replay
buffer limité. Le learner tire des batches depuis ce buffer et corrige le biais off-policy
avec le correcteur V-trace :

```
ρ_t = min(ρ̄, π_θ(a|s) / π_μ(a|s))   # importance sampling clipé
```

**Avantages vs PPO synchrone** : le learner tourne en continu sans attendre la fin des rollouts,
ce qui améliore l'utilisation GPU. Avec 32 envs, l'accélération réelle dépend du ratio
compute/collect — à évaluer sur notre setup.

**Implémentation** : replay buffer circulaire (taille ~4-8 rollouts), actors qui pushent
dès qu'un épisode est terminé, learner qui consomme en continu. Plus complexe à déboguer
que PPO mais bien documenté (Espeholt et al. 2018).

---

## Self-play avancé : League Training (AlphaStar) ou PFSP

### League Training (AlphaStar)

Trois types d'agents dans la ligue :
- **Main agents** : s'entraînent contre tous les autres agents de la ligue
- **Main exploiters** : se spécialisent à trouver les failles du main agent uniquement,
  puis sont réinitialisés périodiquement
- **League exploiters** : trouvent les failles de toute la ligue, réinitialisés aussi

Chaque type a un rôle différent : les main agents apprennent une politique robuste, les
exploiters forcent les main agents à corriger leurs faiblesses. Plus complexe à opérer
(plusieurs checkpoints en parallèle, logique de réinitialisation).

### PFSP — Prioritized Fictitious Self-Play (plus simple, même paper)

Le poids d'un adversaire i dans le pool est `f(win_rate_i)` où f est décroissante :
```
w_i = f(win_rate_i) = 1 - win_rate_i        # linéaire
# ou
w_i = softmax(-win_rate_i / T)              # softmax inversée, T=température
```

L'agent joue plus souvent contre les adversaires qu'il perd le plus — il se concentre
sur ses faiblesses plutôt que de farmer des adversaires faciles. Simple à greffer sur
le pool existant : remplacer le sampling uniforme par un sampling pondéré par `w_i`.
`win_rate_i` est estimé sur une fenêtre glissante des derniers matchs contre cet adversaire.

---

## Architecture : mémoire et capacité temporelle

### Augmenter K (fenêtre temporelle)

Actuellement K=4 tours. Passer à K=8 ou K=12 donnerait plus de contexte historique
(captures de momentum, hazards posés il y a 6 turns, etc.). Coût : séquence plus longue
(K×13 tokens), attention quadratique. À K=8 : 104 tokens, toujours gérable.

### Transformer récurrent (GTrXL / R-Transformer)

Remplacer le sliding window par un état récurrent persistant entre les turns.
GTrXL (Gated Transformer-XL, DeepMind) est la référence pour le RL — il a été utilisé
dans AlphaStar. Plus complexe à entraîner (BPTT tronqué, gestion des épisodes).

### CLS token persistant (mémoire globale légère)

Ajouter un token CLS appris qui est passé dans un petit LSTM (hidden_size=256) entre
les turns. À chaque turn, le CLS post-Transformer est la nouvelle entrée du LSTM ;
l'état caché résultant devient le CLS du turn suivant (detaché du graphe pour éviter
le BPTT complet).

```python
cls_token, lstm_hidden = lstm(cls_post_transformer, lstm_hidden)  # détaché entre turns
# cls_token concaténé à la séquence du turn suivant
```

Plus simple que GTrXL (pas de segment-level recurrence complète), compatible avec
l'architecture actuelle. Coût : ~260K params supplémentaires, gestion du hidden state
dans le rollout buffer (stocker et resetter en fin d'épisode, comme OpenAI Five).

Différence avec un CLS statique : un CLS sans LSTM est juste un embedding appris fixe —
il ne s'adapte pas au déroulé de la bataille. Le LSTM est nécessaire pour que le token
accumule réellement de l'information au fil des turns.

---

## Value head : remplacer le mean-pooling

Le mean-pooling actuel (`tokens.mean(dim=1)`) traite tous les tokens de manière égale —
approximation grossière, le token du Pokémon actif devrait peser plus que les tokens
de bench faintés.

**Attention pooling** :
```python
scores = (tokens @ self.pool_query).squeeze(-1)   # [B, 13]
weights = F.softmax(scores, dim=-1)               # [B, 13]
pooled  = (weights.unsqueeze(-1) * tokens).sum(1) # [B, D]
```

**CLS token appris** : ajouter un token CLS à la séquence Transformer, utiliser sa
représentation post-Transformer pour la value head. Propre, sans paramètre supplémentaire
si CLS est juste un embedding appris concaténé à l'entrée.

---

## Scaling : modèles plus gros

Pour référence : OpenAI Five ~140M params, AlphaStar ~55M params, CynthAI_v2 actuel ~2.5M.

Axes de scaling :
- `d_model` : 256 → 512 ou 1024
- `n_layers` : 3 → 6 ou 8
- `n_heads` : 4 → 8
- FFN : 512 → 2048

Le scaling n'a de sens qu'une fois l'architecture et la boucle d'entraînement validées
sur le petit modèle. À envisager pour le passage à OU où la complexité stratégique
justifie plus de capacité.

---

## Envs asynchrones

Actuellement les 32 envs sont synchrones — le learner attend que tous aient terminé
leur rollout avant d'optimiser. Avec des envs asynchrones, chaque env envoie ses
transitions dès qu'un épisode se termine, et l'optimiseur tourne en continu.

Compatible avec IMPALA/V-trace (correction off-policy). Pour PPO pur, le gain est
limité car PPO est intrinsèquement on-policy — il faudrait accepter un léger biais
off-policy ou passer à V-trace.

---

## Évaluation : fréquence vs robustesse

100 parties donnent un intervalle de confiance à 95% de ±10% sur le win rate — trop
large pour détecter des progressions de 3-5%. 500 parties → ±4.5%, beaucoup plus
informatif.

**Proposition** : passer à `eval_n_games=500` et `eval_freq=250` (au lieu de 100/100).
On évalue deux fois moins souvent mais les métriques sont significatives. Le temps
d'eval passe de ~5 min à ~25 min tous les 250 updates — acceptable.

---

## MuZero / One-step lookahead

Principe : à chaque décision, simuler les issues des actions candidates (1 step) et
utiliser la value head comme heuristique pour scorer chaque branche. Similaire à
AlphaZero mais sans modèle de transition appris — on utilise le simulateur Rust directement.

```python
# À l'inférence (pas au training) :
for action in legal_actions:
    next_state = simulator.step(state, action)
    v_next = agent.value(next_state)
    score[action] = reward(action) + gamma * v_next
# Choisir argmax(score) au lieu de argmax(logits)
```


## Priorités et effort estimé

| Priorité | Item | Effort |
|----------|------|--------|
| 🔴 Quick wins | Normalisation scalaires bruts | 1-2h |
| 🔴 Quick wins | K=4 → K=8 | 1h |
| 🔴 Quick wins | eval_n_games 100 → 500 | 30min |
| 🔴 Quick wins | PFSP sur le pool existant | 2-3h |
| 🟡 Moyen terme | Corriger le générateur de teams | 1 jour |
| 🟡 Moyen terme | Entraîner le cheater | 2-3 jours GPU |
| 🟡 Moyen terme | CLS token + LSTM | 1-2 jours |
| 🟡 Moyen terme | Attention pooling (value head) | 2h |
| 🟡 Moyen terme | Pool snapshot win-rate gate | 1h |
| 🟡 Moyen terme | One-step lookahead (inférence) | 1 jour |
| 🟢 Long terme | IMPALA / V-trace | 1 semaine |
| 🟢 Long terme | League Training | 1 semaine+ |
| 🟢 Long terme | Scaling (d_model 256→512+) | 2-3 jours |
| 🟢 Long terme | SFT replays + OU | projet à part entière |

Les quick wins s'appliquent immédiatement sur le run en cours ou le suivant.
Le moyen terme suppose que le cheater valide l'architecture de base (EV ≥ 0.3,
WR vs Random ≥ 70%). Le long terme n'a de sens que si le projet passe à OU ou
nécessite un niveau compétitif réel.

Coût : N_legal_actions forward passes par décision (4-13×). Utilisable à l'inférence
et en éval pour mesurer le gain vs politique greedy pure.

---

## Normalisation des scalaires bruts

Les 223 scalaires dans `PokemonFeatures` ont des échelles très différentes :
- `hp_raw` : 1-714 (HP bruts)
- `stats` : 1-700+ (atk, def, etc.)
- `hp_ratio` : 0-1
- `boosts` : -6 à +6
- `volatiles` : 0/1 binaires

Sans normalisation, les scalaires à grande échelle dominent les gradients et les
distances dans l'espace d'embedding, ce qui nuit à l'apprentissage du Transformer.

À vérifier dans `env/state_encoder.py` : les stats brutes (`hp_raw`, `stats`) sont-elles
normalisées avant d'être concaténées aux autres scalaires ? Si non, diviser par une
constante fixe (ex: `hp_raw / 714`, `stats / 700`) ou appliquer une normalisation
par running mean/std sur le premier batch.

## Mix d'adversaires : supprimer Random, augmenter Pool

**Constat** (cheater_v5, ~u80) : L'agent bat RandomPolicy à >90% dès les premiers updates. Garder 10% Random n'apporte plus de signal utile.

**Proposition** : Passer de `10% rand / 10% FO / 60% EMA / 20% pool` à `0% rand / 10% FO / 60% EMA / 30% pool`.

```python
if roll < 0.10:
    opponent = FullOffensePolicy()   # 10% FO — plancher de difficulté
elif roll < 0.70:
    opponent = ema_opponent          # 60% EMA
else:
    opponent = pool or ema_opponent  # 30% pool (fallback EMA si vide)
```

**Fichiers** : `training/self_play.py` (~ligne 563), nouveau launcher v6.

---

## Pool snapshot : win-rate gate

Le patch P10c a retiré le gate sur les snapshots (periodic unconditional, tous les
100 updates). Problème : si l'agent régresse temporairement (instabilité PPO, changement
de phase curriculum), les snapshots corrompus polluent le pool et dégradent la qualité
des adversaires.

**Proposition** : réintroduire un gate léger sans revenir au système WR-only strict.

```python
# Snapshotter si l'une des deux conditions est vraie :
# 1. WR vs Random >= seuil (agent compétent)
# 2. Update % snapshot_forced_freq == 0 (snapshot forcé tous les N updates)
#    → garantit la diversité temporelle même en cas de régression prolongée

SNAPSHOT_WR_THRESHOLD = 0.45   # seuil bas — exclut seulement les effondrements
SNAPSHOT_FORCED_FREQ  = 300    # snapshot forcé tous les 300 updates quoi qu'il arrive
```

Ce double critère évite de snapshotter un agent en effondrement complet tout en
garantissant qu'une régression longue ne bloque pas la diversité du pool indéfiniment.
`WR vs Random` est la métrique la plus stable (peu de variance, adversaire fixe) — ne
pas utiliser WR vs EMA qui fluctue avec l'agent lui-même.
