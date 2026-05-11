# CynthAI_v2 — TODO

## Déjà implémenté (vérifié dans le code)

Tout ce qui est listé dans `ARCHITECTURE.md` et `README.md` est implémenté et fonctionnel :
POMDP masking, Reward Curriculum, Pool/opponent mixing, Régularisation, Critic, LR Schedule, Attention maps.

---

## P10 — Correction de l'Opponent Mixing (priorité #1)

**Pourquoi le training ne marche pas : un bug de masking asymétrique + un pool vide**

### Contexte : les 3 problèmes identifiés

**Problème A — POMDP masking asymétrique (le bug bloquant)**

Dans `collect_rollout()`, le masque de révélation n'est appliqué qu'à `agent_self`, pas à `agent_opp` :

```python
# Ligne 609-622 de training/rollout.py
if mask_ratio > 0.0:
    ins_self = (apply_reveal_mask(...),) + ins_self[1:]
    # ins_opp garde l'information complète — JAMAIS masqué
```

Quand le pool est vide (80% des parties), `pool.sample(agent)` retourne l'agent courant :
```python
class OpponentPool:
    def sample(self, current_agent):
        if not self._pool:
            return current_agent   # ← self-play
```

Donc dans 80% des cas, `agent_opp` est le **même objet** que `agent_self`. Résultat :
- `agent_self` joue avec un état masqué (ex: mask_ratio=0.5 cache 50% des infos adverses)
- `agent_opp` (le même réseau !) joue avec l'état complet
- **L'adversaire a un avantage informationnel systématique** → la partie est truquée

C'est pour ça que le win_rate training est à 1-3% au lieu de 50% attendu en self-play symétrique.
Plus `mask_ratio` monte (phase 2 → 0.5, phase 3 → 1.0), plus l'asymétrie empire.

**Problème B — Pool vide**

Le `snapshot_threshold=0.55` n'est jamais atteint pour 3 raisons :
1. Le pool eval utilise `pool.sample(agent)` qui retourne l'agent lui-même (pool vide)
2. La même asymétrie de masking s'applique pendant l'eval
3. L'agent ne peut pas battre une version full-info de lui-même

Conséquence : **0 snapshots dans le pool pendant les 1740 premières updates**, aucune diversité d'adversaires.

**Problème C — Value overestimation en cascade**

1. L'agent perd tout le temps (WR 1%) → `reward` = -1.0 (terminal) presque à chaque épisode
2. La critique apprend V(s) sur des états **masqués** (POMDP)
3. Sur un état masqué, la meilleure prédiction est une moyenne pondérée des issues possibles
4. Mais la moyenne des returns réels est ~ -0.8 (car pertes fréquentes)
5. La critique prédit V(s) ~ 7.0 (complètement décorrélé)
6. Les avantages `A = G - V(s)` sont tous très négatifs (~ -8)
7. Le policy reçoit le signal "tout est mauvais, quoi que tu fasses"
8. `policy_loss` ~ 0.001, `clip_frac` ~ 0.05 → **le policy n'apprend plus rien**

**Pourquoi V(s) monte à 7.0 si les returns sont négatifs ?** Hypothèse : avec le masking POMDP,
le modèle apprend à ignorer les tokens masqués (mis à zéro). La valeur sur ces tokens est proche
de zéro. Mais le mean-pooling du value head moyenne aussi les tokens non-masqués qui ont des
activations élevées. Combiné avec le fait que l'agent voit des états où l'info est partielle,
la valeur apprise est une convolution des vrais returns et de l'incertitude — elle dérive.

### Solutions

#### P10a — Symétriser le POMDP masking

Appliquer le même masque de révélation à `ins_opp` dans `collect_rollout()`, depuis la perspective
opposite (`tracker` suit déjà les révélations pour les deux côtés via `update(i, log_entries, state, side_opp)`).

**Fichier** : `training/rollout.py`, fonction `collect_rollout()`, autour de la ligne 609.

**Effet attendu** : en self-play (pool vide), les deux côtés ont la même information → win rate ~50%.

#### P10b — Remplacer le snapshot pool par une EMA des poids

Le système actuel (copie profonde quand WR > threshold) est fragile car :
- Dépend d'un seuil arbitraire
- Ne produit aucun snapshot si le seuil n'est pas atteint
- Les snapshots sont des sauts discrets, pas une progression lisse

**Solution : EMA Opponent** (Exponential Moving Average)

```python
class EMAOpponent:
    def __init__(self, agent, decay=0.995):
        self.ema = copy.deepcopy(agent)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, agent):
        with torch.no_grad():
            for ema_p, online_p in zip(self.ema.parameters(), agent.parameters()):
                ema_p.data.mul_(self.decay).add_(online_p.data, alpha=1 - self.decay)

    def sample(self, current_agent):
        return self.ema  # toujours disponible
```

Avantages :
- **Toujours disponible** : pas de pool vide, pas de seuil
- **Progression lisse** : l'adversaire suit l'agent avec un lag, pas de sauts brusques
- **Stabilité** : l'EMA atténue le bruit des mises à jour PPO, donnant une cible stable
- **Pas de deep copy à chaque snapshot** : l'EMA se met à jour en continu (O(n params) vs O(n params) aussi mais sans le pic mémoire du deep copy)

**Fichier** : Nouvelle classe dans `training/self_play.py`, modifications dans `train()`.

#### P10c — Conserver un OpponentPool réduit pour la diversité

L'EMA donne un adversaire principal stable, mais on peut garder un petit pool (size=5-10)
de snapshots espacés (tous les 200-500 updates) pour ajouter de la diversité.
Mélange : 70% EMA / 20% pool / 5% Random / 5% FullOffense.

**Fichier** : `training/self_play.py`, configuration du mixing.

---

## P11 — Stabilisation de l'entraînement

**Objectif** : Corriger l'overestimation de la value et améliorer le signal d'apprentissage,
une fois le bug P10a corrigé (sinon les métriques n'auront pas de sens).

### P11a — Normalisation des rewards / returns

Pourquoi : les valeurs de reward sont arbitraires (terminal ±1, KO ±0.05, HP ±0.01).
Sans normalisation, la valeur V(s) peut dériver et prendre des échelles ininterprétables
(comme les 7.0 observés). La loss value (MSE) n'est pas à une échelle stable.

Solution : **Reward normalization** (running estimate de la mean/std des rewards, qu'on
utilise pour normaliser les rewards avant GAE, puis dé-normaliser les avantages).

Ou **PopArt** (normalisation adaptative des targets de la value head).

Alternative plus simple : normaliser les returns dans `compute_losses()` avec un running mean/std.

**Fichier** : `training/losses.py` ou `training/rollout.py`.

### P11b — c_value augmenté

Pourquoi : `c_value=1.0` donne un poids égal à la loss value et à la loss policy.
Avec des avantages bruités (surtout pendant le warmup), la value peut dériver.
En augmentant à 2.0-4.0, on régularise plus fortement la valeur.

Attention : si P10a (masking symétrique) résout l'overestimation, c_value=1.0 pourrait
redevenir suffisant. À tester après P10a.

**Fichier** : `training/self_play.py`, `TrainingConfig.c_value`.

### P11c — min_steps augmenté

Pourquoi : `min_steps=1024` avec `n_envs=32` donne ~32 steps par env par rollout.
Les batailles durent 10-20 turns, donc chaque env termine ~1-2 batailles par rollout.
Avec 2 epochs de PPO sur les mêmes données, le rapport signal/bruit est faible.

Solution : passer à `min_steps=2048` (64 steps/env, ~3-4 batailles par rollout).

**Fichier** : `run_curriculum_max.py`, `TrainingConfig.min_steps`.

### P11d — Vérifier le gradient norm

`max_grad_norm=0.5` semble correct (`grad_norm` observé ~3-15, clipping actif).
Ne pas changer sauf si P10a+P11a+b+c modifient significativement la dynamique.

### P11e — Reward design check

Actuellement : terminal ±1, KO ±0.05, HP Δ ±0.01×Δadv.
Le Δ HP est très petit comparé au terminal (±1). Avec `dense_scale=0.5`, il devient 0.005×Δadv.
C'est négligeable. Le signal d'apprentissage vient quasi exclusivement du terminal (+/-1),
qui n'arrive qu'en fin de partie (10-20 steps). C'est un problème de reward sparse.

**Solution implémentée en P13c** :
- Augmenter `HP_ADV_SCALE` à 0.05 (×5)
- Ajouter un reward d'avantage de compteur (`COUNT_ADV_SCALE=0.03`)
- Voir P13c pour le détail complet

**Fichier** : `training/rollout.py`, constantes de reward + `compute_step_reward()`.

---

## P12 — Curriculum lié à la progression

**Objectif** : Les phases du curriculum (POMDP mask, reward dense) doivent être conditionnées
par la progression réelle de l'agent, pas par des breakpoints fixes.

### Pourquoi les breakpoints fixes posent problème

Le curriculum actuel utilise des breakpoints absolus (600, 2500 updates) :
```
Phase 1 (updates 0-600)   : mask=0.0, dense=1.0
Phase 2 (updates 601-2500): mask=0.5, dense=0.5
Phase 3 (updates 2501+)   : mask=1.0, dense=0.1
```

Si l'agent n'apprend pas (pool vide, masking asymétrique), le curriculum continue
d'avancer et empire la situation (mask=1.0 rend le jeu injouable).

### Proposition

Conditionner les transitions de phase sur la progression de l'agent :

1. **Phase 1** (mask=0.0, dense=1.0) : jusqu'à ce que l'EMA soit stabilisée
   → `pool_size ≥ 5` (pour l'ancien système) OU `EMA_counter ≥ 100 updates`
2. **Phase 2** (mask=0.5, dense=0.5) : jusqu'à ce que l'agent gagne contre l'EMA
   → `WR against EMA ≥ 55% sur 100 games`
3. **Phase 3** (mask=1.0, dense=0.1) : après phase 2

Alternative plus simple : utiliser des breakpoints plus tardifs (2000, 5000 au lieu de 600, 2500)
et se concentrer d'abord sur la fixation du système d'adversaires.

**Fichier** : `training/self_play.py`, `TrainingConfig` + logique dans `train()`.

---

## P13 — Améliorations architecturales

Ces améliorations concernent la structure même du modèle, indépendamment des bugs
de training (P10) ou de stabilisation (P11). Elles visent à rendre l'architecture
plus propre, plus efficace, et plus propice à l'apprentissage.

---

### P13a — Value Head : Pooling own+field uniquement

**Pourquoi** : Actuellement, la value head fait `current_tokens.mean(dim=1)` sur les
13 tokens (6 own + 6 opponent + 1 field). Après POMDP masking, les tokens adverses
sont partiellement zéros (espèce/item/ability cachés, stats masquées). Le mean-pool
mélange des représentations riches (own) avec des représentations dégradées (opp
masqué), ce qui injecte du bruit directement dans V(s). Le critique apprend sur
des features corrompues.

De plus, la value fonction V(s) estime le retour espéré depuis la perspective de
l'agent. Elle dépend de l'état de ses propres Pokémon et du field, pas directement
des tokens adverses (qui sont une source d'incertitude). Les tokens adverses dans
le pool forcent le critique à modéliser l'incertitude plutôt que la valeur.

**Solution** : Dans `BattleBackbone.encode()` (`model/backbone.py`), remplacer :

```python
pooled = current_tokens.mean(dim=1)                         # [B, D_MODEL]
```

Par :

```python
# Pool only own (0-5) + field (12) — exclude opponent (6-11)
own_and_field = torch.cat([current_tokens[:, :6, :],
                           current_tokens[:, 12:, :]], dim=1)  # [B, 7, D_MODEL]
pooled = own_and_field.mean(dim=1)                             # [B, D_MODEL]
```

**Option avancée — Weighted Pooling** : Remplacer le mean par une attention
learnable sur les 7 tokens (own + field) pour que le modèle apprenne à pondérer
le Pokémon actif plus fort que le bench. Mais le mean simple est plus robuste
et suffit probablement — le Transformer a déjà encodé l'importance relative
dans les représentations.

**Variante minimale** : Si on veut garder tous les tokens mais avec pondération
adaptative, on peut utiliser un weighted pooling appris :

```python
self.value_pool_weight = nn.Linear(D_MODEL, 1)  # dans __init__
weights = self.value_pool_weight(current_tokens).softmax(dim=1)  # [B, 13, 1]
pooled = (current_tokens * weights).sum(dim=1)  # [B, D_MODEL]
```

Mais cette approche ajoute des paramètres et le modèle pourrait apprendre à
quand même regarder les tokens adverses bruités. L'approche own+field seule
est plus propre architecturalement.

**Fichier** : `model/backbone.py`, méthode `encode()`.

---

### P13b — Masque de padding pour les tours 1 à 3 (début de partie)

**Pourquoi** : En début de partie, le `BattleWindow` n'a que 1 ou 2 vrais tours.
Les tours manquants (jusqu'à K=4) sont paddés avec des `PokemonFeatures()` vides
(tous les indices à UNK=0, scalars à 0). Sur une séquence de 52 tokens, jusqu'à
36 tokens (3 tours × 12 Pokémon + 3 field tokens) peuvent être du padding.

Le Transformer fait de l'attention bidirectionnelle complète (`batch_first=True`,
sans masque causal). Les tokens réels peuvent donc attendre sur les tokens de
padding, qui ont des représentations non-informatives mais non-nulles (après
projection, un vecteur de zéros devient le biais de la couche Linear).

Le modèle doit apprendre tout seul à ignorer ces tokens — c'est du gaspillage
de capacité, surtout en début d'entraînement. Sur des séquences aussi courtes
(52 tokens), ce n'est pas catastrophique, mais c'est une source de bruit évitable.

**Solution** : Dans `_build_sequence()`, détecter quels tokens sont du padding
en vérifiant si les features brutes (avant projection) sont toutes à zéro.
Retourner un `padding_mask` booléen [B, 52] et le passer comme
`src_key_padding_mask` au `TransformerEncoder`.

Détection du padding :
```python
# Les scalars sont les 222 dernières dimensions du token brut
# Un Pokémon de padding a tous ses scalars à 0 (hp_ratio=0, is_active=0, etc.)
is_padding_poke = pokemon_tokens[:, :, -222:].abs().sum(dim=-1) < 1e-6  # [B, K*12]

# Un field token de padding a tous ses champs à 0
is_padding_field = field_tokens.abs().sum(dim=-1) < 1e-6  # [B, K]

# Interleave en [B, K*13] puis reshape en [B, 52]
```

Puis dans `encode()` :
```python
seq = self._build_sequence(pokemon_tokens, field_tokens)
seq = self.transformer(seq, src_key_padding_mask=padding_mask)
```

**Note** : `TransformerEncoder.forward()` accepte `src_key_padding_mask` au format
`[B, S]` où `True` = ignorer ce token. C'est exactement le format qu'on produit.

**Fichier** : `model/backbone.py`, méthodes `_build_sequence()` et `encode()`.

---

### P13c — Réparation du signal de récompense (Sparse Rewards)

**Pourquoi** : Le reward dense actuel a deux problèmes qui s'aggravent en Phase II/III :

1. **Échelle trop petite** : `HP_ADV_SCALE = 0.01` × `dense_scale` (0.5 en Phase II,
   0.1 en Phase III) donne une contribution de 0.005 à 0.001 par step.
   Sur une battle de 15 tours, le reward dense total est ~0.015-0.075 contre
   ±1 de terminal. Le ratio est de 1:15 à 1:100 — le signal dense est noyé.

2. **Le Δ HP advantage est bruité** : l'avantage HP fluctue à chaque coup,
   switch, et soin. Le signal est instable et difficile à interpréter pour
   le critique, surtout avec un reward aussi petit.

**Solution en deux volets** :

**Volet A — Augmenter HP_ADV_SCALE** (simple, immédiat) :

```python
# rollout.py — constantes de reward
HP_ADV_SCALE = 0.05  # était 0.01 — 5x plus fort
```

Avec `dense_scale=0.1` (Phase III), le reward par step devient ~0.005 au lieu
de ~0.001. Sur 15 tours : ~0.075 vs ±1 — encore sparse mais 5x plus de signal.

**Volet B — Ajouter un reward d'avantage de compteur** (plus robuste) :

Le Δ HP est bruité à court terme. L'avantage de compteur (nombre de Pokémon
vivants de chaque côté) est plus stable et plus directement lié à l'issue :

```python
def _count_advantage(state, side_idx):
    own_alive = sum(1 for p in state["sides"][side_idx]["pokemon"]
                    if not p.get("fainted", False))
    opp_alive = sum(1 for p in state["sides"][1-side_idx]["pokemon"]
                    if not p.get("fainted", False))
    return (own_alive - opp_alive) / 6.0  # normalisé dans [-1, 1]

COUNT_ADV_SCALE = 0.03  # reward par unité de Δ compteur
```

Ce reward est plus sparse que le HP (change seulement sur KO) mais plus fiable.
Combiné avec le HP_ADV_SCALE, il donne un signal intermédiaire entre le dense
(chaque step) et le sparse (terminal).

**Fichiers** : `training/rollout.py`, constantes de reward + `compute_step_reward()`.

---

### P13d — Factorisation du Move Head (réduction de paramètres)

**Pourquoi** : Le move head actuel est une simple linéaire :
`Linear(D_MODEL=256, N_MOVES*4=2744)` = 256 × 2744 + 2744 = **705,688 paramètres**.
C'est 42% du total du modèle (~6.7M pour une tête auxiliaire).

Le problème n'est pas juste la taille — c'est que cette tête reçoit un signal
d'entraînement très sparse. Les moves adverses ne sont révélés que quand
PP < maxPP, ce qui arrive typiquement 1-2 fois par battle. Avec 705K paramètres
et peu de signal, le move head risque soit l'overfitting, soit le sous-apprentissage
(gradients dilués sur trop de paramètres).

**Solution** : Factoriser en deux couches avec bottleneck :

```python
# Actuel
self.move_head = nn.Linear(D_MODEL, N_MOVES * N_MOVE_SLOTS)  # 705K params

# Factorisé
self.move_head = nn.Sequential(
    nn.Linear(D_MODEL, 128),
    nn.ReLU(),
    nn.Linear(128, N_MOVES * N_MOVE_SLOTS),  # ~384K params
)
```

La couche cachée de 128 dimensions force une représentation compressée, ce qui
régularise naturellement l'apprentissage. Le nombre de paramètres passe de 705K
à 256×128 + 128 + 128×2744 + 2744 = 32,768 + 128 + 351,232 + 2744 ≈ **387K**.

Toutes les prédictions pour un slot (item, ability, tera, moves) passent de
~851K à ~535K params combinés — une réduction de 37% sans perte de capacité.

**Fichier** : `model/prediction_heads.py`, `PredictionHeads.__init__()`.

---

### P13e — Suppression de l'active_token dans les queries de move

**Pourquoi** : Dans `ActionEncoder.forward()`, chaque query de move est construite
en concaténant `[move_emb(32) + active_token(256) + scalars(2)]`. Le token du
Pokémon actif (celui qui utilise l'attaque) est donc directement dans la query.

En cross-attention produit-scalaire (Q·K), la query contenant `active_token` va
naturellement produire un score élevé avec... ce même `active_token` dans les keys.
Le mécanisme apprend un **self-match trivial** : "je suis l'attaque du Pokémon
actif, je matche avec le Pokémon actif".

Ça court-circuite le raisonnement stratégique. Le modèle n'a pas besoin de
regarder le Pokémon adverse ou le field pour scorer une action — il peut juste
se baser sur la similarité active_token↔active_token et apprendre une politique
qui dépend uniquement de l'identité du Pokémon actif.

**Solution** : Supprimer `active_exp` du `move_input`. La query de move devient
`[move_emb + pp_ratio + move_disabled]`, décrivant l'attaque sans référence
directe au lanceur. Le cross-attention doit alors chercher activement dans les
current_tokens pour trouver : le Pokémon actif (suis-je assez fort ?), le Pokémon
adverse (est-il faible à ce type ?), et le field (y a-t-il des hazards ?).

```python
# Actuel (action_space.py, ligne 64-67) :
active_exp = active_token.unsqueeze(1).expand(-1, 4, -1)       # [B, 4, D_MODEL]
move_input = torch.cat([mv_emb, active_exp, scalars], dim=-1)  # 32+256+2 = 290

# Corrigé :
move_input = torch.cat([mv_emb, scalars], dim=-1)              # 32+2 = 34
# Et ajuster move_proj : Linear(D_MOVE + 2, D_MODEL) au lieu de Linear(D_MOVE + D_MODEL + 2, D_MODEL)
```

**Fichier** : `env/action_space.py`, `ActionEncoder.__init__()` et `forward()`.

---

### P13f — Contexte global pour les queries d'action (switch + move)

**Pourquoi** : C'est le pendant du P13e pour les actions de switch. Actuellement,
les queries de switch sont `switch_proj(bench_tokens[i])` — la projection du
token du Pokémon de banc lui-même. Le cross-attention produit un score élevé
quand la query (le token du banc) matche la key (le même token du banc dans
current_tokens). C'est un autre **self-match** : "je switch sur Dracaufeu parce
que Dracaufeu est Dracaufeu", sans considérer la menace adverse.

Plus généralement, toutes les queries d'action manquent de **contexte global**.
Une bonne query devrait encoder "quelle décision je représente" et laisser le
cross-attention chercher l'information pertinente dans l'état.

**Solution** :

**1. Ajouter un paramètre `global_context` à `ActionEncoder.forward()`** :

```python
# Dans agent.py, avant l'appel à action_enc :
global_context = current_tokens.mean(dim=1)  # [B, D_MODEL] — pool sur les 13 tokens
```

**2. Pour les queries switch (slots 8-12)** — Remplacer la projection des bench
tokens par une query conditionnée sur le contexte global + un embedding de
position de slot :

```python
# __init__ : remplacer switch_proj par
self.switch_ctx_proj = nn.Linear(D_MODEL, D_MODEL)    # contexte global → query switch
self.switch_slot_emb = nn.Embedding(5, D_MODEL)       # identité "bench slot N"

# forward :
switch_base = self.switch_ctx_proj(global_context)               # [B, D_MODEL]
slot_ids = torch.arange(5, device=device)                        # [5]
switch_acts = (switch_base.unsqueeze(1) +
               self.switch_slot_emb(slot_ids).unsqueeze(0))      # [B, 5, D_MODEL]
```

Chaque query switch encode "je suis une action de switch vers le slot N, dans
ce contexte global de bataille". Le cross-attention doit alors raisonner :
"le slot N contient Dracaufeu (info dans les keys), l'adversaire a un type Eau
(info dans les keys), le field a des Stealth Rocks (info dans les keys) → score".

**3. Pour les queries move (slots 0-3) et mechanic (slots 4-7)** — Ajouter le
contexte global après la suppression de l'active_token (P13e) :

```python
# move_input devient [move_emb + scalars + global_context]
global_exp = global_context.unsqueeze(1).expand(-1, 4, -1)        # [B, 4, D_MODEL]
move_input = torch.cat([mv_emb, scalars, global_exp], dim=-1)     # 32+2+256 = 290
# move_proj: Linear(D_MOVE + 2 + D_MODEL, D_MODEL)
```

Le contexte global apporte une vue d'ensemble sans faire de self-match (contrairement
à active_token qui cible un token spécifique). La query décrit l'attaque dans le
contexte de la bataille, et le cross-attention fait le travail de matching fin.

**Fichiers** : `env/action_space.py` (modification de `ActionEncoder`),
`model/agent.py` (passage de `global_context`).

---

## P8 — Opponent Action Query System (futur lointain)

**Objectif** : Prédire l'action adverse et l'injecter dans la décision.

**Détail** :
- Nouveau module `OpponentActionHead` :
  - Extrait les moves disponibles de l'adversaire depuis l'état
  - Cross-attend queries adverses sur les tokens d'état (symétrique à l'actor head)
  - Loss de prédiction cross-entropy sur l'action adverse réelle
- Injection : concaténer la prédiction d'action adverse aux action_embeds
- À faire seulement après avoir résolu P10/P11 (la boucle d'apprentissage doit être stable d'abord)

**Fichiers** : Nouveau fichier `model/opponent_head.py`, modifications dans `model/agent.py`, `training/rollout.py`

---

## P9 — Analyse des matchups (futur)

**Objectif** : Analyser quels types de Pokémon / moves l'agent maîtrise ou non.

**Détail** :
- Collecter par type de Pokémon adverse le win rate de l'agent (e.g., vs Pokémon de type Feu, vs type Eau, etc.)
- Collecter par move catégorie (physique/spécial/statut) la fréquence d'utilisation et le win rate associé
- Corrélation entre le win rate et les variables d'état (nombre de Pokémon restants, terrain, etc.)
- Dashboard ou heatmap pour visualiser les forces/faiblesses de l'agent

**Fichiers** : Nouveau module `training/matchup_analysis.py`, intégration dans `training/evaluate.py`

---

## Notes diverses (non-bloquantes)

### N1 — Persistance du `sent_log_pos` dans PyBattle

`PyBattle.get_new_log_entries()` avance un compteur interne `sent_log_pos` dans le module Rust. Si un `PyBattle` est sérialisé/désérialisé, ce compteur est perdu et `get_new_log_entries()` retournerait tous les logs depuis le début, ce qui ferait traiter les révélations en double par `RevealedTracker`.

**Pas un problème actuellement** : les environnements sont recréés à chaque reset (fin d'épisode ou crash) et les checkpoints sauvegardent l'agent, pas les états de combat. Si un jour on sérialise des `PyBattle` en cours de partie, il faudra exposer `sent_log_pos` côté Python et le restaurer.

### N2 — Cas edge Zoroark / Illusion / Transform

Le `RevealedTracker` marque `species_revealed` dès qu'un Pokémon entre sur le terrain (`|switch|`). Mais Zoroark (talent Illusion) se présente sous l'apparence d'un autre Pokémon. Quand Illusion tombe (dégât direct), l'espèce réelle est révélée — le tracker ne gère pas ce désaveu. De même, Transform change types, stats et capacités sans que le tracker le détecte.

**Pas une priorité** : ces cas sont très rares en random battle. Solution simple quand le moment viendra : détecter le changement d'espèce via les logs (`|-illusion|`, `|-transform|`) et réinitialiser les attributs du slot concerné.

### N3 — Masque POMDP uniforme sur K=4 tours

Actuellement, `apply_reveal_mask()` applique le **même** masque de révélation aux K=4 tours du sliding window. Si l'item adverse est révélé au tour 5, les tours 2-3-4 dans l'historique du window montrent aussi l'item — comme si le modèle avait toujours su.

**Solution si nécessaire un jour** : stocker `reveal_state` à chaque push dans `BattleWindow`, et modifier `_build_sequence()` pour que `apply_reveal_mask()` reçoive un masque par turn au lieu d'un masque global. Pas prioritaire tant que les performances n'indiquent pas un plafond lié à ce comportement.