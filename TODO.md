# CynthAI_v2 — TODO

## Déjà implémenté (vérifié dans le code)

Tout ce qui est listé dans `ARCHITECTURE.md` et `README.md` est implémenté et fonctionnel :
POMDP masking, Reward Curriculum, Pool/opponent mixing, Régularisation, Critic, LR Schedule, Attention maps.

**P10a** — Symétrisation du POMDP masking (ins_opp) ✅
**P10b** — EMA Opponent (remplace snapshot pool) ✅
**P10c** — Pool résiduel avec snapshots périodiques ✅
**P11a** — Normalisation per-batch des avantages/returns (z-score dans compute_losses) ✅
**P11b** — c_value 1.0 → 2.0 ✅
**P11c** — min_steps 1024 → 2048 ✅
**P13b** — Masque de padding pour les tours 1 à 3 (padding mask dans backbone) ✅
**P13c** — Reward signal : HP_ADV_SCALE ×5 + COUNT_ADV_SCALE 0.03 ✅
**P13d** — Factorisation du move head (bottleneck 128, -318K params) ✅
**P13e** — Queries d'action construites à partir des tokens pre-transformer (pas de self-match) ✅

---

## P14 — Attention collapse (cross-attention action → FIELD uniquement)

**Constat** (update 1000) : Les 13 action queries (M1-M4, T1-T4, S1-S5) attendent toutes exclusivement sur le token FIELD (colonne 12). La carte d'attention est une matrice 13×13 avec uniquement la dernière colonne à 1, le reste à 0. Ce comportement a commencé vers update 800 et s'est complètement installé à update 1000.

**Cause probable** : Le token FIELD a une norme ou une distribution différente des tokens Pokémon, ce qui en fait une cible "facile" pour l'attention softmax. Le modèle prend ce raccourci car il suffit à générer une politique approximative, mais il ne peut plus s'améliorer au-delà.

**Impact** : Le win rate plafonne à ~50% car les action queries n'extraient plus aucune information des Pokémon (own ou opp). La politique est aveugle aux matchups, aux menaces, aux opportunités.

**Solutions candidates** :
1. **Régularisation d'entropie d'attention** — Ajouter une loss `H(α) = -Σ α log(α)` sur les poids d'attention cross-attention pour forcer la dispersion sur plusieurs tokens. Léger (1 scalaire par batch), efficace.
2. **Attention dropout plus élevé** — Passer dropout de 0.1 à 0.3 dans `nn.MultiheadAttention` pour forcer le modèle à ne pas dépendre d'un seul token.
3. **Gating / Skip connection** — Ajouter `output = cross_attn(q, kv) + α * q` avec α appris pour court-circuiter l'attention si nécessaire.
4. **Query diversity loss** — Pénaliser la similarité entre patterns d'attention des différentes queries.

**Fichiers** : `model/backbone.py` (cross-attention), `training/losses.py` (régularisation).

---

## P15 — Têtes de prédiction : accuracy anormale (item=0%, ability/move=100%)

**Constat** (update 1000) : Les métriques d'accuracy des têtes de prédiction sont pathologiques :
- **Item accuracy : 0.00%** — La tête n'a jamais prédit un objet correct. Soit les objets ne sont jamais révélés dans l'état, soit il y a un bug de ciblage.
- **Ability accuracy : ~100%** — Trop parfait. Probablement parce que l'ability est toujours révélée dès le départ (espèce connue → ability déduite). La tâche est triviale.
- **Move accuracy : ~100%** — Idem, les moves sont peut-être toujours connus.

**Impact** : La loss de prédiction (pred_loss) est proche de zéro car les têtes ability/move dominent avec leur accuracy parfaite, tandis que item/tera n'apprennent rien. Le gradient de la loss auxiliaire n'apporte pas d'information utile au backbone.

**À investiguer** :
- Est-ce que les items sont bien présents dans le state dict PyBattle (`item` field dans chaque Pokémon) ?
- Est-ce que `build_targets()` dans `prediction_heads.py` reçoit bien des targets non-UNK pour item/tera ?
- Vérifier si `RevealedTracker` révèle correctement les items via les logs PS (`|item|`).
- Vérifier le dataflow complet : state dict → PokemonBatch → build_targets → compute_loss.

**Fichiers** : `model/prediction_heads.py`, `training/rollout.py` (passage du full_opp_batch), `env/revealed_tracker.py`.

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

### P11a — Normalisation des avantages / returns

**Statut : ✅ Fait** — normalisation per-batch (z-score) dans `compute_losses()`.

Les avantages et returns sont normalisés par batch (mean=0, std=1) avant la policy loss
et la value loss. Voir `training/losses.py` lignes 61-62 et 76-80.

**Si V(s) dérive encore** : implémenter un running std sur les rewards denses (EMA)
avant GAE. L'idée est de diviser les rewards par `running_std` (sans soustraire la
moyenne, pour préserver le signe victoire/défaite). Le terminal (±1) reste brut :
```
# Dans collect_rollout, avant GAE :
reward_norm = reward / max(running_std, 1e-8)  # std seulement, pas de centrage
```
Ce running std s'adapte aux phases du curriculum via EMA (momentum ~0.01).
**Fichier** : `training/rollout.py` (+ nouvelle classe `RunningNormalizer`).

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

### P13a — Asymmetric Actor-Critic (privileged information pour le critique)

**Pourquoi** : Actuellement, le critique et l'acteur reçoivent le même état masqué
(POMDP). Le critique doit estimer V(s) sur des tokens partiellement zéros, ce qui
dilue le signal de valeur et contribue à l'overestimation.

**Solution** : Donner au critique l'état complet (unmasked) pendant que l'acteur
reçoit l'état masqué. C'est une technique éprouvée en RL (Pinto et al. 2017,
"Asymmetric Actor Critic for Image-Based Robot Learning") :

```python
# Forward acteur avec masking (comme aujourd'hui)
masked_poke = apply_reveal_mask(unmasked_poke, ...)
out = agent(masked_poke, ...)   # acteur + critique sur état masqué

# Forward critique séparé avec état complet
with torch.no_grad():  # ou en passant le masking dans le détachage
    out_full = agent.backbone.encode(unmasked_poke, field_tensor)
    value_full = out_full.value  # [B, 1] — V(s) sur info complète

# PPO : utiliser value_full pour les avantages, out.action_logits pour la policy
```

**Variantes** :
1. **Deux forward passes** : acteur sur masqué, critique sur complet → plus lent
   mais propre
2. **Forward unique avec détachage** : un seul passage Transformer, critic head
   séparée qui skip le masking → Plus efficace, mais demande de dupliquer le
   backbone ou d'ajouter une tête value non masquée
3. **Shared backbone, 2 value heads** : le backbone encode l'état masqué,
   mais on ajoute une `full_value_head` entraînée séparément sur les vrais
   tokens (non masqués). Le plus simple : garder la value_head actuelle sur le
   masqué pour l'acteur, et ajouter une petite MLP head sur les tokens bruts.

**Quand faire** : Après P10a (masking symétrique) et P10b (EMA). Si le win rate
ne décolle toujours pas, l'asymmetric AC est la prochaine carte à jouer.

**Note** : Si on implémente l'asymmetric AC, le problème du pooling des tokens
adverses ne se pose plus (le critique voit l'état complet, tous les tokens sont
informatifs). P13a rend donc l'ancienne proposition "Value Head own+field only"
obsolète.

**Fichiers** : `model/backbone.py`, `training/rollout.py` (passage des features
complètes au critique), `training/self_play.py` (loss computation).

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

---

## P15b — Bug: le mask RevealedTracker écrase la ground truth pour la perte auxiliaire

**Constat** : `item_acc = 0%` et `tera_acc = 0%` depuis le début de l'entraînement.

**Cause racine** : dans `self_play.py` lignes 604-611, le mask de la loss de prédiction est **systématiquement remplacé** par les données du `RevealedTracker` (POMDP), même quand la vérité terrain du simulateur est disponible :

```python
# self_play.py L604-611 — BUG: override inconditionnel
pred_loss = PredictionHeads.compute_loss(
    out.pred_logits,
    targets[0], targets[1], targets[2], targets[3],
    item_mask    = batch["reveal_item"],     # ← RevealedTracker (presque toujours False)
    ability_mask = batch["reveal_ability"],  # ← pareil
    tera_mask    = batch["reveal_tera"],     # ← pareil
    move_mask    = batch["reveal_moves"],    # ← pareil
)["total"]
```

Le `RevealedTracker` ne set `item=True` que sur `|-enditem|` ou `[from] Item` dans les logs — des événements rares. En phase II (mask_ratio=0.5), les items adverses sont quasiment jamais "révélés" → `item_mask = False` pour 99% des transitions → **la tête item ne reçoit aucun gradient**.

**Solution** : Utiliser `build_targets()` qui compare `item_idx != UNK` — le simulateur connaît toujours la vérité terrain. Les slots UNK (hors-liste, bench vide) sont naturellement exclus. Les masks RevealedTracker doivent servir pour le POMDP, pas pour la perte auxiliaire.

```python
# Fix: utiliser le mask de build_targets (ground truth du simulateur)
targets = PredictionHeads.build_targets(opp_batch)
item_mask, ability_mask, tera_mask, move_mask = targets[4], targets[5], targets[6], targets[7]
```

**Note** : `ability_mask` et `move_mask` sont probablement aussi affectés — les abilities sont parfois évidentes de l'espèce mais pas toujours. La ground truth du simulateur est fiable pour toutes ces cibles.

**Fichiers** : `training/self_play.py` (lignes 602-623), `model/prediction_heads.py` (`build_targets`).

---

## P15c — Têtes de prédiction pour les stats brutes adverses (regression)

**Objectif** : Ajouter une tête de régression pour prédire les stats actuelles (atk, def, spa, spd, spe) de chaque Pokémon adverse. Dans le POMDP, l'agent ne voit pas ces stats — elles sont critiques pour évaluer la menace.

**Conception** :
```python
# Nouveau head dans PredictionHeads
self.stats_head = nn.Linear(D_MODEL, 5)  # atk, def, spa, spd, spe

# Perte: MSE masquée (masque = slot non-UNK)
# Les stats sont présentes dans PokemonFeatures.stats (5 scalaires normalisés)
```

**Pourquoi c'est utile** : En Random Battle, le nature est aléatoire → les stats brutes varient pour une même espèce. Prédire les stats adverses permet à l'agent d'estimer la menace (ex: Dracaufeu a-t-il du SpA boosté par nature ?) sans avoir à les voir directement.

**Fichiers** : `model/prediction_heads.py`, `training/self_play.py`, `training/losses.py` (c_stats coefficient).

---

## P16 — Ajustements des hyperparamètres PPO

### P16a — Réduire l'entropy bonus

Le `c_entropy=0.01` actuel, combiné à une entropy de ~2.3 nats (sur max 2.56 pour 13 actions), signifie que **l'entropie contribue à ~0.023 de la loss totale** (~0.72). Le policy_loss est ~0.001. **Ratio entropy/policy ≈ 23:1** — l'entropie domine complètement.

**Solution** : Baisser `c_entropy` à 0.001-0.003, ou utiliser un annealing (`c_entropy * (1 - update/max_updates)`).

### P16b — Augmenter les PPO epochs

Actuellement `n_epochs=2` avec `clip_frac=0.03`. Le clipping est quasi inactif → le modèle peut supporter plus d'itérations sans dépasser le ratio 1.2.

**Solution** : Passer à 4-8 epochs. Le coût compute est ~2-4× sur la partie PPO (déjà rapide comparé au rollout).

### P16c — Ajuster le mix d'adversaires

Actuellement : 5% FO / 5% Random / 70% EMA / 20% pool. L'agent ne voit presque jamais FO, donc n'apprend jamais à le contrer.

**Solution** : Passer à 20% FO / 10% Random / 50% EMA / 20% pool. Ou utiliser un curriculum d'adversaires (augmenter FO progressivement).

### P16d — Diagnostiquer les advantages

Ajouter dans les logs de `compute_losses()` :
```python
# Dans losses.py, avant de return:
"adv_mean": advantages.mean().detach().item(),
"adv_std":  advantages.std().detach().item(),
"ratio_mean": (ratio - 1.0).abs().mean().detach().item(),
```

Si `adv_std < 0.5` après normalisation ou `ratio_mean < 0.01`, le signal policy est trop faible.

---

## R1 — Distillation depuis un modèle tricheur (Cheater → Partial Info)

**Objectif** : Utiliser un modèle entraîné en information complète (mask_ratio=0.0 sur toute la durée) comme "professeur" pour guider l'apprentissage d'un modèle entraîné en information partielle. Le modèle tricheur a accès à l'état adverse complet (espèce, moves, stats, item) — il apprend une politique supérieure que l'agent POMDP ne peut pas atteindre directement.

**Principe** : Knowledge Distillation (Hinton et al. 2015). Le modèle tricheur fournit des distributions de policy "douces" (soft targets) au lieu de labels one-hot. L'élève (modèle partiel) est entraîné avec :

```
loss = α * PPO_loss(élève) + (1-α) * KL(π_élève || π_prof)
```

Le KL est calculé sur les distributions de policy (logits), pas sur les actions individuelles.

**Pourquoi ça marche ici** :
- Le modèle tricheur connaît les bonnes actions même sur les états que l'élève voit comme ambigus
- La distribution du prof encode plus d'information que le signal RL sparse (victoire/défaite)
- L'élève apprend à imiter le comportement du prof sur les états partiellement observables

**Variantes** :
1. **Online distillation** : le prof est le modèle tricheur figé, l'élève est mis à jour par PPO + KL
2. **Privileged critic** (lié à P13a) : le prof fournit aussi une estimation de valeur V(s_complet) que l'élève utilise pour ses avantages — signal de retour plus propre
3. **DAgger-style** : collecter des rollouts avec la politique du prof et faire du SFT sur ces trajectoires (voir R2)

**Implémentation** :
```python
# Dans compute_losses(), ajouter :
if teacher_logits is not None:
    kl = F.kl_div(
        F.log_softmax(logits_new, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean"
    )
    total = total + c_distill * kl
```

**Prérequis** : entraîner un cheater jusqu'à convergence (~2000 updates, win rate ~65% vs Random), puis l'utiliser comme prof figé pour l'entraînement de l'élève.

**Fichiers** : `training/losses.py`, `training/self_play.py` (passage des teacher_logits depuis le forward du prof), `run_curriculum_max.py` (config c_distill + cheater_checkpoint).

---

## R2 — Pré-entraînement Supervised Fine-Tuning (SFT) sur replays Showdown

**Objectif** : Initialiser l'agent avec une politique humaine/experte via SFT sur des replays Showdown avant le RL. Cela accélère le bootstrap et donne une prior sensée (pas aléatoire) au début du RL.

**Problème** : Les replays Showdown sont en **information partielle des deux côtés** — les logs `.json` ne révèlent que ce qui a été affiché en jeu (moves utilisés, Pokémon envoyés). L'état complet adverse (nature, EVs, moves non-utilisés) n'est jamais dans le replay.

**Options pour contourner** :

1. **SFT sur états partiels** : Encoder l'état du log tel qu'il est (info partielle), label = action jouée par le joueur humain. Le modèle apprend à imiter les humains dans les mêmes conditions POMDP. Limitation : les humains font des erreurs, et les replays ladder ne sont pas filtrés par niveau.

2. **SFT filtré (replays haut-niveau)** : Télécharger uniquement des replays de joueurs top-ladder (ELO > 1800). La qualité des actions augmente significativement, au prix d'une quantité réduite de données.

3. **SFT + distillation** (R1 + R2 combinés) : Faire tourner le modèle tricheur sur les états initiaux des replays (en recontruisant l'état complet côté simulateur), utiliser sa distribution de policy comme label soft pour le SFT. Contourne le POMDP des replays en utilisant le simulateur pour reconstruire la vérité terrain.

4. **SFT sur trajectoires tricheur** (approche DAgger) : Collecter des rollouts avec le modèle tricheur (R1), utiliser ces trajectoires pour un SFT de l'élève — données en information partielle puisque l'élève ne voit que l'état masqué, mais labels venant de la politique optimale du prof.

**Format des replays Showdown** : `.json` avec `inputLog` (choix du joueur) et `log` (protocole PS complet). L'état peut être reconstruit step-by-step via le simulateur Rust en rejouant les inputs, ce qui donne accès à `get_state()` côté simulateur (information complète).

**Fichiers à créer** : `scripts/parse_replays.py` (parsing + reconstruction d'état), `training/sft.py` (boucle SFT avec BCE sur les actions), nouveau script `run_sft.py`.

---

## P19 — Ajuster le mix d'adversaires (supprimer Random, augmenter Pool)

**Constat** (cheater_v5, ~u80) : L'agent bat déjà RandomPolicy à >90% dès les premiers updates. Garder 10% Random dans le mix n'apporte plus de signal utile — l'agent n'apprend rien de ces parties.

**Proposition** : Passer de `10% rand / 10% FO / 60% EMA / 20% pool` à `0% rand / 10% FO / 60% EMA / 30% pool`. Le FO reste pour ancrer un plancher de difficulté minimal. Le pool à 30% apporte plus de diversité stylistique une fois qu'il se remplit.

**À faire** : Modifier les seuils dans `training/self_play.py` (bloc opponent selection, ~ligne 563) une fois que cheater_v5 a fini ou qu'on relance une nouvelle run.

**Fichiers** : `training/self_play.py`, `run_cheater_v5.py` (ou nouveau launcher).

---

## R3 — Correction du générateur de team (Team Builder)

**Problème** : Le générateur de team actuel ne reproduit pas fidèlement le comportement de Showdown Gen 9 Random Battle. Les équipes générées dévient du format officiel (sets incorrects, distribution d'items/moves erronée, règles de rôle non respectées).

**Impact** : Le simulateur entraîne l'agent sur des équipes irréalistes, ce qui crée un gap train/test : l'agent ne généralise pas bien quand il joue contre de vraies équipes Showdown (lors d'une évaluation en ligne ou en live).

**À investiguer** :
- Comparer les sets générés par le simulateur avec les sets officiels de `random-teams.js` (source Pokémon Showdown)
- Vérifier les règles de rôle (attaquant physique/spécial, tank, rapide, lent-mais-robuste) et leur distribution
- Vérifier la logique d'attribution des items (Choice Band/Specs/Scarf, Life Orb, Leftovers, etc.)
- Vérifier la sélection des EVs/nature (en Random Battle Gen 9, les EVs sont fixes par rôle, pas aléatoires)
- Vérifier la compatibilité move/espèce (learnsets)

**Solution probable** : Réécrire le générateur en se basant directement sur `random-teams.js` de Pokémon Showdown (source officielle) plutôt que sur une approximation. Le fichier de données officiel (`randombattlesets.json`) donne pour chaque espèce les sets autorisés — il suffit de le parser.

**Fichiers** : `simulator/src/lib.rs` (générateur de team côté Rust) ou un générateur Python qui appelle le simulateur Showdown JS directement via `child_process`.