# Pokémon Token — Design Document

Représentation d'un Pokémon comme token pour le Transformer backbone de CynthAI_v2.

---

## Philosophie générale

### Token vs vecteur plat

L'ancienne architecture (CynthAI_v1) concaténait les 6 Pokémon en un vecteur plat de 984 dims.
Le modèle devait apprendre que "dims 165–328 = Pokémon 2" — bruit structurel inutile.

Dans CynthAI_v2, chaque Pokémon est un **token indépendant** passé au Transformer.
Le Transformer opère par attention entre tokens : il découvre lui-même les relations entre
Pokémon sans qu'on lui impose un ordre.

### Structure d'un token

Un token Pokémon est un mélange de :
- **indices entiers** → passent dans des `nn.Embedding` (espèce, objet, talent, capacités, types)
- **scalaires float** → concaténés directement (HP ratio, stats normalisées, boosts, flags)

La concaténation finale `[embed(espèce) | embed(objet) | ... | scalaires]` est faite dans le modèle,
pas dans l'encodeur. L'encodeur produit un `PokemonFeatures` structuré.

---

## Gestion de l'information cachée (POMDP)

Un combat Pokémon est un POMDP : l'objet, le talent, le type Téra et les capacités adverses
sont inconnus. La solution adoptée est un **belief state implicite par forward pass**.

### Initialisation au tour 0

Avant le premier vrai tour, on effectue un **forward pass d'initialisation** sans prendre d'action :

```
state_0 = encode(visible_info, stats=base_stats, is_predicted=False)
_, predictions_0 = backbone(state_0)   ← pas d'action, seulement les têtes auxiliaires
```

Après pretraining sur 162k sets, les têtes auxiliaires donnent une bonne estimation
depuis l'espèce seule ("je vois un Landorus-T → je prédis Scarf / Intimidate").

### Tours suivants

```
state_t = encode(visible_info, predictions=predictions_{t-1}, is_predicted=True)
action, predictions_t = backbone(state_t)
```

Les prédictions s'améliorent au fil des révélations. Le gradient PPO remonte jusqu'aux
têtes de prédiction via le backbone partagé.

### Pokémon non encore révélés

Token entièrement à zéros + flag `revealed=0`.
Dès qu'un Pokémon est révélé pour la première fois, un forward pass d'init est lancé pour ce slot.

### Tableau récapitulatif

| Pokémon            | Stats actuelles      | Item/Ability/Moves      | is_predicted |
|--------------------|----------------------|-------------------------|--------------|
| Propre, toujours   | stored_stats réelles | valeurs réelles         | 0            |
| Adverse, tour 0    | base_stats (prior)   | prédictions_0 (init FP) | 0            |
| Adverse, tour t    | prédictions_{t-1}    | prédictions_{t-1}       | 1            |
| Adverse, non révélé| zéros                | zéros                   | 0 (revealed=0)|

---

## Index et vocabulaires

Construits depuis les données du simulateur Rust (`pokemon-showdown-rs-master/data/`).
Couvrent le **dex Gen 1-9 complet** — valables pour random battle ET OU (et tout autre format).
Index 0 = `__unk__` pour les cas inconnus / non révélés.

| Table    | Taille | Source                  |
|----------|--------|-------------------------|
| Species  | 1381   | species.json, num 1–1025 |
| Moves    | 686    | moves.json (sans Z/Max) |
| Items    | 250    | items.json              |
| Abilities| 311    | abilities.json          |
| Types    | 19     | 18 types Gen 9 + __unk__|

Les 97 Pokémon CAP (num < 0) sont exclus — ils ne devraient pas apparaître en compétitif standard.
Si le simulateur en génère par bug, ils tombent sur `__unk__` (comportement sûr).

Le fichier `species_index.json` stocke également les **base stats** par espèce,
utilisées pour initialiser la table `nn.Embedding(N_SPECIES, embed_dim)`.

---

## Sections du token Pokémon

### ✅ Section 1 — Identité

| Champ       | Type  | Encodage              | Dims (embed_dim=64) |
|-------------|-------|-----------------------|---------------------|
| species_idx | int   | nn.Embedding          | 64                  |
| level       | float | level / 100 ∈ (0, 1]  | 1                   |

**Pourquoi le level ?** En random battle certains Pokémon sont réduits (Mewtwo lvl 73, etc.).
En OU les niveaux varient. Inclus dès maintenant pour éviter de devoir réentraîner plus tard.

**Initialisation des embeddings espèce :** depuis les base stats + types de chaque espèce,
projetés vers `embed_dim`. Donne un prior géométrique utile (espèces similaires → proches).

---

### ✅ Section 2 — HP

| Champ    | Type  | Encodage    | Dims |
|----------|-------|-------------|------|
| hp_ratio | float | hp / maxhp  | 1    |

`maxhp` exclu : redondant avec `base_stats.hp` + level (section 4).
Le modèle infère la bulkiness depuis les base stats.

---

### ✅ Section 3 — Types

| Champ        | Type | Encodage                          | Dims |
|--------------|------|-----------------------------------|------|
| type1_idx    | int  | nn.Embedding(N_TYPES, d_type)     | d_type |
| type2_idx    | int  | nn.Embedding (0 = __unk__ si mono)| d_type |
| tera_idx     | int  | nn.Embedding                      | d_type |
| terastallized| float| 0/1                               | 1    |

Types courants (après Forme Change, Roost, etc.), pas les types de base.

---

### ✅ Section 4 — Stats

| Champ         | Type   | Encodage                        | Dims |
|---------------|--------|---------------------------------|------|
| base_stats    | float[5]| chaque stat / stat_max_théorique| 5    |
| stats         | float[5]| idem (ou prédiction si adverse) | 5    |
| is_predicted  | float  | 0.0 / 1.0                       | 1    |

Stats : atk / def / spa / spd / spe (HP séparé en section 2).
Valeurs max théoriques pour normalisation : atk=526, def=526, spa=526, spd=526, spe=526.

---

### ✅ Section 5 — Boosts

| Champ   | Type    | Encodage        | Dims |
|---------|---------|-----------------|------|
| boosts  | float[7]| valeur / 6      | 7    |

Ordre : atk, def, spa, spd, spe, accuracy, evasion. Normalisé dans [-1, 1].

---

### ✅ Section 6 — Objet

| Champ    | Type | Encodage          | Dims   |
|----------|------|-------------------|--------|
| item_idx | int  | nn.Embedding(N_ITEMS, d) | d |

`__unk__` si item inconnu (adverse non révélé) ou absent.

---

### ✅ Section 7 — Talent

| Champ       | Type | Encodage               | Dims |
|-------------|------|------------------------|------|
| ability_idx | int  | nn.Embedding(N_ABILITIES, d) | d |

---

### ✅ Section 8 — Statut

| Champ  | Type    | Encodage                              | Dims |
|--------|---------|---------------------------------------|------|
| status | float[7]| one-hot (none/brn/frz/par/psn/tox/slp)| 7    |

---

### ✅ Section 9 — Capacités

4 slots, chacun :

| Champ      | Type  | Encodage                  | Dims   |
|------------|-------|---------------------------|--------|
| move_idx   | int   | nn.Embedding(N_MOVES, d)  | d      |
| pp_ratio   | float | pp / maxpp                | 1      |
| disabled   | float | 0/1                       | 1      |

`__unk__` si capacité non encore révélée (adverse).
PP ratio plutôt que PP bruts — invariant au PP max de la capacité.

---

### ✅ Section 10 — Volatiles

| Champ     | Type     | Encodage              | Dims |
|-----------|----------|-----------------------|------|
| volatiles | float[N] | flags binaires 0/1    | N    |

N = 181 effets — couverture complète de l'enum poke-env + IDs Rust-spécifiques.
Flags binaires (0/1) pour la majorité ; valeurs entières (0-N) pour les effets
à compteur : perishsong (0-3), stockpile (0-3), gmaxchistrike (0-N), fallen (0-5), etc.
Exporté par le simulateur Rust comme dict {id: valeur}.

---

### ✅ Section 11 — Flags de combat

| Champ            | Type  | Encodage | Dims |
|------------------|-------|----------|------|
| is_active        | float | 0/1      | 1    |
| fainted          | float | 0/1      | 1    |
| trapped          | float | 0/1      | 1    |
| force_switch_flag| float | 0/1      | 1    |
| revealed         | float | 0/1      | 1    |

---

## Implémentation

- Index : `CynthAI_v2/data/dicts/` (générés par `data/build_dicts.py`)
- Encodeur : `CynthAI_v2/env/state_encoder.py` — sections 1–11 complètes
- Tables nn.Embedding : `CynthAI_v2/model/embeddings.py` — `PokemonEmbeddings` implémenté
- `N_SCALARS = 222`, `TOKEN_DIM = 438`

### Section 12 — Belief confidence (v2 — inactif en v1)

Trois scalaires réservés pour l'injection du belief state (voir `BELIEF_STATE.md`).
Actuellement à 0.0 (placeholders).

| Index | Champ               | Valeur v1 | Valeur v2              |
|-------|---------------------|-----------|------------------------|
| [222] | item_confidence     | 0.0       | confiance prédicateur  |
| [223] | ability_confidence  | 0.0       | confiance prédicateur  |
| [224] | tera_confidence     | 0.0       | confiance prédicateur  |
