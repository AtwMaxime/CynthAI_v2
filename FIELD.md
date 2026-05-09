# Field Token — Design Document

Représentation de l'état global du terrain comme token pour le Transformer backbone de CynthAI_v2.

---

## Philosophie

Le combat Pokémon n'est pas seulement une question de Pokémon — le terrain, la météo et les conditions de camp influencent directement les calculs de dégâts, les immunités, les entrées sur le terrain. Ces informations sont encodées dans un **token unique de 72 dims** passé au Transformer aux côtés des 12 tokens Pokémon.

Le backbone reçoit donc **13 tokens** :
- 12 tokens Pokémon `[B, 12, 438]` (voir `POKEMON_VECTOR.md`)
- 1 token field `[B, 1, 72]` (ce document)

---

## Implémentation

- Encodeur : `CynthAI_v2/env/state_encoder.py` — `encode_field(state)` → `FieldFeatures`
- Batching : `CynthAI_v2/model/embeddings.py` — `collate_field_features(batch)` → `FieldBatch`
- Source : `PyBattle.get_state()["field"]` + `["sides"]`

---

## Structure du token field

**FIELD_DIM = 72**

```
[0  : 9 ]  Weather       one-hot  (9 valeurs)
[9  : 14]  Terrain       one-hot  (5 valeurs)
[14 : 22]  Pseudo-weather flags   (8 flags binaires)
[22 : 45]  Side 0 conditions      (23 valeurs normalisées)
[45]       Side 0 pokemon_left / 6
[46]       Side 0 total_fainted / 6
[47 : 70]  Side 1 conditions      (23 valeurs normalisées)
[70]       Side 1 pokemon_left / 6
[71]       Side 1 total_fainted / 6
```

---

## Section 1 — Weather `[0:9]`

One-hot sur 9 valeurs. Source : `state["field"]["weather"]` (string Rust ID).

| Index | ID Rust | Signification |
|-------|---------|---------------|
| 0 | `""` | Aucune météo |
| 1 | `"sunnyday"` | Soleil — boost Feu, affaiblit Eau, active Chlorophylle/Chaleur |
| 2 | `"raindance"` | Pluie — boost Eau, affaiblit Feu, active Hydraucité |
| 3 | `"sandstorm"` | Tempête de sable — dégâts chip, boost DéfSpé Roche |
| 4 | `"hail"` | Grêle Gen 1–8 — dégâts chip, active Voile Neige |
| 5 | `"snowscape"` | Neige Gen 9 — remplace grêle, +Déf physique types Glace |
| 6 | `"primordialsea"` | Pluie diluvienne (Kyogre) — pluie permanente, annule Feu |
| 7 | `"desolateland"` | Chaleur extrême (Groudon) — soleil permanent, annule Eau |
| 8 | `"deltastream"` | Vents violents (Rayquaza) — annule faiblesses type Vol |

---

## Section 2 — Terrain `[9:14]`

One-hot sur 5 valeurs. Source : `state["field"]["terrain"]`.

| Index | ID Rust | Signification |
|-------|---------|---------------|
| 0 | `""` | Aucun terrain |
| 1 | `"electricterrain"` | Terrain Électrique — +30% Élec, empêche Sommeil au sol |
| 2 | `"grassyterrain"` | Terrain Herbeux — +30% Herbe, régénère HP, réduit Séisme |
| 3 | `"mistyterrain"` | Terrain Brumeux — immunité statuts au sol, réduit Dragon |
| 4 | `"psychicterrain"` | Terrain Psychique — +30% Psy, annule priorité au sol |

---

## Section 3 — Pseudo-weather `[14:22]`

8 flags binaires (0/1). Source : `state["field"]["pseudo_weather"]` (liste de strings actifs).

| Index | ID Rust | Signification |
|-------|---------|---------------|
| 0 | `"gravity"` | Gravité — cloue tout au sol, annule immunités Vol/Lévitation |
| 1 | `"magicroom"` | Chambre Magique — neutralise tous les objets tenus |
| 2 | `"trickroom"` | Monde Inverse — **inverse l'ordre de vitesse** (lent en premier) |
| 3 | `"wonderroom"` | Pièce Étrange — échange Déf et DéfSpé pour tous |
| 4 | `"echoedvoice"` | Voix d'Écho — la puissance augmente à chaque utilisation consécutive |
| 5 | `"mudsport"` | Boue Sport — moves Électrik affaiblis de 67% |
| 6 | `"watersport"` | Aqua Sport — moves Feu affaiblis de 67% |
| 7 | `"fairylock"` | Verrou Fée — empêche tout changement de Pokémon au prochain tour |

---

## Section 4 — Side conditions `[22:70]`

23 valeurs normalisées par side (layers / max_layers ∈ [0, 1]).
Side 0 : `[22:47]`, Side 1 : `[47:72]`.
Source : `state["sides"][i]["side_conditions"]` (dict `{id: layers}`).

| Offset | ID Rust | Signification | Max couches |
|--------|---------|---------------|-------------|
| +0 | `"stealthrock"` | Roc Stellaire — dégâts typés à chaque entrée | 1 |
| +1 | `"spikes"` | Pics — 1/8 / 1/6 / 1/4 HP à l'entrée (non-Vol) | 3 |
| +2 | `"toxicspikes"` | Pics Toxik — empoisonne à l'entrée | 2 |
| +3 | `"stickyweb"` | Toile Gluante — −1 SPE à l'entrée | 1 |
| +4 | `"reflect"` | Mur Miroir — réduit dégâts physiques de 50% | 1 |
| +5 | `"lightscreen"` | Écran Lumineux — réduit dégâts spéciaux de 50% | 1 |
| +6 | `"auroraveil"` | Voile Aurore — réduit les deux (requiert neige active) | 1 |
| +7 | `"tailwind"` | Vent Arrière — double la SPE pendant 4 tours | 1 |
| +8 | `"mist"` | Brume — immunité aux baisses de stats | 1 |
| +9 | `"safeguard"` | Rune Protect — immunité aux statuts | 1 |
| +10 | `"luckychant"` | Porte-Bonheur — immunité aux coups critiques | 1 |
| +11 | `"craftyshield"` | Bouclier Malin — bloque les moves de statut ce tour | 1 |
| +12 | `"matblock"` | Blocage — bloque les moves de contact ce tour | 1 |
| +13 | `"quickguard"` | Garde Rapide — bloque les moves à priorité+ ce tour | 1 |
| +14 | `"wideguard"` | Large Garde — bloque les moves multi-cibles ce tour | 1 |
| +15 | `"gmaxsteelsurge"` | G-Max Acier — danger d'entrée type Acier | 1 |
| +16 | `"gmaxcannonade"` | G-Max Boulet — dégâts sur 4 tours (non-Eau) | 1 |
| +17 | `"gmaxvolcalith"` | G-Max Roc — dégâts sur 4 tours (non-Roche) | 1 |
| +18 | `"gmaxvinelash"` | G-Max Lianes — dégâts sur 4 tours (non-Herbe) | 1 |
| +19 | `"gmaxwildfire"` | G-Max Feu — dégâts sur 4 tours (non-Feu) | 1 |
| +20 | `"grasspledge"` | Serment Herbeux — combo (marécage ou arc-en-ciel) | 1 |
| +21 | `"waterpledge"` | Serment Aquatique — combo | 1 |
| +22 | `"firepledge"` | Serment Brûlant — combo | 1 |

---

## Section 5 — Team stats `[45:47]` et `[70:72]`

| Offset | Champ | Encodage |
|--------|-------|----------|
| +23 | `pokemon_left` | `pokemon_left / 6` ∈ [0, 1] |
| +24 | `total_fainted` | `total_fainted / 6` ∈ [0, 1] |

---

## Ce qui n'est PAS dans ce token

- **Actions disponibles** — géré par `env/action_space.py` (à implémenter)
- **Volatiles par Pokémon** — dans les tokens Pokémon Section 10 (181 flags)
- **Statuts individuels** — dans les tokens Pokémon Section 8
- **Force switch / trapped** — dans les tokens Pokémon Section 11
