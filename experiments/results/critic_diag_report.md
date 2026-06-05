# Critic diagnostic — rapport de synthèse (cheater_v7 / v8)

Date : 2026-06-05. Objectif : réparer le critic sur deux axes — **(R) robustesse**
(blowups de valeur) et **(G) généralisation** (écart EV train vs held-out).

## TL;DR

- Le « blowup » du critic (R² catastrophique dans l'EV exp) n'est **pas** un bug
  d'input ni un état pathologique : c'est une **instabilité d'entraînement
  transitoire localisée à l'update 1000** de cheater_v7, qui s'est **résorbée
  d'elle-même** ensuite.
- Pour **v7 (reward dense)** il n'y a **pas** d'écart train/held-out en agrégat :
  EV train ≈ EV held-out ≈ 0.06. Le critic est honnêtement médiocre.
- Pour **v8 (reward sparse / terminal-only)** il y a un **gros écart optimiste** :
  EV train ~0.4–0.95 vs held-out (calib) r~0.3 → R²~0.09.

## Phase 1-2.5 — Robustesse : nature du blowup

Source : `experiments/diag_critic_blowup.py` (collecteur `--fast`, 320 états /
423 parties, CPU, seed 42), résultats dans `critic_blowup_v7/critic_blowup_diag.json`.

2 états de blowup (|v_theta| > 5) avec la politique d'update 1000 :

| idx | turn | u300 | u600 | **u1000** | u1400 | u1700 |
|----:|-----:|-----:|-----:|----------:|------:|------:|
| 212 | 11   | 0.0  | -0.0 | **-119.6**| 0.0   | 0.1   |
| 213 | 12   | 0.1  | 0.1  | **-179.6**| 0.1   | 0.1   |

**Constats :**
- Le blowup est **checkpoint-local à u1000** : les *mêmes* (state, window) donnent
  ~0.1 à u300/600/1400/1700. Donc **pas** déclenché par l'input.
- Inputs propres : 0 NaN/inf, index d'embedding dans le vocab, 0 padding K-turns,
  `pb_scalars` borné (max 362). → pas de bug d'encodage.
- Hooks par couche (u1000) : la magnitude explose **dès `pokemon_proj`** (max_abs
  ~3865) et se propage (transformer ~3866, value_head.0 ~4209, sortie ~180).
  C'est un **blowup d'échelle des poids du critic**, pas une couche isolée.
- `raw_value` (pré-tanh, lu avec `value_bound=0`) est ~0.1 à u1400/1700 → le critic
  a **réellement guéri** ; le `critic_value_bound` (tanh ±10) ne fait pas que
  masquer — il a probablement *aidé* la guérison en gardant les cibles explosées
  (-180) hors du bootstrap GAE.

→ **Axe R : pas de bug d'input.** Phénomène = instabilité de stabilité/training (axe G).

## Phase 3 — Généralisation : EV train vs held-out

`explained_variance` (training/losses.py:83) est calculé **in-sample** sur le
minibatch tout juste fitté, avec des `returns` GAE **normalisés** (z-scorés).

### v7 (dense) — pas d'écart agrégé

| source | EV |
|---|---|
| train (loggé, mean u300→1700) | ~0.06 |
| held-out (EV exp u1000, nettoyé des 2 outliers) | ~0.05–0.10 |

Notable : à u1000 le `vp_max` **loggé** = 0.69 (sain) alors que le diag trouve
-119/-179 sur des états held-out → faille de **généralisation** invisible dans les
métriques de training (le critic est sain sur sa distribution d'entraînement).

### v8 (sparse) — gros écart optimiste

| source | EV |
|---|---|
| train (loggé) | ~0.4–0.95 |
| held-out (calib plots, proxy) | r~0.3 → R²~0.09 |

Le reward terminal-only gonfle l'EV in-sample (returns bimodaux ±1 très
structurés) alors que la généralisation held-out reste ~0.1, comme v7.

## Reco / suites

1. **EV exp rigoureuse sur v8** (Vπ̂ MC) pour confirmer l'écart au-delà du proxy calib.
2. Garder `critic_value_bound` actif (déjà le cas v8) : utile comme garde-fou
   anti-instabilité, sans masquer une fois le critic sain.
3. Surveiller la généralisation, pas seulement l'EV in-sample : la métrique loggée
   peut être trompeuse (cf v8). Envisager un EV held-out périodique.

## Repro

```bash
# Diag rapide (CPU, machine GPU souvent saturée -> CPU plus rapide en batch=1)
CUDA_VISIBLE_DEVICES="" python experiments/diag_critic_blowup.py --device cpu --fast \
  --ckpt-dir checkpoints/cheater_v7 --collect-update 1000 \
  --updates 300,600,1000,1400,1700 --n-per-bucket 80 \
  --output-dir experiments/results/critic_blowup_v7
# Ré-analyse instantanée (réutilise states_cache.pt) :
python experiments/diag_critic_blowup.py --load-cache ...
```
