# Belief State — Design Document

Pourquoi on ne l'implémente pas maintenant, et exactement ce qu'il faudra faire en v2.

---

## 1. Le problème

Le Pokémon Showdown est un jeu à information imparfaite (POMDP). L'adversaire cache son item, son talent, son type Tera, et les moves non utilisés. Avec une fenêtre glissante de K=4 tours, tout contexte antérieur est perdu.

**Nuance importante :** l'information *révélée* est permanente. Dès qu'un item est utilisé, une ability déclenchée, un move lancé, `state_encoder` l'écrit dans le token du Pokémon adverse et il est présent dans tous les tours suivants du sliding window. Ce qui disparaît après K tours, c'est uniquement l'information *inférée mais jamais confirmée* (item caché d'un Pokémon qui n'est jamais sorti, 4ème move jamais utilisé, type Tera jamais activé).

**Quand ça pose problème :** bataille longue (>10 tours), adversaire jouant sur la surprise, informations critiques jamais révélées directement.

---

## 2. Pourquoi on ne l'implémente pas en v1

### Problème d'amorçage
En début de training, le predictor génère des prédictions aléatoires. Réinjecter ces prédictions pollue le token adverse avec du bruit pendant les 10 000+ premiers steps. Le modèle apprend à ignorer le canal de belief, ce qui annule l'utilité du système.

Pour que l'injection aide, il faut que le predictor soit déjà bon. Pour que le predictor soit bon, il faut que le backbone ait appris à utiliser l'injection. Dépendance circulaire.

### Complexité de training
Faire converger PPO en self-play est déjà difficile. Ajouter une boucle de feedback latente multiplie les sources d'instabilité et rend le debugging opaque.

### Le Transformer compense partiellement
Les `opp_tokens` enrichis par le backbone encodent déjà implicitement les croyances du modèle sur l'adversaire. Le predictor comme tête auxiliaire pure force ces représentations internes à être de meilleure qualité sans nécessiter d'injection.

---

## 3. Design cible — v2

### Principe
À chaque tour t, le Predictor génère des prédictions pour les champs non révélés des Pokémon adverses. Ces prédictions sont stockées dans un buffer et réinjectées au tour t+1 comme signal de croyance. Le Transformer sait que c'est une croyance (pas un fait) grâce au scalaire de confiance.

### Chaîne d'exécution
```
Tour t
  └─ backbone → opp_tokens [B, 6, D_MODEL]
  └─ predictor → item_logits [B, 6, 250], ability_logits [B, 6, 311], ...
  └─ stocker dans belief_buffer :
       item_pred_idx[b, p]        = argmax(item_logits[b, p])       int
       item_pred_conf[b, p]       = max(softmax(item_logits[b, p])) float ∈ [0,1]
       ability_pred_idx[b, p]     = argmax(ability_logits[b, p])
       ability_pred_conf[b, p]    = max(softmax(ability_logits[b, p]))
       tera_pred_idx[b, p]        = argmax(tera_logits[b, p])
       tera_pred_conf[b, p]       = max(softmax(tera_logits[b, p]))

Tour t+1
  └─ state_encoder reçoit belief_buffer
  └─ Pour chaque Pokémon adverse non révélé sur ce champ :
       poke.item_idx         = item_pred_idx          (si conf > seuil, sinon UNK)
       poke.item_confidence  = item_pred_conf          float [0,1] injecté en scalaire
       poke.ability_idx      = ability_pred_idx
       poke.ability_confidence = ability_pred_conf
       poke.tera_idx         = tera_pred_idx
       poke.tera_confidence  = tera_pred_conf
  └─ Si revealed == 1 pour ce champ → on ignore le buffer, on met confidence = 1.0
```

### Règle absolue : stop_gradient au point d'injection
```python
# Dans ShowdownEnv, avant d'injecter dans state_encoder :
injected_item_idx = belief_buffer.item_pred_idx.detach()   # pas de gradient
```
Sans `detach`, les gradients remontent à travers les prédictions de t pour mettre à jour les poids qui ont produit ces prédictions, créant un RNN non déclaré. La perte de gradient dans la chaîne bayésienne est intentionnelle : le predictor apprend via l'auxiliary loss, pas via la policy loss.

### Moves adverses : NON injectés
4 moves × 686 classes × 6 Pokémon = trop de dimensions pour compresser proprement. Les moves sont inférés implicitement par le Transformer via les patterns d'attaque observés. Uniquement item, ability, tera sont injectés.

### Scalaires ajoutés à PokemonFeatures (v2 uniquement)
```
item_confidence     float  # confiance de la prédiction item  [0, 1]    (0 si révélé = valeur exacte)
ability_confidence  float  # confiance de la prédiction ability [0, 1]
tera_confidence     float  # confiance de la prédiction tera  [0, 1]
```
Ces 3 scalaires remplacent les anciens flags binaires `item_predicted / ability_predicted / tera_predicted`.
Le backbone apprend à interpréter "confiance 0.9 sur Leftovers" vs "confiance 0.3 sur Choice Scarf" comme une information douce.

### N_SCALARS en v2
```
N_SCALARS_V2 = N_SCALARS_V1 + 3   # +3 scalaires de confiance (item, ability, tera)
# moves_predicted supprimé définitivement
```

---

## 4. Ce qui change dans le code (v2)

### `env/state_encoder.py`
- Ajouter `item_confidence`, `ability_confidence`, `tera_confidence` à `PokemonFeatures` (Section 12)
- `encode_pokemon()` reçoit un `belief: dict | None` optionnel en paramètre
- Si `belief` fourni et `revealed == 0` pour ce champ → injecter index + confidence
- Si `revealed == 1` → confidence = 1.0, index = valeur réelle (pas de changement)

### `model/embeddings.py`
- Ajouter 3 scalaires dans le layout : `[222]` item_confidence, `[223]` ability_confidence, `[224]` tera_confidence
- `N_SCALARS` : 222 → 225
- Mettre à jour `collate_features()`

### `env/showdown_env.py`
- Maintenir un `belief_buffer` par partie (dict indexed par Pokémon ID)
- Après chaque forward pass, appeler `predictor.update_buffer(opp_tokens, belief_buffer)`
- Passer `belief_buffer` à `state_encoder` au tour suivant

### `model/prediction_heads.py`
- `PredictorHeads.forward(opp_tokens) → dict[str, Tensor]` (logits par champ)
- Méthode `update_buffer(opp_tokens, buffer)` : applique argmax + softmax confidence, detach, stocke

---

## 5. Conditions pour activer l'injection

Ne brancher le belief state qu'après :
1. Le predictor atteint une accuracy > 40% sur item et ability en eval (prédictions meilleures qu'un prior uniforme)
2. Le backbone de base (PPO sans injection) a convergé sur des matchs simples
3. Introduire l'injection progressivement : commencer avec `seuil_confiance = 0.8` (injection uniquement quand très sûr), baisser progressivement

---

## 6. Ce qu'on N'implémente PAS

- **Prédiction d'espèce** : révélée dès l'entrée du Pokémon (Team Preview ou premier combat). Cas Zoroark géré implicitement par le Transformer via les types et dégâts observés.
- **Injection des logits bruts** : trop volumineuse (+3500 dims). On passe uniquement argmax + confidence scalaire.
- **BPTT à travers le belief** : interdit par le `detach`. Le predictor apprend via auxiliary loss uniquement.
