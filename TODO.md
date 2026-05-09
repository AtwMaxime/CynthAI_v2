# CynthAI_v2 — TODO Améliorations

## P1 — POMDP Masking

**Objectif** : Forcer les prédiction heads à fonctionner en masquant progressivement les infos adverses.

**Détail** :
- Dans `encode_state()` dans `rollout.py`, ajouter un paramètre `mask_opponent_ratio: float = 0.0`
- Quand `mask_ratio > 0`, pour chaque attribut opponent (item, ability, moves, tera, species), zero-mask avec probabilité `mask_ratio`
- La reconstruction loss (`pred_logits`) est calculée *sur les valeurs originales non masquées*
- Courbe : mask_ratio = 0 pour les 200 premiers updates, puis linear increase jusqu'à 0.50 à l'update 2000
- Dans l'entraînement, le mask_ratio est calculé à partir de l'update courant

**Fichiers** : `env/state_encoder.py`, `training/rollout.py`, `training/self_play.py`

---

## P2 — Reward Curriculum (Sparsification progressive)

**Objectif** : Éviter que l'agent overfit sur les récompenses proxy (HP adv, KO) et force la priorité sur la victoire.

**Détail** :
- Multiplicateur `dense_scale` qui décroît linéairement de 1.0 → 0.25 sur la durée de l'entraînement
- Ce facteur s'applique à `KO_REWARD`, `OWN_KO_PENALTY`, `HP_ADV_SCALE`
- Le terminal WIN/REWARD (±1.0) n'est jamais affecté
- Courbe : `dense_scale = max(0.25, 1.0 - update / total_updates * 0.75)`
- Paramètres ajustés : `KO_REWARD = 0.05 * dense_scale`, `HP_ADV_SCALE = 0.01 * dense_scale`

**Fichiers** : `training/rollout.py` (dans `compute_step_reward`)

---

## P3 — Pool Size Debug + Opponent Mixing

**Objectif** : Empêcher le sur-apprentissage au pool d'opposants et diversifier les adversaires.

**Détail** :
- **Debug** : Vérifier le seuil de snapshot dans `self_play.py` — actuellement pool_size=5 bloqué. Ajouter un ratio (ex: snapshot quand `win_rate > 0.55` sur les 50 dernières parties) au lieu d'un seuil fixe.
- **Opponent mixing** : À chaque rollout, avec probabilité 20%, remplacer `agent_opp` par une politique externe :
  - 10% `RandomPolicy`
  - 10% random move selection (sélectionne un move légal aléatoirement parmis les 4, mais pas switch — style "weak bot")
- Ces opponents ne sont pas ajoutés au pool, ils servent juste de bruit régularisant

**Fichiers** : `training/self_play.py`, `training/rollout.py`

---

## P4 — Régularisation & Monitoring

**Objectif** : Stabiliser l'entraînement et détecter les dérives.

**Détail** :
- **Weight decay** : `optimizer = torch.optim.AdamW(lr, weight_decay=1e-4)` (remplace Adam)
- **Gradient clipping** : `torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)` avant l'optim step
- **Monitoring** : Logger dans metrics.csv :
  - `grad_norm` : norme totale des gradients
  - `explained_variance` : `1 - var(returns - values) / var(returns)` — qualité du critic
  - `clip_frac` : fraction d'échantillons clippés dans PPO
  - `pool_size` (déjà loggé)
- **Entropy bonus** : Si l'entropie tombe sous un seuil (`min_entropy = -2.8`), augmenter le coefficient dans la loss PPO

**Fichiers** : `training/self_play.py` (optimizer, logging), `model/agent.py` (si besoin)

---

## P5 — Optimisation du Critic (Value Loss)

**Objectif** : Stabiliser V(s) qui oscille fortement.

**Détail** :
- **Coefficient value loss** : Passer de `c_value = 0.5` à `c_value = 1.0` pour donner plus de poids au critic
- **Return normalisation** : Normaliser les returns avant de calculer la value loss :
  ```python
  returns = (returns - returns.mean()) / (returns.std() + 1e-8)
  ```
- **Value clipping** : Optionnellement, ajouter du clipping sur la value loss (similaire au PPO clip) :
  - Clipping la différence entre V(s) et la target (return) à ±1.0 en valeur absolue (Huber-like)

**Fichiers** : `training/self_play.py` (loss computation)

---

## P6 — Learning Rate Schedule

**Objectif** : Meilleur schedule que le linear decay actuel.

**Détail** :
- **Warmup** : 20 premiers updates, LR = `base_lr * (update + 1) / warmup_steps`
- **Cosine decay** : Après warmup, LR = `base_lr * 0.5 * (1 + cos(π * (update - warmup) / (total_updates - warmup)))`
- **Base LR** : Passer de 3e-4 à 2.5e-4 (légère baisse pour stabilité)
- Fin du training : LR ~ 2.5e-4 * 0.5 * (1 + cos(π)) ≈ 0 (decroissance complète)

**Fichiers** : `training/self_play.py`

---

## P7 — Vérification K (à mesurer avant décision)

**Objectif** : Décider si augmenter K=4 → K=6 ou K=8.

**Détail** :
- Lancer `test_attention_maps.py` sur une battle complète
- Analyser les poids d'attention : est-ce que le modèle utilise les tours T-3 et T-4 (les plus vieux)?
- Si les poids d'attention sur les vieux tokens sont proches de zéro → augmenter K ne sert à rien
- Si le modèle distribue son attention sur tout l'historique → K=6 (78 tokens) peut aider

**Fichiers** : `test_attention_maps.py`, `model/backbone.py`

---

## P8 — Opponent Action Query System (futur, après tout le reste)

**Objectif** : Prédire l'action adverse et l'injecter dans la décision.

**Détail** :
- Nouveau module `OpponentActionHead` :
  - Extrait les moves disponibles de l'adversaire depuis l'état
  - Cross-attend queries adverses sur les tokens d'état (symétrique à l'actor head)
  - Loss de prédiction cross-entropy sur l'action adverse réelle
- Injection : concaténer la prédiction d'action adverse aux action_embeds
- À faire seulement après que POMDP masking (P1) est en place, car les deux sont complémentaires

**Fichiers** : Nouveau fichier `model/opponent_head.py`, modifications dans `model/agent.py`, `training/rollout.py`

---

## Résumé de l'ordre d'implémentation

| # | Tâche | Effort | Impact |
|---|-------|--------|--------|
| P1 | POMDP Masking | 1 journée | Élevé |
| P2 | Reward Curriculum | 2h | Moyen |
| P3 | Pool Debug + Opponent Mixing | 4h | Élevé |
| P4 | Régularisation & Monitoring | 2h | Moyen |
| P5 | Optimisation Critic | 1h | Moyen |
| P6 | LR Schedule | 1h | Faible-Moyen |
| P7 | Vérification K | 30 min | Information |
| P8 | Opponent Action Query | 2-3 jours | Futur |