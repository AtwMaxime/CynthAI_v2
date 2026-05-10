# CynthAI_v2 — TODO

## Déjà implémenté (vérifié dans le code)

- [P1] POMDP Masking — `compute_mask_ratio()` + `apply_reveal_mask()` avec curriculum (warmup=200 → max_ratio=0.5)
- [P2] Reward Curriculum — `compute_dense_scale()` avec decay linéaire 1.0 → 0.25
- [P3] Pool Debug + Opponent Mixing — pool_size=20, snapshot_threshold=0.55, mixing 80% pool / 10% Random / 10% FullOffense
- [P4] Régularisation & Monitoring — AdamW (wd=1e-4), clip_grad_norm=0.5, grad_norm/explained_variance/clip_frac loggés
- [P5] Critic — c_value=1.0, return normalization dans `losses.py`
- [P6] LR Schedule — warmup=20 + cosine decay, base_lr=2.5e-4

---

## P7 — Vérification K (analyse attention)

**Objectif** : Décider si augmenter K=4 → K=6 ou K=8.

**Détail** :
- Lancer `test_attention_maps.py` sur une battle complète
- Analyser les poids d'attention : est-ce que le modèle utilise les tours T-3 et T-4 (les plus vieux)?
- Si les poids d'attention sur les vieux tokens sont proches de zéro → augmenter K ne sert à rien
- Si le modèle distribue son attention sur tout l'historique → K=6 (78 tokens) peut aider

**Fichiers** : `tests/test_attention_maps.py`, `model/backbone.py`

---

## P8 — Opponent Action Query System (futur)

**Objectif** : Prédire l'action adverse et l'injecter dans la décision.

**Détail** :
- Nouveau module `OpponentActionHead` :
  - Extrait les moves disponibles de l'adversaire depuis l'état
  - Cross-attend queries adverses sur les tokens d'état (symétrique à l'actor head)
  - Loss de prédiction cross-entropy sur l'action adverse réelle
- Injection : concaténer la prédiction d'action adverse aux action_embeds
- À faire seulement après avoir validé que POMDP masking (P1) fonctionne bien, car les deux sont complémentaires

**Fichiers** : Nouveau fichier `model/opponent_head.py`, modifications dans `model/agent.py`, `training/rollout.py`

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

En pratique, avec K=4 et le curriculum P1 (mask_ratio qui monte jusqu'à 0.5), l'impact est limité : le modèle apprend à ne pas trop compter sur les attributs non-révélés. Mais l'approximation idéale serait de stocker un reveal state par turn dans le `BattleWindow` et de les appliquer individuellement.

**Solution si nécessaire un jour** : stocker `reveal_state` à chaque push dans `BattleWindow`, et modifier `_build_sequence()` pour que `apply_reveal_mask()` reçoive un masque par turn au lieu d'un masque global. Pas prioritaire tant que les performances n'indiquent pas un plafond lié à ce comportement.