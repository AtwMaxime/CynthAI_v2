# Implémentation de la Térastallisation — Guide complet

Date : 2026-05-15  
Scope : diagnostic + plan d'implémentation pour Gen 9 Random Battle (cheater CynthAI_v2).

---

## TL;DR — Ce qu'il faut faire

| Étape | Fichier Rust à modifier | Effort | Priorité |
|---|---|---|---|
| **A** | `src/battle/run_action.rs` | 30 min | 🔴 Critique — débloque tout |
| **B** | `src/battle_actions/get_damage.rs` | 1h | 🟡 Important |
| **C** | `src/battle_actions/modify_damage.rs` | 1h | 🟡 Important |
| **D** | `src/pokemon/pokemon_struct.rs` + `modify_damage.rs` | 3h | 🟢 Optionnel (Stellar/Terapagos) |

**Référence JS officielle :** `pokemon-showdown/data/scripts.ts` (fonctions `terastallize`, `getDamage`, `getModifiedDamage`) et `pokemon-showdown/sim/pokemon.ts` (méthode `getTypes`).

---

## État actuel — Ce qui fonctionne déjà

Presque toute la logique est codée. Le seul bug est que `terastallize()` n'est **jamais appelée**.

| Composant | Fichier Rust | Statut |
|---|---|---|
| Parsing `move N terastallize` | `src/side/choose.rs` l.192–211 | ✅ OK |
| Enqueue de l'action Terastallize | `src/battle_queue/resolve_action.rs` l.271–300 | ✅ OK |
| Logique complète de `terastallize()` | `src/battle_actions/terastallize.rs` | ✅ OK mais jamais appelée |
| Changement de type (`pokemon.types → [tera_type]`) | dans `terastallize.rs` l.136–155 | ✅ OK mais jamais appelée |
| Émission `\|-terastallize\|` dans le battle log | dans `terastallize.rs` l.123–134 | ✅ OK mais jamais appelée |
| Flag usage unique (`can_terastallize = None`) | dans `terastallize.rs` l.141–146 | ✅ OK mais jamais appelée |
| `get_types()` retourne `[tera_type]` si térastallisé | `src/pokemon/get_types.rs` l.22–28 | ✅ OK |
| `terastallized` exposé dans `get_state()` Python | `simulator/src/lib.rs` l.195 | ✅ OK |
| `canTerastallize` check | `src/battle_actions/can_terastallize.rs` | ✅ OK |
| Tera Blast (BP, type, Stellar self-debuff) | `src/data/move_callbacks/terablast.rs` | ✅ OK |
| Tera Starstorm | `src/data/move_callbacks/terastarstorm.rs` | ✅ OK |
| Tera Shell, Tera Shift, Teraform Zero | `src/data/ability_callbacks/tera*.rs` | ✅ OK |
| `add_type()` bloqué si térastallisé | `src/pokemon/add_type.rs` l.14–18 | ✅ OK |
| **Exécution de Terastallize dans `run_action`** | `src/battle/run_action.rs` | ❌ **BUG PRINCIPAL** |
| BP floor < 60 → 60 BP | `src/battle_actions/get_damage.rs` | ❌ Manquant |
| STAB tera x2 (non-Stellar) | `src/battle_actions/modify_damage.rs` | ❌ Manquant (TODO explicite) |
| Stellar STAB + `stellarBoostedTypes` | `src/battle_actions/modify_damage.rs` + struct | ❌ Manquant |

---

## Étape A — Appeler `terastallize()` dans `run_action` (30 min)

### Cause racine du bug

**Fichier :** `src/battle/run_action.rs`, lignes 954–1076

Le `match poke_action.choice` ne gère que `PokemonActionType::RunSwitch`. Tous les autres types tombent dans `_ => {}` vide :

```rust
match poke_action.choice {
    PokemonActionType::RunSwitch => {
        // ... 100+ lignes ...
    }
    _ => {
        // Other Pokemon actions (mega evo, terastallize, etc.)
        // ← VIDE
    }
}
```

### Ce qu'il faut ajouter

Remplacer `_ => {}` par :

```rust
PokemonActionType::Terastallize => {
    crate::battle_actions::terastallize(
        self,
        (poke_action.side_index, poke_action.pokemon_index),
    );
}
_ => {}
```

### Référence JS

**Fichier JS :** `pokemon-showdown/sim/battle-actions.ts`, méthode `runAction()`

```js
} else if (action.choice === 'terastallize') {
    this.terastallize(action.pokemon);
```

La méthode `terastallize()` elle-même est dans `pokemon-showdown/sim/battle-actions.ts` (chercher `terastallize(pokemon: Pokemon)`). Son port Rust complet est dans `src/battle_actions/terastallize.rs` — il n'y a rien à y modifier.

### Vérification après implémentation

```python
battle = simulator.PyBattle("gen9randombattle", seed)
# jouer "move 1 terastallize"
state = battle.get_state()
assert state["sides"][0]["pokemon"][0]["terastallized"] is not None
```

---

## Étape B — BP floor < 60 → 60 BP (1h)

### Où et quoi

**Fichier Rust à modifier :** `src/battle_actions/get_damage.rs`

Le code JS correspondant est déjà en commentaire dans ce fichier (lignes 80–88). Il faut le porter en Rust et l'insérer **après** la ligne `base_power = base_power.max(1)` (vers la ligne 394 du fichier).

### Référence JS

**Fichier JS :** `pokemon-showdown/sim/battle-actions.ts`, méthode `getDamage()`

```js
const dexMove = this.dex.moves.get(move.id);
if (source.terastallized && (source.terastallized === 'Stellar' ?
    !source.stellarBoostedTypes.includes(move.type) : source.hasType(move.type)) &&
    basePower < 60 && dexMove.priority <= 0 && !dexMove.multihit &&
    // Hard move.basePower check for moves like Dragon Energy that have variable BP
    !((move.basePower === 0 || move.basePower === 150) && move.basePowerCallback)
) {
    basePower = 60;
}
```

### Port Rust à insérer

```rust
// Tera BP floor : STAB moves < 60 BP → boosted to 60
if let Some(ref tera_type) = source_terastallized {
    // hasType() check : does the move type match the active type of the pokemon?
    // For Stellar: applies to types not yet in stellarBoostedTypes (étape D)
    // Pour l'instant on ignore Stellar (cas très rare en Random Battle)
    let move_type_matches = tera_type != "Stellar"
        && battle.pokemon_at(source_pos.0, source_pos.1)
            .map(|p| p.get_types(battle, false, false).contains(&active_move.move_type))
            .unwrap_or(false);

    if move_type_matches && base_power < 60 {
        let dex_move = battle.dex.moves().get(active_move.id.as_str());
        let priority = dex_move.map(|m| m.priority).unwrap_or(0);
        let multihit = dex_move.and_then(|m| m.multihit.as_ref()).is_some();
        // Exclure les moves à BP variable (Dragon Energy : base_power=0 avec callback,
        // Explosion Plasma : base_power=150 avec callback)
        let has_variable_bp = (active_move.base_power == 0 || active_move.base_power == 150)
            && active_move.base_power_callback;
        if priority <= 0 && !multihit && !has_variable_bp {
            base_power = 60;
        }
    }
}
```

**Note :** `source_terastallized` doit être extrait en phase 1 (lecture immutable) avant modification de `base_power`. Suivre le two-phase borrow pattern déjà utilisé dans ce fichier.

**Exemples concrets d'impact :**
- Flammèche (35 BP) + tera Feu → 60 BP
- Pistolet à Eau (40 BP) + tera Eau → 60 BP
- Acide (40 BP) + tera Poison → 60 BP
- Hydraucanon (110 BP) → inchangé (déjà > 60)
- Tranche (70 BP) + tera Eau → inchangé (pas STAB tera)

---

## Étape C — Tera STAB x2 non-Stellar (1h)

### Où et quoi

**Fichier Rust à modifier :** `src/battle_actions/modify_damage.rs`

Il y a un TODO explicite à la ligne 268 :
```rust
// TODO: Handle Terastallized/Stellar cases (pokemon.terastallized)
// For now, skip Stellar tera handling
```

### Référence JS

**Fichier JS :** `pokemon-showdown/sim/battle-actions.ts`, méthode `getModifiedDamage()` (ou `getDamage()` selon la version — chercher `stellarBoostedTypes` dans ce fichier)

```js
let stab = 1;
const isSTAB = move.forceSTAB || pokemon.hasType(type) || pokemon.getTypes(false, true).includes(type);
if (isSTAB) {
    stab = 1.5;
}
if (pokemon.terastallized === 'Stellar') {
    // Cas Stellar — voir Étape D
    if (!pokemon.stellarBoostedTypes.includes(type) || move.stellarBoosted) {
        stab = isSTAB ? 2 : [4915, 4096];
        move.stellarBoosted = true;
        if (pokemon.species.name !== 'Terapagos-Stellar') {
            pokemon.stellarBoostedTypes.push(type);
        }
    }
} else {
    if (pokemon.terastallized === type && pokemon.getTypes(false, true).includes(type)) {
        stab = 2;  // ← C'EST ÇA qu'on implémente ici
    }
    stab = this.battle.runEvent('ModifySTAB', pokemon, target, move, stab);
}
baseDamage = this.battle.modify(baseDamage, stab);
```

### Port Rust à insérer (après `if has_stab { stab = 1.5; }`)

```rust
// Tera STAB x2 : si tera type = move type ET type originel du pokemon
// Ex: Dracaufeu tera Feu utilise Lance-Flammes → 2x au lieu de 1.5x
// (Stellar géré à l'étape D)
let (tera_type_opt, pre_tera_types) = {
    battle.pokemon_at(pokemon_pos.0, pokemon_pos.1)
        .map(|p| (
            p.terastallized.clone(),
            p.get_types(battle, false, true),  // preterastallized=true → types originels
        ))
        .unwrap_or((None, vec![]))
};

if let Some(ref tera_type) = tera_type_opt {
    if tera_type != "Stellar"
        && tera_type == &move_type
        && pre_tera_types.contains(&move_type)
    {
        stab = 2.0;
    }
}
// Note : ModifySTAB event à appeler ici si besoin (abilities comme Adaptability)
```

**Logique :** `pokemon.getTypes(false, true)` avec `preterastallized=true` retourne les types **avant** tera. Si le move type correspond au tera type ET était déjà un type originel, le STAB passe de 1.5x à 2x.

**Exemple :**
- Dracaufeu (Feu/Vol) + tera Feu + Lance-Flammes (Feu) → `pre_tera_types = [Feu, Vol]` → Feu ∈ pre_tera_types → stab = 2.0
- Carabaffe (Eau) + tera Psycho + Hydrocanon (Eau) → tera_type="Psycho" ≠ move_type="Eau" → stab = 1.5 (STAB normal)
- Carabaffe (Eau) + tera Eau + Hydrocanon (Eau) → tera_type="Eau" = move_type="Eau" ET Eau ∈ pre_tera_types → stab = 2.0

---

## Étape D — Stellar STAB + `stellarBoostedTypes` (3h, optionnel)

**Priorité pour Random Battle : faible.** Terapagos-Stellar est ultra-rare.

### Ce qu'il faut ajouter

**1. Nouveau champ sur `Pokemon`** — `src/pokemon/pokemon_struct.rs` :
```rust
pub stellar_boosted_types: Vec<String>,
```
Initialiser à `vec![]` dans `src/pokemon/new.rs` (ligne 183, commentaire JS déjà présent).
Vider dans `src/pokemon/clear_volatile.rs` lors du switch-out.

**2. Nouveau flag sur `ActiveMove`** — `src/battle_queue/active_move.rs` (ou struct équivalente) :
```rust
pub stellar_boosted: bool,  // déjà boosté ce tour par Stellar
```

**3. Logique Stellar dans `modify_damage.rs`** — remplacer le TODO par :
```rust
if let Some(ref tera_type) = tera_type_opt {
    if tera_type == "Stellar" {
        let already_boosted = stellar_boosted_types.contains(&move_type);
        if !already_boosted || move_stellar_boosted {
            stab = if has_stab { 2.0 } else {
                // [4915, 4096] ≈ 1.2x exprimé en fraction
                chain_modify_fraction(stab_val, 4915, 4096)
            };
            // Marquer le type comme boosté (sauf Terapagos-Stellar qui peut re-booster)
            if species_name != "Terapagos-Stellar" {
                stellar_boosted_types.push(move_type.clone());
            }
            active_move.stellar_boosted = true;
        }
    }
}
```

**Référence JS :** même section que l'Étape C dans `pokemon-showdown/sim/battle-actions.ts`.

---

## Fichiers concernés — récapitulatif complet

### Fichiers Rust à modifier

| Fichier | Étape | Ce qu'on change |
|---|---|---|
| `src/battle/run_action.rs` | A | Ajouter arm `Terastallize` dans le match |
| `src/battle_actions/get_damage.rs` | B | Insérer BP floor après `base_power.max(1)` |
| `src/battle_actions/modify_damage.rs` | C+D | Remplacer le TODO par le STAB tera |
| `src/pokemon/pokemon_struct.rs` | D | Ajouter `stellar_boosted_types: Vec<String>` |
| `src/pokemon/new.rs` | D | Init `stellar_boosted_types: vec![]` |
| `src/pokemon/clear_volatile.rs` | D | Vider `stellar_boosted_types` au switch |

### Fichiers Rust à consulter (déjà corrects, ne pas modifier)

| Fichier | Rôle |
|---|---|
| `src/battle_actions/terastallize.rs` | Logique complète de terastallize() — port 1-to-1 du JS, complet |
| `src/pokemon/get_types.rs` | Résolution du type (tera vs pre-tera) — complet |
| `src/battle_queue/resolve_action.rs` | Enqueue de l'action — complet |
| `src/side/choose.rs` | Parsing `move N terastallize` — complet |
| `simulator/src/lib.rs` | Exposition `terastallized` dans `get_state()` Python — complet |
| `src/data/move_callbacks/terablast.rs` | Tera Blast — complet |
| `src/data/ability_callbacks/terashift.rs` | Tera Shift — complet |
| `src/data/ability_callbacks/teraformzero.rs` | Teraform Zero — complet |

### Fichiers JS de référence (pokemon-showdown officiel)

| Fichier JS | Méthode / section | Utilisé pour |
|---|---|---|
| `sim/battle-actions.ts` | `runAction()` | Étape A — où appeler terastallize |
| `sim/battle-actions.ts` | `terastallize(pokemon)` | Référence complète (déjà portée) |
| `sim/battle-actions.ts` | `getDamage()` | Étape B — BP floor |
| `sim/battle-actions.ts` | `getModifiedDamage()` | Étapes C+D — STAB tera |
| `sim/pokemon.ts` | `getTypes(excludeAdded, preterastallized)` | Référence get_types (déjà portée) |
| `data/scripts.ts` | Section terastallization | Vue d'ensemble des hooks |

---

## Ordre d'implémentation recommandé

```
A (run_action.rs — 30 min)
  ↓ débloque : type change, |-terastallize|, get_state(), usage unique
  ↓ tester : p.get("terastallized") retourne le type après "move 1 terastallize"

B (get_damage.rs — 1h)
  ↑ dépend de A (sinon inutile de tester)
  ↓ tester : Flammèche sur tera Feu → 60 BP dans les logs

C (modify_damage.rs — 1h)
  ↑ dépend de A
  ↓ tester : Lance-Flammes Dracaufeu tera Feu → damage × (2/1.5) vs baseline

D (struct + modify_damage.rs — 3h)
  ↑ dépend de A+C
  ↓ tester : Terapagos-Stellar, vérifier stellarBoostedTypes persist entre moves
```

**Total A+B+C : ~2h30 pour un tera fonctionnel à 99% en Gen 9 Random Battle.**
