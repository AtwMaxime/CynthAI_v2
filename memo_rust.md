# Rust Simulator Fixes (pokemon-showdown-rs-master)

Ce document recense les modifications apportées au port Rust de Pokémon Showdown
([`pokemon-showdown-rs-master`](../../pokemon-showdown-rs-master))
pour faire fonctionner CynthAI_v2. Le dépôt upstream n'est plus maintenu,
ces correctifs sont appliqués localement.

## Fix 1: `get_choice_index` — auto-pass conditionnel en mode Switch

**Fichier :** `src/side/get_choice_index.rs`
**Date :** 2026-05-10

**Problème :** En `RequestState::Switch`, le code Rust auto-passe tous les slots
actifs où `switch_flag.is_none()`. En JavaScript, `switchFlag` est toujours défini
quand `requestState == 'switch'`, mais le port Rust ne le définit pas dans tous
les scénarios (post-U-turn, Roar, Dragon Tail, etc.). Résultat : tous les slots
sont auto-passés, `get_choice_index()` retourne `index >= active.len()`, et
`choose_switch()` rejette avec `"You sent more switches than needed"`.

**Fix :** Vérifier si au moins un slot actif a `switch_flag` avant d'activer
l'auto-pass. Si aucun slot n'a le flag, sauter l'auto-pass et laisser
`choose_switch()` gérer l'appel.

```rust
RequestState::Switch => {
    let any_switch_flag = (0..self.active.len()).any(|i| {
        self.active.get(i)
            .and_then(|&a| a)
            .and_then(|idx| self.pokemon.get(idx))
            .map_or(false, |p| p.switch_flag.is_some())
    });
    if any_switch_flag {
        while index < self.active.len() {
            // auto-pass des slots sans switch_flag...
        }
    }
}
```

## Fix 2: `choose_switch` — validation de Pokémon actif

**Fichier :** `src/side/choose_switch.rs`
**Date :** 2026-05-10

**Problème :** La validation `target.position < self.active.len()` pour
déterminer si un Pokémon cible est "actif" est incorrecte en Rust. En JS, les
Pokémon sont physiquement réarrangés dans le tableau `pokemon` pour que les
actifs soient aux indices `0..active.length`. En Rust, on utilise un tableau
d'indices `active: Vec<Option<usize>>` à la place — la position d'un Pokémon
dans son tableau n'indique pas son statut actif.

**Conséquence :** En simples (`active.len() = 1`), un Pokémon de banc avec
`position = 0` (état initial avant tout switch) échoue le test et le switch
est rejeté avec `"Can't switch to an active Pokémon"`, masquant le vrai
problème (Revival Blessing).

**Fix :** Remplacer par `self.active.contains(&Some(slot))`, qui vérifie
l'appartenance au vecteur d'indices actifs.

```rust
// Avant (bug) :
let target_is_active = target.position < self.active.len();

// Après (corrigé) :
let target_is_active = self.active.iter().any(|&a| a == Some(slot));
```

## Fix 3: `lib.rs` — exposition de `slot_conditions` à Python

**Fichier :** `simulator/src/lib.rs` (CynthAI_v2)
**Date :** 2026-05-10

**Problème :** Le `dict` Python retourné par `get_state()` n'incluait pas
`slot_conditions`, une structure côté Rust qui contient les effets attachés
à un slot (comme `revivalblessing`). Sans cette info, Python ne pouvait pas
savoir quand un switch devait cibler un Pokémon K.O. plutôt qu'un vivant.

**Fix :** Ajout de `slot_conditions` au dict `side` dans `get_state()`.

```rust
let slot_cond_dict = PyDict::new_bound(py);
for (pos, conditions) in side.slot_conditions.iter().enumerate() {
    let cond_list = PyList::empty_bound(py);
    for (id, _eff) in conditions {
        cond_list.append(id.as_str())?;
    }
    if !cond_list.is_empty() {
        slot_cond_dict.set_item(pos, cond_list)?;
    }
}
side_dict.set_item("slot_conditions", slot_cond_dict)?;
```

## Dépendance

Le `Cargo.toml` du simulateur pointe vers le dépôt local :

```toml
[dependencies]
pokemon-showdown = { path = "../../pokemon-showdown-rs-master" }
```

Ces modifications sont locales au workspace. Si un jour le dépôt
`pokemon-showdown-rs` est officiellement maintenu, ces correctifs
devront être portés par PR.