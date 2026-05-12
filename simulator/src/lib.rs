use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pokemon_showdown::battle::TeamFormat;
use pokemon_showdown::dex::Dex;
use pokemon_showdown::{Battle, BattleOptions, PlayerOptions, PRNGSeed, PRNG, ID, SideID};

#[pyclass(unsendable)]
struct PyBattle {
    inner: Battle,
}

#[pymethods]
impl PyBattle {
    /// Create a new battle.
    /// format: e.g. "gen9randombattle"
    /// seed: integer used to seed both team generation and the battle PRNG
    #[new]
    fn new(format: &str, seed: u32) -> Self {
        let dex = Dex::global();

        // Generate teams with a PRNG seeded from `seed`, matching test_unified.rs behaviour
        let mut prng = PRNG::new(Some(PRNGSeed::Gen5([0, 0, 0, seed])));
        let team1 = pokemon_showdown::team_generator::generate_random_team(&mut prng, &dex);
        let team2 = pokemon_showdown::team_generator::generate_random_team(&mut prng, &dex);

        let battle = Battle::new(BattleOptions {
            format_id: ID::new(format),
            seed: Some(PRNGSeed::Gen5([0, 0, 0, seed])),
            p1: Some(PlayerOptions {
                name: "p1".to_string(),
                team: TeamFormat::Sets(team1),
                avatar: None,
                rating: None,
                seed: None,
            }),
            p2: Some(PlayerOptions {
                name: "p2".to_string(),
                team: TeamFormat::Sets(team2),
                avatar: None,
                rating: None,
                seed: None,
            }),
            ..Default::default()
        });

        PyBattle { inner: battle }
    }

    /// Advance the battle by one turn.
    /// p1_choice / p2_choice: e.g. "move 1", "switch 2", "default"
    /// Returns True if the turn was committed, False if choices were invalid.
    fn make_choices(&mut self, p1_choice: &str, p2_choice: &str) -> bool {
        // Use Battle::choose() (the public API, same as JS battle.ts choose()).
        // It handles: side.choose() validation, is_choice_done() check,
        // and auto-commit when all sides have chosen.
        if !p1_choice.is_empty() {
            if !self.inner.choose(SideID::P1, p1_choice) {
                return false;
            }
        }
        if !p2_choice.is_empty() {
            if !self.inner.choose(SideID::P2, p2_choice) {
                return false;
            }
        }
        true
    }

    /// Try a choice for one side. Returns true if the choice was accepted, false otherwise.
    /// side_idx: 0 for p1, 1 for p2
    /// Never panics — the caller handles invalid choices by trying alternatives.
    fn choose_side(&mut self, side_idx: usize, choice: &str) -> bool {
        let side_id = match side_idx {
            0 => SideID::P1,
            1 => SideID::P2,
            _ => return false,
        };
        self.inner.choose(side_id, choice)
    }

    /// Commit both sides' choices and advance the turn.
    /// Both sides MUST have successfully called choose_side() first.
    /// Panics if choices are incomplete, so only call after both sides return true.
    fn commit_choices(&mut self) {
        self.inner.commit_choices();
    }

    /// Undo the choice for one side, allowing a different choice to be tried.
    fn undo_choice(&mut self, side_idx: usize) {
        let side_id = match side_idx {
            0 => SideID::P1,
            1 => SideID::P2,
            _ => return,
        };
        self.inner.undo_choice(side_id);
    }

    /// Return new log entries since the last call to get_new_log_entries().
    /// Each entry is a line of the Pokémon Showdown protocol (e.g.
    /// "|-ability|p1a: Pikachu|Intimidate|..." or "|-item|p1a: Snorlax|leftovers|...").
    /// The returned list is empty if nothing new happened since the last call.
    fn get_new_log_entries(&mut self) -> Vec<String> {
        let new_entries: Vec<String> = self.inner.log[self.inner.sent_log_pos..]
            .iter()
            .cloned()
            .collect();
        self.inner.sent_log_pos = self.inner.log.len();
        new_entries
    }

    #[getter]
    fn ended(&self) -> bool {
        self.inner.ended
    }

    #[getter]
    fn turn(&self) -> i32 {
        self.inner.turn
    }

    #[getter]
    fn winner(&self) -> Option<String> {
        self.inner.winner.clone()
    }

    /// Return the full observable battle state as a Python dict.
    /// This is the raw data consumed by env/state_encoder.py.
    fn get_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = PyDict::new_bound(py);

        state.set_item("turn", self.inner.turn)?;
        state.set_item("ended", self.inner.ended)?;

        // --- Field ---
        let field = PyDict::new_bound(py);
        field.set_item("weather", self.inner.field.weather.as_str())?;
        field.set_item("terrain", self.inner.field.terrain.as_str())?;
        let pseudo_weather: Vec<&str> = self.inner.field.pseudo_weather
            .keys()
            .map(|k| k.as_str())
            .collect();
        field.set_item("pseudo_weather", pseudo_weather)?;
        state.set_item("field", field)?;

        // --- Sides ---
        let sides_list = PyList::empty_bound(py);
        for side in &self.inner.sides {
            let side_dict = PyDict::new_bound(py);
            side_dict.set_item("pokemon_left", side.pokemon_left)?;
            side_dict.set_item("total_fainted", side.total_fainted)?;
            side_dict.set_item("request_state", format!("{:?}", side.request_state))?;
            side_dict.set_item("choice_error", side.choice.error.as_str())?;

            // Side conditions
            let sc = PyDict::new_bound(py);
            for (id, eff) in &side.side_conditions {
                let layers = eff.borrow().layers.unwrap_or(1);
                sc.set_item(id.as_str(), layers)?;
            }
            side_dict.set_item("side_conditions", sc)?;

            // Slot conditions (e.g. revivalblessing)
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

            // Active Pokemon slot indices (None = empty slot)
            let active: Vec<Option<usize>> = side.active.clone();
            side_dict.set_item("active", active)?;

            // Pokemon
            let poke_list = PyList::empty_bound(py);
            for poke in &side.pokemon {
                let pd = PyDict::new_bound(py);
                pd.set_item("species_id", poke.species_id.as_str())?;
                pd.set_item("name", &poke.name)?;
                pd.set_item("level", poke.level)?;
                pd.set_item("hp", poke.hp)?;
                pd.set_item("maxhp", poke.maxhp)?;
                pd.set_item("is_active", poke.is_active)?;
                pd.set_item("position", poke.position)?;
                pd.set_item("fainted", poke.fainted)?;
                pd.set_item("status", poke.status.as_str())?;
                pd.set_item("types", poke.types.clone())?;
                pd.set_item("tera_type", poke.tera_type.as_deref())?;
                pd.set_item("terastallized", poke.terastallized.as_deref())?;
                pd.set_item("item", poke.item.as_str())?;
                pd.set_item("ability", poke.ability.as_str())?;
                pd.set_item("trapped", poke.trapped.is_trapped())?;
                pd.set_item("force_switch_flag", poke.force_switch_flag)?;

                // Stat boosts (-6 to +6)
                let boosts = PyDict::new_bound(py);
                boosts.set_item("atk", poke.boosts.atk)?;
                boosts.set_item("def", poke.boosts.def)?;
                boosts.set_item("spa", poke.boosts.spa)?;
                boosts.set_item("spd", poke.boosts.spd)?;
                boosts.set_item("spe", poke.boosts.spe)?;
                boosts.set_item("accuracy", poke.boosts.accuracy)?;
                boosts.set_item("evasion", poke.boosts.evasion)?;
                pd.set_item("boosts", boosts)?;

                // Current in-battle stats (atk/def/spa/spd/spe — hp is tracked via hp/maxhp)
                let stats = PyDict::new_bound(py);
                stats.set_item("atk", poke.stored_stats.atk)?;
                stats.set_item("def", poke.stored_stats.def)?;
                stats.set_item("spa", poke.stored_stats.spa)?;
                stats.set_item("spd", poke.stored_stats.spd)?;
                stats.set_item("spe", poke.stored_stats.spe)?;
                pd.set_item("stats", stats)?;

                // Base stats (before any in-battle modifications)
                let base_stats = PyDict::new_bound(py);
                base_stats.set_item("hp", poke.base_stored_stats.hp)?;
                base_stats.set_item("atk", poke.base_stored_stats.atk)?;
                base_stats.set_item("def", poke.base_stored_stats.def)?;
                base_stats.set_item("spa", poke.base_stored_stats.spa)?;
                base_stats.set_item("spd", poke.base_stored_stats.spd)?;
                base_stats.set_item("spe", poke.base_stored_stats.spe)?;
                pd.set_item("base_stats", base_stats)?;

                // Move slots
                let moves = PyList::empty_bound(py);
                for ms in &poke.move_slots {
                    let md = PyDict::new_bound(py);
                    md.set_item("id", ms.id.as_str())?;
                    md.set_item("pp", ms.pp)?;
                    md.set_item("maxpp", ms.maxpp)?;
                    md.set_item("disabled", ms.disabled)?;
                    moves.append(md)?;
                }
                pd.set_item("moves", moves)?;

                // Volatile conditions: {id: value} where value = layers > counter > duration > 1
                let volatiles = PyDict::new_bound(py);
                for (id, eff) in &poke.volatiles {
                    let state = eff.borrow();
                    let value = state.layers
                        .or(state.counter)
                        .or(state.duration)
                        .unwrap_or(1);
                    volatiles.set_item(id.as_str(), value)?;
                }
                pd.set_item("volatiles", volatiles)?;

                poke_list.append(pd)?;
            }
            side_dict.set_item("pokemon", poke_list)?;
            sides_list.append(side_dict)?;
        }
        state.set_item("sides", sides_list)?;

        Ok(state)
    }
}

#[pymodule]
fn simulator(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBattle>()?;
    Ok(())
}
