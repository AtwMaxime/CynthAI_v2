use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use pokemon_showdown::battle::TeamFormat;
use pokemon_showdown::dex::Dex;
use pokemon_showdown::{Battle, BattleOptions, PlayerOptions, PRNGSeed, PRNG, ID};

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
    fn make_choices(&mut self, p1_choice: &str, p2_choice: &str) {
        self.inner.make_choices(&[p1_choice, p2_choice]);
        // Don't call py.allow_threads: the underlying Battle type uses
        // Rc<RefCell<...>> which is not Send-safe (pokemon-showdown-rs is
        // single-threaded). The GIL stays held for the full call. This is
        // fine for the current rollout architecture (N parallel Python
        // processes, each with its own GIL).
        self.inner.sent_log_pos = self.inner.log.len();
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

            // Side conditions: { condition_id: layers }
            // layers defaults to 1 for conditions without a layer count (Stealth Rock, etc.)
            let sc = PyDict::new_bound(py);
            for (id, eff) in &side.side_conditions {
                let layers = eff.borrow().layers.unwrap_or(1);
                sc.set_item(id.as_str(), layers)?;
            }
            side_dict.set_item("side_conditions", sc)?;

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
