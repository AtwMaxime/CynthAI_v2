"""
CynthAI_v2 State Encoder — converts PyBattle.get_state() dicts into feature vectors.

Exports:
  PokemonFeatures, encode_pokemon
  SideFeatures, FieldFeatures, encode_side, encode_field
  All vocabulary sizes (N_SPECIES, N_MOVES, N_ITEMS, N_ABILITIES, N_TYPES)
  Embedding priors (TYPE_EMBEDDING_PRIOR, MOVE_EMBEDDING_PRIOR)
  Volatiles catalogue (VOLATILE_FLAGS, N_VOLATILES)
  Field constants (WEATHER_LIST, TERRAIN_LIST, PSEUDO_WEATHER_FLAGS, SIDE_CONDITIONS)
  UNK = 0  — unknown / padding index for all categorical fields
"""

import json
import torch
from dataclasses import dataclass, field
from pathlib import Path

# ── Data directory ─────────────────────────────────────────────────────────────

DICTS = Path(__file__).parent.parent / "data" / "dicts"


def _load(name: str) -> dict:
    with open(DICTS / name, encoding="utf-8") as f:
        return json.load(f)


# ── Raw data ───────────────────────────────────────────────────────────────────

_species_data  = _load("species_index.json")
_move_data     = _load("move_index.json")
_item_data     = _load("item_index.json")
_ability_data  = _load("ability_index.json")
_type_data     = _load("type_index.json")

SPECIES_INDEX: dict[str, int] = _species_data["index"]
MOVE_INDEX:    dict[str, int] = _move_data["index"]
ITEM_INDEX:    dict[str, int] = _item_data["index"]
ABILITY_INDEX: dict[str, int] = _ability_data["index"]
TYPE_INDEX:    dict[str, int] = _type_data["index"]

N_SPECIES:   int = _species_data["size"]   # 1381
N_MOVES:     int = _move_data["size"]      # 686
N_ITEMS:     int = _item_data["size"]      # 250
N_ABILITIES: int = _ability_data["size"]   # 311
N_TYPES:     int = _type_data["size"]      # 19

# ── Embedding priors ───────────────────────────────────────────────────────────

_type_emb_data     = _load("type_embeddings.json")
TYPE_EMBEDDING_PRIOR = torch.tensor(_type_emb_data["embeddings"], dtype=torch.float32)
D_TYPE: int          = _type_emb_data["d_type"]   # 8

_move_emb_data       = _load("move_embeddings.json")
MOVE_EMBEDDING_PRIOR = torch.tensor(_move_emb_data["embeddings"], dtype=torch.float32)
D_MOVE_PRIOR: int    = _move_emb_data["d_move"]   # 32

# ── Unknown index ──────────────────────────────────────────────────────────────

UNK: int = 0   # all index files reserve slot 0 for __unk__

# ── Volatile conditions (Gen 9) ────────────────────────────────────────────────

VOLATILE_FLAGS: tuple[str, ...] = (
    'afteryou', 'aquaring', 'aromaveil', 'attract', 'autotomize', 'baddreams',
    'banefulbunker', 'battlebond', 'beakblast', 'bide', 'burnup', 'charge',
    'chillyreception', 'choicelock', 'commanded', 'commanding', 'confusion',
    'counter', 'craftyshield', 'cudchew', 'curse', 'custapberry', 'dancer',
    'destinybond', 'disable', 'disguise', 'doomdesire', 'dynamax', 'eeriespell',
    'embargo', 'emergencyexit', 'encore', 'endure', 'fairylock', 'fallen',
    'feint', 'ficklebeam', 'flashfire', 'flowerveil', 'focusband', 'focusenergy',
    'focuspunch', 'foresight', 'forewarn', 'futuresight', 'gastroacid', 'gem',
    'gmaxcentiferno', 'gmaxchistrike', 'gmaxoneblow', 'gmaxrapidflow',
    'gmaxsandblast', 'grudge', 'guardsplit', 'gulpmissile', 'hadronengine',
    'healblock', 'healer', 'hydration', 'hyperspacefury', 'hyperspacehole',
    'iceball', 'iceface', 'illusion', 'immunity', 'imprison', 'ingrain',
    'innardsout', 'insomnia', 'ironbarbs', 'laserfocus', 'leechseed',
    'leppaberry', 'lightningrod', 'limber', 'liquidooze', 'lockedmove',
    'lockon', 'magnetrise', 'matblock', 'maxguard', 'mefirst', 'mimic',
    'mimicry', 'mindreader', 'minimize', 'miracleeye', 'mirrorcoat', 'mummy',
    'mustrecharge', 'neutralizinggas', 'nightmare', 'noretreat', 'oblivious',
    'octolock', 'orichalcumpulse', 'owntempo', 'partiallytrapped', 'pastelveil',
    'perishsong', 'phantomforce', 'poltergeist', 'powder', 'powerconstruct',
    'powersplit', 'powertrick', 'protect', 'protectivepads', 'protosynthesis',
    'protosynthesisatk', 'protosynthesisdef', 'protosynthesisspa',
    'protosynthesisspd', 'protosynthesisspe', 'pursuit', 'quarkdrive',
    'quarkdriveatk', 'quarkdrivedef', 'quarkdrivespa', 'quarkdrivespd',
    'quarkdrivespe', 'quash', 'quickclaw', 'quickguard', 'ripen', 'rollout',
    'rolloutstorage', 'roughskin', 'safetygoggles', 'saltcure', 'screencleaner',
    'shadowforce', 'shedskin', 'shelltrap', 'sketch', 'skillswap', 'skydrop',
    'slowstart', 'smackdown', 'snaptrap', 'snatch', 'speedswap', 'spite',
    'stall', 'stickyhold', 'stockpile', 'stormdrain', 'substitute', 'suctioncups',
    'supremeoverlord', 'sweetveil', 'symbiosis', 'synchronize', 'syrupbomb',
    'tarshot', 'taunt', 'telekinesis', 'telepathy', 'terashell', 'terashift',
    'thermalexchange', 'throatchop', 'tidyup', 'torment', 'toxicdebris',
    'trapped', 'trapper', 'trick', 'truant', 'twoturnmove', 'unburden',
    'uproar', 'vitalspirit', 'wanderingspirit', 'waterbubble', 'waterveil',
    'wideguard', 'wimpout', 'yawn', 'zenmode', 'zerotohero',
)

_VOLATILE_SET: set[str]      = set(VOLATILE_FLAGS)
N_VOLATILES:   int           = len(VOLATILE_FLAGS)
_VOLATILE_IDX: dict[str, int] = {v: i for i, v in enumerate(VOLATILE_FLAGS)}

# ── Pokemon features dataclass ─────────────────────────────────────────────────

@dataclass
class PokemonFeatures:
    """
    Structured representation of one Pokémon token.
    Indices go through name→index lookup; scalars are floats in [0,1] where possible.
    """
    species_idx:       int   = UNK
    level:             float = 0.0
    hp_ratio:          float = 0.0
    type1_idx:         int   = UNK
    type2_idx:         int   = UNK
    tera_idx:          int   = UNK
    terastallized:     float = 0.0
    base_stats:        list  = field(default_factory=lambda: [0.0] * 5)
    stats:             list  = field(default_factory=lambda: [0.0] * 5)
    is_predicted:      float = 0.0
    boosts:            list  = field(default_factory=lambda: [0.0] * 7)
    item_idx:          int   = UNK
    ability_idx:       int   = UNK
    status:            list  = field(default_factory=lambda: [0.0] * 7)
    move_indices:      list  = field(default_factory=lambda: [UNK]  * 4)
    move_pp:           list  = field(default_factory=lambda: [0.0]  * 4)
    move_disabled:     list  = field(default_factory=lambda: [0.0]  * 4)
    volatiles:         list  = field(default_factory=lambda: [0.0]  * N_VOLATILES)
    is_active:         float = 0.0
    fainted:           float = 0.0
    trapped:           float = 0.0
    force_switch_flag: float = 0.0
    revealed:          float = 0.0


# ── Pokemon encoder ────────────────────────────────────────────────────────────

def encode_pokemon(poke: dict) -> PokemonFeatures:
    """
    Encode a single Pokémon dict (from PyBattle.get_state()) into PokemonFeatures.
    """
    features = PokemonFeatures()

    features.species_idx = SPECIES_INDEX.get(poke.get("species_id", ""), UNK)
    features.level       = poke.get("level", 0) / 100.0

    maxhp = poke.get("maxhp", 0)
    features.hp_ratio = poke.get("hp", 0.0) / (maxhp or 1)

    types = poke.get("types", [])
    t1    = types[0] if types else ""
    features.type1_idx = TYPE_INDEX.get(t1.lower(), UNK)
    features.type2_idx = TYPE_INDEX.get(types[1].lower(), UNK) if len(types) > 1 else UNK

    tera = poke.get("tera_type", "")
    features.tera_idx     = TYPE_INDEX.get(tera.lower(), UNK) if tera else UNK
    features.terastallized = 1.0 if poke.get("terastallized") else 0.0

    _STAT_KEYS = ("atk", "def", "spa", "spd", "spe")
    bs = poke.get("base_stats", {})
    features.base_stats = [bs.get(k, 0) for k in _STAT_KEYS]

    st = poke.get("stats", {})
    features.stats = [st.get(k, 0) for k in _STAT_KEYS]

    features.is_predicted = float(poke.get("is_predicted", False))

    _BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
    b = poke.get("boosts", {})
    features.boosts = [b.get(k, 0) for k in _BOOST_KEYS]

    item    = poke.get("item", "")
    ability = poke.get("ability", "")
    features.item_idx    = ITEM_INDEX.get(item, UNK)
    features.ability_idx = ABILITY_INDEX.get(ability, UNK)

    _STATUS_ORDER = ("", "brn", "frz", "par", "psn", "tox", "slp")
    status = poke.get("status", "")
    idx    = _STATUS_ORDER.index(status) if status in _STATUS_ORDER else 0
    features.status = [1.0 if i == idx else 0.0 for i in range(7)]

    for slot, mv in enumerate((poke.get("moves") or [])[:4]):
        if mv is None:
            continue
        features.move_indices[slot] = MOVE_INDEX.get(mv.get("id", ""), UNK)
        maxpp = mv.get("maxpp", 1) or 1
        features.move_pp[slot]       = mv.get("pp", 0) / maxpp
        features.move_disabled[slot] = float(mv.get("disabled", False))

    active = poke.get("volatiles", {})
    features.volatiles = [float(active.get(v, 0)) for v in VOLATILE_FLAGS]

    features.is_active         = float(poke.get("is_active", False))
    features.fainted           = float(poke.get("fainted", False))
    features.trapped           = float(poke.get("trapped", False))
    features.force_switch_flag = float(poke.get("force_switch_flag", False))
    features.revealed          = float(poke.get("revealed", True))

    return features


# ── Field constants ────────────────────────────────────────────────────────────

WEATHER_LIST: tuple[str, ...] = (
    "", "sunnyday", "raindance", "sandstorm", "hail",
    "snowscape", "primordialsea", "desolateland", "deltastream",
)
WEATHER_INDEX: dict[str, int] = {w: i for i, w in enumerate(WEATHER_LIST)}
N_WEATHER: int = len(WEATHER_LIST)   # 9

TERRAIN_LIST: tuple[str, ...] = (
    "", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain",
)
TERRAIN_INDEX: dict[str, int] = {t: i for i, t in enumerate(TERRAIN_LIST)}
N_TERRAIN: int = len(TERRAIN_LIST)   # 5

PSEUDO_WEATHER_FLAGS: tuple[str, ...] = (
    "gravity", "magicroom", "trickroom", "wonderroom",
    "echoedvoice", "mudsport", "watersport", "fairylock",
)
N_PSEUDO: int = len(PSEUDO_WEATHER_FLAGS)   # 8

SIDE_CONDITIONS: tuple[tuple[str, int], ...] = (
    ("stealthrock",    1),
    ("spikes",         3),
    ("toxicspikes",    2),
    ("stickyweb",      1),
    ("reflect",        1),
    ("lightscreen",    1),
    ("auroraveil",     1),
    ("tailwind",       1),
    ("mist",           1),
    ("safeguard",      1),
    ("luckychant",     1),
    ("craftyshield",   1),
    ("matblock",       1),
    ("quickguard",     1),
    ("wideguard",      1),
    ("gmaxsteelsurge", 1),
    ("gmaxcannonade",  1),
    ("gmaxvolcalith",  1),
    ("gmaxvinelash",   1),
    ("gmaxwildfire",   1),
    ("grasspledge",    1),
    ("waterpledge",    1),
    ("firepledge",     1),
)
N_SIDE_CONDITIONS: int          = len(SIDE_CONDITIONS)   # 23
_SIDE_COND_INDEX: dict[str, int] = {k: i for i, (k, _) in enumerate(SIDE_CONDITIONS)}
_SIDE_COND_MAX:   dict[str, int] = {k: m for k, m in SIDE_CONDITIONS}

SIDE_DIM:  int = N_SIDE_CONDITIONS + 2   # 25  (conditions + pokemon_left + total_fainted)
FIELD_DIM: int = N_WEATHER + N_TERRAIN + N_PSEUDO + 2 * SIDE_DIM   # 72


# ── Side / Field features ──────────────────────────────────────────────────────

@dataclass
class SideFeatures:
    conditions:    list  = field(default_factory=lambda: [0.0] * N_SIDE_CONDITIONS)
    pokemon_left:  float = 0.0
    total_fainted: float = 0.0


@dataclass
class FieldFeatures:
    weather:       list         = field(default_factory=lambda: [0.0] * N_WEATHER)
    terrain:       list         = field(default_factory=lambda: [0.0] * N_TERRAIN)
    pseudo_weather: list        = field(default_factory=lambda: [0.0] * N_PSEUDO)
    side0:         SideFeatures = field(default_factory=SideFeatures)
    side1:         SideFeatures = field(default_factory=SideFeatures)


def encode_side(side: dict) -> SideFeatures:
    f  = SideFeatures()
    sc = side.get("side_conditions", {})
    for i, (cond_id, max_layers) in enumerate(SIDE_CONDITIONS):
        layers = sc.get(cond_id, 0)
        f.conditions[i] = layers / max_layers
    f.pokemon_left  = side.get("pokemon_left",  6) / 6.0
    f.total_fainted = side.get("total_fainted", 0) / 6.0
    return f


def encode_field(state: dict) -> FieldFeatures:
    """Encode the global field state from a PyBattle state dict."""
    raw = state.get("field", {})
    f   = FieldFeatures()

    w = raw.get("weather", "")
    if w in WEATHER_INDEX:
        f.weather[WEATHER_INDEX[w]] = 1.0

    t = raw.get("terrain", "")
    if t in TERRAIN_INDEX:
        f.terrain[TERRAIN_INDEX[t]] = 1.0

    pw_active = set(raw.get("pseudo_weather", []))
    for i, pw in enumerate(PSEUDO_WEATHER_FLAGS):
        f.pseudo_weather[i] = float(pw in pw_active)

    sides = state.get("sides", [])
    if len(sides) > 0:
        f.side0 = encode_side(sides[0])
    if len(sides) > 1:
        f.side1 = encode_side(sides[1])

    return f
