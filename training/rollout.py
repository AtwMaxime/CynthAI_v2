"""
CynthAI_v2 Rollout â€” Trajectory collection for PPO self-play.

Reward design (zero-sum):
  Win / Loss          : +1.0 / -1.0    â€” terminal, sparse
  Opponent KO         : +0.05          â€” clear positive event
  Own KO              : -0.05          â€” symmetric penalty
  Delta HP advantage  : 0.01 * Î”adv    â€” dense proxy for battle progress

  HP advantage = (own total hp / own total maxhp) - (opp total hp / opp total maxhp)
  Î”adv = adv_current - adv_previous   (bounded â‰ˆ [-1, 1] per turn)

Collection strategy:
  - Complete episodes (not fixed-length rollout): cleaner with terminal win/loss reward
  - N parallel battles (n_envs=16 default): fills buffer fast, GIL released in Rust sim
  - GAE with Î³=0.99, Î»=0.95 computed after collection

Self-play:
  - agent_self and agent_opp are separate nn.Module instances
  - Both play under torch.no_grad() during rollout
  - At training start, agent_opp = agent_self (same object); updated to old checkpoint later

Action mask convention:
  True = ILLEGAL (consistent with backbone.py)
  Derived from get_state() since the Rust sim has no get_legal_actions() method.

Pokemon ordering in the 12-token sequence (per turn):
  [0-5]  own side: active first, then bench in team order
  [6-11] opp side: same convention
"""

from __future__ import annotations

import random
import sys
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from model.backbone import K_TURNS, D_MODEL
from model.embeddings import (
    PokemonBatch, FieldBatch,
    collate_features, collate_field_features,
    FIELD_DIM, apply_reveal_mask,
)
from env.state_encoder import (
    PokemonFeatures, FieldFeatures,
    encode_pokemon, encode_field,
    MOVE_INDEX, TYPE_INDEX, UNK,
)
from env.action_space import MECH_NONE, MECH_TERA
from env.revealed_tracker import RevealedTracker

if TYPE_CHECKING:
    from model.agent import CynthAIAgent


# â”€â”€ Reward hyperparameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

WIN_REWARD       =  1.0
LOSS_REWARD      = -1.0
KO_REWARD        =  0.5      # P17: 10× (was 0.05) — signal dense trop faible
OWN_KO_PENALTY   = -0.5      # P17: 10× (was -0.05)
HP_ADV_SCALE     =  0.5      # P17: 10× (was 0.05)
COUNT_ADV_SCALE  =  0.3      # P17: 10× (was 0.03)
STATUS_REWARD          =  0.1      # P17: nouveau — statut infligé à l'adversaire
HAZARD_REWARD          =  0.1      # P17: nouveau — hazard posé côté adverse
HAZARD_REMOVE_REWARD   =  0.1      # P17: nouveau — hazard retiré de notre côté (Rapid Spin/Defog)


# â”€â”€ Random policy (for evaluation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class RandomPolicyOutput:
    value:     torch.Tensor   # [B, 1]
    log_probs: torch.Tensor   # [B, 13]


class RandomPolicy:
    """
    Lightweight policy that samples uniformly from legal actions.
    Matches the call signature of CynthAIAgent for use in collect_rollout.
    """

    def __call__(
        self,
        poke_batch, field_tensor,
        move_idx, pp_ratio, move_disabled,
        mechanic_id, mechanic_type_idx,
        action_mask,   # [B, 13] bool, True=illegal
    ) -> RandomPolicyOutput:
        B = action_mask.shape[0]
        dev = action_mask.device
        logits = torch.zeros(B, 13, device=dev)
        logits = logits.masked_fill(action_mask, -1e9)
        return RandomPolicyOutput(
            value=torch.zeros(B, 1, device=dev),
            log_probs=F.log_softmax(logits, dim=-1),
        )

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return []

    def state_dict(self):
        return {}


# â”€â”€ Transition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class Transition:
    """One (s, a, r, done) step for one player, stored on CPU."""

    # â”€â”€ State for the backbone (K turns already stacked) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Stored as PokemonBatch fields (int indices) rather than float tensors so
    # training can re-run poke_emb with gradients.
    species_idx:  torch.Tensor    # [K*12]
    type1_idx:    torch.Tensor    # [K*12]
    type2_idx:    torch.Tensor    # [K*12]
    tera_idx_emb: torch.Tensor    # [K*12]   (named tera_idx_emb to avoid clash)
    item_idx:     torch.Tensor    # [K*12]
    ability_idx:  torch.Tensor    # [K*12]
    move_idx_emb: torch.Tensor    # [K*12, 4]
    scalars:      torch.Tensor    # [K*12, N_SCALARS]
    field_tensor: torch.Tensor    # [K, FIELD_DIM]

    # â”€â”€ ActionEncoder inputs (current turn, active PokÃ©mon) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    move_idx:          torch.Tensor   # [4]  int64
    pp_ratio:          torch.Tensor   # [4]  float32
    move_disabled:     torch.Tensor   # [4]  float32
    mechanic_id:       int
    mechanic_type_idx: int

    # â”€â”€ Action / mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    action:       int
    log_prob_old: float
    action_mask:  torch.Tensor   # [13] bool

    # â”€â”€ Reward / episode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reward:    float
    done:      bool
    value_old: float

    # Reward decomposition (populated during eval, empty during training)
    reward_components: dict = dc_field(default_factory=dict)

    # â”€â”€ Reveal state (from RevealedTracker) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reveal_state: dict = dc_field(default_factory=dict)   # species/item/ability/tera/moves bools

    def to_poke_batch(self) -> PokemonBatch:
        return PokemonBatch(
            species_idx = self.species_idx,
            type1_idx   = self.type1_idx,
            type2_idx   = self.type2_idx,
            tera_idx    = self.tera_idx_emb,
            item_idx    = self.item_idx,
            ability_idx = self.ability_idx,
            move_idx    = self.move_idx_emb,
            scalars     = self.scalars,
        )


# â”€â”€ Reward â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _hp_ratio(pokemon_list: list[dict]) -> float:
    total_hp    = sum(p["hp"]    for p in pokemon_list)
    total_maxhp = sum(p["maxhp"] for p in pokemon_list)
    return total_hp / total_maxhp if total_maxhp > 0 else 0.0


def _count_advantage(state: dict, side_idx: int) -> float:
    """Normalised Pokémon count advantage in [-1, 1]."""
    own_alive = sum(1 for p in state["sides"][side_idx]["pokemon"] if not p.get("fainted", False))
    opp_alive = sum(1 for p in state["sides"][1 - side_idx]["pokemon"] if not p.get("fainted", False))
    return (own_alive - opp_alive) / 6.0


def compute_step_reward(
    prev_state: dict,
    curr_state: dict,
    done:       bool,
    won:        bool | None,
    side_idx:   int = 0,
    dense_scale: float = 1.0,   # P2: scale non-terminal rewards (0 = sparse only)
) -> tuple[float, dict]:
    """Returns (total_reward, components_dict) where components has keys:
    ko_own, ko_opp, hp_adv, count_adv, status, hazard, hazard_remove, terminal."""
    reward  = 0.0
    opp_idx = 1 - side_idx
    components: dict[str, float] = {"ko_own": 0.0, "ko_opp": 0.0, "hp_adv": 0.0, "count_adv": 0.0, "status": 0.0, "hazard": 0.0, "hazard_remove": 0.0, "terminal": 0.0}

    own_prev = prev_state["sides"][side_idx]
    opp_prev = prev_state["sides"][opp_idx]
    own_curr = curr_state["sides"][side_idx]
    opp_curr = curr_state["sides"][opp_idx]

    new_opp_ko = opp_curr["total_fainted"] - opp_prev["total_fainted"]
    new_own_ko = own_curr["total_fainted"] - own_prev["total_fainted"]
    ko_opp_rew = KO_REWARD      * max(new_opp_ko, 0) * dense_scale
    ko_own_rew = OWN_KO_PENALTY * max(new_own_ko, 0) * dense_scale
    reward += ko_opp_rew + ko_own_rew
    components["ko_opp"] = ko_opp_rew
    components["ko_own"] = ko_own_rew

    hp_adv_prev = _hp_ratio(own_prev["pokemon"]) - _hp_ratio(opp_prev["pokemon"])
    hp_adv_curr = _hp_ratio(own_curr["pokemon"]) - _hp_ratio(opp_curr["pokemon"])
    hp_rew = HP_ADV_SCALE * (hp_adv_curr - hp_adv_prev) * dense_scale
    reward += hp_rew
    components["hp_adv"] = hp_rew

    # P13c: count advantage — reliable signal on KOs
    count_prev = _count_advantage(prev_state, side_idx)
    count_curr = _count_advantage(curr_state, side_idx)
    count_rew = COUNT_ADV_SCALE * (count_curr - count_prev) * dense_scale
    reward += count_rew
    components["count_adv"] = count_rew

    # P17: status inflicted on opponent
    for poke_prev, poke_curr in zip(opp_prev["pokemon"], opp_curr["pokemon"]):
        if not poke_prev.get("status") and poke_curr.get("status"):
            reward += STATUS_REWARD * dense_scale
            components["status"] += STATUS_REWARD * dense_scale

    # P17: hazards set on opponent's side
    opp_sc_prev = opp_prev.get("side_conditions", {})
    opp_sc_curr = opp_curr.get("side_conditions", {})
    for hazard_id in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
        prev_layers = opp_sc_prev.get(hazard_id, 0)
        curr_layers = opp_sc_curr.get(hazard_id, 0)
        new_layers = max(0, curr_layers - prev_layers)
        if new_layers > 0:
            haz_rew = HAZARD_REWARD * new_layers * dense_scale
            reward += haz_rew
            components["hazard"] += haz_rew

    # P17: hazards removed from our side (Rapid Spin / Defog)
    own_sc_prev = own_prev.get("side_conditions", {})
    own_sc_curr = own_curr.get("side_conditions", {})
    for hazard_id in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
        prev_layers = own_sc_prev.get(hazard_id, 0)
        curr_layers = own_sc_curr.get(hazard_id, 0)
        removed_layers = max(0, prev_layers - curr_layers)
        if removed_layers > 0:
            haz_rem_rew = HAZARD_REMOVE_REWARD * removed_layers * dense_scale
            reward += haz_rem_rew
            components["hazard_remove"] += haz_rem_rew

    if done:
        terminal = WIN_REWARD if won else LOSS_REWARD   # terminal rewards never scaled
        components["terminal"] = terminal
        reward += terminal
    return reward, components


# â”€â”€ State encoding helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _encode_side_pokemon(state: dict, side_idx: int) -> list[PokemonFeatures]:
    """6 PokemonFeatures for one side: active first, then bench in team order."""
    side = state["sides"][side_idx]
    active_set = {i for i in side["active"] if i is not None}
    team = side["pokemon"]

    result: list[PokemonFeatures] = []
    for pos in sorted(active_set):
        result.append(encode_pokemon(team[pos]))
    for j, poke in enumerate(team):
        if j not in active_set:
            result.append(encode_pokemon(poke))
    while len(result) < 6:
        result.append(PokemonFeatures())
    return result[:6]


def encode_state(state: dict, side_idx: int) -> tuple[list[PokemonFeatures], FieldFeatures]:
    """
    Encode a battle state from the perspective of side_idx.

    Returns (12 PokemonFeatures, FieldFeatures):
      [0-5]  own side (active first)
      [6-11] opponent side (active first)

    Note: POMDP masking (hiding unrevealed opponent info) is deferred to v2.
    For v1 self-play, both sides see full state â€” fine for random battle training.
    """
    own = _encode_side_pokemon(state, side_idx)
    opp = _encode_side_pokemon(state, 1 - side_idx)
    return own + opp, encode_field(state)


def build_action_mask(state: dict, side_idx: int) -> torch.Tensor:
    """
    Derive [13] bool action mask from the current battle state.
    True = ILLEGAL. Slot layout: 0-3 normal moves, 4-7 mechanic, 8-12 switches.
    """
    mask = torch.ones(13, dtype=torch.bool)   # all illegal by default
    side = state["sides"][side_idx]
    active_set = {i for i in side["active"] if i is not None}
    if not active_set:
        return mask

    active_pos = next(iter(active_set))
    active     = side["pokemon"][active_pos]
    fainted    = bool(active.get("fainted", False))
    force_sw   = bool(active.get("force_switch_flag", False))
    trapped    = bool(active.get("trapped", False))

    # Use the Rust sim's own request_state if available â€” it's the source of truth
    # for whether a switch is expected, covering cases like U-turn, Baton Pass, etc.
    req_state = side.get("request_state", "")

    # Guard: if the Rust sim has no active request for this side, ALL actions
    # are illegal â€” even if we have an active, unfainted PokÃ©mon. This prevents
    # trying to act in an inconsistent sim state (e.g. after a partial turn
    # resolution where one side has "None" request_state but the other has
    # "Switch"). The env will be reset by the caller.
    if req_state == "None":
        return mask

    # When Rust sim expects a switch, only switch slots are legal
    if req_state == "Switch":
        # Check for Revival Blessing: when the active slot has revivalblessing
        # condition, we MUST target a fainted PokÃ©mon (the Rust sim rejects
        # non-fainted targets during revival blessing).
        slot_conds = side.get("slot_conditions", {})
        is_revival = "revivalblessing" in slot_conds.get(active.get("position", 0), [])

        if is_revival:
            # Revival Blessing: only fainted bench PokÃ©mon are legal targets
            bench = [
                j for j, p in enumerate(side["pokemon"])
                if j not in active_set and p.get("fainted", False)
            ]
        else:
            # Normal switch: only alive bench PokÃ©mon are legal targets
            bench = [
                j for j, p in enumerate(side["pokemon"])
                if j not in active_set and not p.get("fainted", False)
            ]
        for k in range(min(len(bench), 5)):
            mask[8 + k] = False
        return mask

    # Fainted PokÃ©mon must switch â€” no move slots legal
    if not fainted:
        # Move slots 0-3
        if not force_sw:
            for i, mv in enumerate(active.get("moves", [])[:4]):
                disabled = mv.get("disabled", False)
                no_pp = mv.get("pp", 0) <= 0
                if not disabled and not no_pp:
                    mask[i] = False   # normal move legal

            # Mechanic slots 4-7: legal when Tera not yet used and base move is legal
            tera_used = any(p.get("terastallized") is not None for p in side["pokemon"])
            if not tera_used:
                for i in range(4):
                    if not mask[i]:
                        mask[i + 4] = False

    # Switch slots 8-12
    if fainted or force_sw or not trapped:
        bench = [
            j for j, p in enumerate(side["pokemon"])
            if j not in active_set and not p.get("fainted", False)
        ]
        for k in range(min(len(bench), 5)):
            mask[8 + k] = False

    return mask


def action_to_choice(action: int, state: dict, side_idx: int) -> str:
    """Convert a 0-12 action slot index to a PS choice string."""
    if 0 <= action <= 3:
        return f"move {action + 1}"
    if 4 <= action <= 7:
        return f"move {action - 3} terastallize"
    if 8 <= action <= 12:
        side       = state["sides"][side_idx]
        active_set = {i for i in side["active"] if i is not None}
        # Detect Revival Blessing â€” during revival, the bench includes fainted PokÃ©mon
        if active_set:
            active_pos = next(iter(active_set))
            active = side["pokemon"][active_pos]
            slot_conds = side.get("slot_conditions", {})
            is_revival = "revivalblessing" in slot_conds.get(active.get("position", 0), [])
        else:
            is_revival = False
        bench = [
            j for j, p in enumerate(side["pokemon"])
            if j not in active_set and (p.get("fainted", False) if is_revival else not p.get("fainted", False))
        ]
        team_pos = bench[action - 8]   # 0-indexed team position
        return f"switch {team_pos + 1}"  # PS is 1-indexed
    raise ValueError(f"Invalid action index: {action}")


# â”€â”€ Sliding window per env â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class BattleWindow:
    """K=4 sliding window of encoded turns for one battle environment."""

    def __init__(self):
        self._poke:  deque[list[PokemonFeatures]] = deque(maxlen=K_TURNS)
        self._field: deque[FieldFeatures]          = deque(maxlen=K_TURNS)

    def push(self, poke_feats: list[PokemonFeatures], field_feat: FieldFeatures) -> None:
        self._poke.append(poke_feats)
        self._field.append(field_feat)

    def reset(self) -> None:
        self._poke.clear()
        self._field.clear()

    def as_padded(self) -> tuple[list[list[PokemonFeatures]], list[FieldFeatures]]:
        """Returns K turns, zero-padded for early turns."""
        k   = len(self._poke)
        pad = K_TURNS - k
        poke  = [[PokemonFeatures() for _ in range(12)] for _ in range(pad)] + list(self._poke)
        field = [FieldFeatures() for _ in range(pad)] + list(self._field)
        return poke, field


# â”€â”€ Rollout buffer + GAE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RolloutBuffer:
    """Stores complete episodes; computes GAE; yields shuffled minibatches."""

    def __init__(self):
        self._transitions: list[Transition] = []
        self._advantages:  list[float]      = []
        self._returns:     list[float]      = []

    def add(self, t: Transition) -> None:
        self._transitions.append(t)

    def __len__(self) -> int:
        return len(self._transitions)

    def compute_gae(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        """
        Generalised Advantage Estimation over complete episodes.
        Episode boundaries are detected via Transition.done == True.
        next_value = 0 at episode end (no bootstrapping needed for complete episodes).
        """
        n   = len(self._transitions)
        adv = [0.0] * n
        ret = [0.0] * n
        gae = 0.0

        for t in reversed(range(n)):
            tr = self._transitions[t]
            if tr.done:
                next_val = 0.0
                gae      = 0.0
            else:
                next_val = self._transitions[t + 1].value_old if t + 1 < n else 0.0

            delta = tr.reward + gamma * next_val - tr.value_old
            gae   = delta + gamma * lam * (0.0 if tr.done else gae)
            adv[t] = gae
            ret[t] = gae + tr.value_old

        self._advantages = adv

        # Normalise returns globally (once per rollout, before minibatch splits)
        # so the value head has a stable target across all minibatches.
        ret_t = torch.tensor(ret, dtype=torch.float32)
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
        self._returns = ret_t.tolist()

    def minibatches(self, batch_size: int, device: torch.device):
        """Yield shuffled minibatches as dicts of batched tensors."""
        n       = len(self._transitions)
        indices = list(range(n))
        random.shuffle(indices)
        for start in range(0, n - batch_size + 1, batch_size):
            yield self._gather(indices[start:start + batch_size], device)

    def _gather(self, indices: list[int], device: torch.device) -> dict:
        trs = [self._transitions[i] for i in indices]

        poke_batch = PokemonBatch(
            species_idx = torch.stack([t.species_idx  for t in trs]).to(device),
            type1_idx   = torch.stack([t.type1_idx    for t in trs]).to(device),
            type2_idx   = torch.stack([t.type2_idx    for t in trs]).to(device),
            tera_idx    = torch.stack([t.tera_idx_emb for t in trs]).to(device),
            item_idx    = torch.stack([t.item_idx     for t in trs]).to(device),
            ability_idx = torch.stack([t.ability_idx  for t in trs]).to(device),
            move_idx    = torch.stack([t.move_idx_emb for t in trs]).to(device),
            scalars     = torch.stack([t.scalars      for t in trs]).to(device),
        )

        return {
            "poke_batch":        poke_batch,
            "field_tensor":      torch.stack([t.field_tensor  for t in trs]).to(device),
            "move_idx":          torch.stack([t.move_idx          for t in trs]).to(device),
            "pp_ratio":          torch.stack([t.pp_ratio          for t in trs]).to(device),
            "move_disabled":     torch.stack([t.move_disabled      for t in trs]).to(device),
            "mechanic_id":       torch.tensor([t.mechanic_id       for t in trs], dtype=torch.long,    device=device),
            "mechanic_type_idx": torch.tensor([t.mechanic_type_idx for t in trs], dtype=torch.long,    device=device),
            "actions":           torch.tensor([t.action            for t in trs], dtype=torch.long,    device=device),
            "log_prob_old":      torch.tensor([t.log_prob_old      for t in trs], dtype=torch.float32, device=device),
            "action_mask":       torch.stack([t.action_mask        for t in trs]).to(device),
            "advantages":        torch.tensor([self._advantages[i] for i in indices], dtype=torch.float32, device=device),
            "returns":           torch.tensor([self._returns[i]    for i in indices], dtype=torch.float32, device=device),
            "reveal_species":  torch.tensor([t.reveal_state.get("species", (False,)*6) for t in trs], device=device),
            "reveal_item":     torch.tensor([t.reveal_state.get("item",    (False,)*6) for t in trs], device=device),
            "reveal_ability":  torch.tensor([t.reveal_state.get("ability", (False,)*6) for t in trs], device=device),
            "reveal_tera":     torch.tensor([t.reveal_state.get("tera",    (False,)*6) for t in trs], device=device),
            "reveal_moves":    torch.tensor([t.reveal_state.get("moves",   ((False,)*4,)*6) for t in trs], device=device),
        }


# â”€â”€ Batch input builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_agent_inputs(
    windows:    list[BattleWindow],
    states:     list[dict],
    side_idx:   int,
    device:     torch.device,
) -> tuple:
    """
    Build batched model inputs for N parallel envs from their BattleWindows.

    Returns (poke_batch, field_tensor, move_idx, pp_ratio, move_disabled,
             mechanic_id, mechanic_type_idx) â€” all on device.
    """
    B = len(windows)

    all_poke_flat:  list[list[PokemonFeatures]] = []
    all_field_flat: list[FieldFeatures]         = []

    for win in windows:
        poke_turns, field_turns = win.as_padded()
        flat = []
        for turn in poke_turns:
            flat.extend(turn)
        all_poke_flat.append(flat)
        all_field_flat.extend(field_turns)

    poke_batch   = collate_features(all_poke_flat).to(device)
    field_tensor = collate_field_features(all_field_flat).field
    field_tensor = field_tensor.reshape(B, K_TURNS, FIELD_DIM).to(device)

    move_idx_list  = []
    pp_ratio_list  = []
    move_dis_list  = []
    mech_id_list   = []
    mech_type_list = []

    for state in states:
        side       = state["sides"][side_idx]
        active_set = {i for i in side["active"] if i is not None}
        active_pos = next(iter(active_set)) if active_set else 0
        active     = side["pokemon"][active_pos]

        midx, ppr, mdis = [], [], []
        for mv in (active.get("moves") or [])[:4]:
            midx.append(MOVE_INDEX.get(mv["id"], UNK))
            maxpp = mv["maxpp"]
            ppr.append(mv["pp"] / maxpp if maxpp > 0 else 0.0)
            mdis.append(1.0 if mv.get("disabled") else 0.0)
        while len(midx) < 4:
            midx.append(UNK); ppr.append(0.0); mdis.append(0.0)

        move_idx_list.append(midx)
        pp_ratio_list.append(ppr)
        move_dis_list.append(mdis)

        tera_used = any(p.get("terastallized") is not None for p in side["pokemon"])
        mech_id   = MECH_NONE if tera_used else MECH_TERA
        tera_str  = active.get("tera_type") or ""
        mech_type = TYPE_INDEX.get(tera_str.lower(), UNK) if tera_str else UNK
        mech_id_list.append(mech_id)
        mech_type_list.append(mech_type)

    return (
        poke_batch,
        field_tensor,
        torch.tensor(move_idx_list,  dtype=torch.long,    device=device),
        torch.tensor(pp_ratio_list,  dtype=torch.float32, device=device),
        torch.tensor(move_dis_list,  dtype=torch.float32, device=device),
        torch.tensor(mech_id_list,   dtype=torch.long,    device=device),
        torch.tensor(mech_type_list, dtype=torch.long,    device=device),
    )


def _sample_action(log_probs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Sample one action per row from masked log-probability distribution."""
    probs = torch.zeros_like(log_probs)
    legal = ~action_mask
    probs[legal] = log_probs[legal].exp()
    # Fallback if no legal actions (all PokÃ©mon fainted): pick action 0
    no_legal = ~legal.any(dim=1)
    if no_legal.any():
        probs[no_legal, 0] = 1.0
    return torch.multinomial(probs, num_samples=1).squeeze(1)   # [B]


# â”€â”€ Main collection function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def collect_rollout(
    agent_self: "CynthAIAgent",
    agent_opp:  "CynthAIAgent",
    n_envs:     int   = 16,
    min_steps:  int   = 512,
    format_id:  str   = "gen9randombattle",
    gamma:      float = 0.99,
    lam:        float = 0.95,
    device:     torch.device = torch.device("cpu"),
    side_self:  int   = 0,
    mask_ratio: float = 0.0,    # P1: POMDP masking ratio (0=off, 1=full)
    dense_scale: float = 1.0,   # P2: scale non-terminal rewards
    max_crashes: int = 50,      # abort rollout if too many invalid choices
) -> RolloutBuffer:
    """
    Collect complete episodes across n_envs parallel battles until min_steps
    transitions are stored for agent_self. agent_opp acts on the other side.

    Both agents run under torch.no_grad().
    Battles are seeded randomly; reseeded after each episode end.

    Returns a RolloutBuffer with GAE already computed.
    """
    from simulator import PyBattle

    side_opp  = 1 - side_self
    buffer    = RolloutBuffer()
    seeds     = list(range(n_envs))

    envs        = [PyBattle(format_id, seed=s) for s in seeds]
    wins_self   = [BattleWindow() for _ in range(n_envs)]
    wins_opp    = [BattleWindow() for _ in range(n_envs)]
    prev_states = [e.get_state() for e in envs]
    tracker_self = RevealedTracker(n_envs)  # what self sees of opponent
    tracker_opp  = RevealedTracker(n_envs)  # what opponent sees of self

    for i, s in enumerate(prev_states):
        pf, ff = encode_state(s, side_self); wins_self[i].push(pf, ff)
        pf, ff = encode_state(s, side_opp);  wins_opp[i].push(pf, ff)

    next_seed = n_envs
    total_crashes = 0

    with torch.no_grad():
        while len(buffer) < min_steps:

            ins_self = _build_agent_inputs(wins_self, prev_states, side_self, device)
            ins_opp  = _build_agent_inputs(wins_opp,  prev_states, side_opp,  device)

            masks_self = torch.stack([build_action_mask(s, side_self) for s in prev_states]).to(device)
            masks_opp  = torch.stack([build_action_mask(s, side_opp)  for s in prev_states]).to(device)

            # P1: POMDP masking â€” apply reveal mask during rollout decision
            unmasked_pb_self = ins_self[0]
            unmasked_pb_opp  = ins_opp[0]
            if mask_ratio > 0.0:
                rs_self = [tracker_self.get_state(i) for i in range(n_envs)]
                ins_self = (
                    apply_reveal_mask(
                        unmasked_pb_self,
                        reveal_species = torch.tensor([r["species"] for r in rs_self], device=device),
                        reveal_item    = torch.tensor([r["item"]    for r in rs_self], device=device),
                        reveal_ability = torch.tensor([r["ability"] for r in rs_self], device=device),
                        reveal_tera    = torch.tensor([r["tera"]    for r in rs_self], device=device),
                        reveal_moves   = torch.tensor([r["moves"]   for r in rs_self], device=device),
                        mask_ratio     = mask_ratio,
                    ),
                ) + ins_self[1:]

                rs_opp = [tracker_opp.get_state(i) for i in range(n_envs)]
                ins_opp = (
                    apply_reveal_mask(
                        unmasked_pb_opp,
                        reveal_species = torch.tensor([r["species"] for r in rs_opp], device=device),
                        reveal_item    = torch.tensor([r["item"]    for r in rs_opp], device=device),
                        reveal_ability = torch.tensor([r["ability"] for r in rs_opp], device=device),
                        reveal_tera    = torch.tensor([r["tera"]    for r in rs_opp], device=device),
                        reveal_moves   = torch.tensor([r["moves"]   for r in rs_opp], device=device),
                        mask_ratio     = mask_ratio,
                    ),
                ) + ins_opp[1:]

            out_self = agent_self(*ins_self, masks_self)

            # Bot vs neural opponent dispatch
            if hasattr(agent_opp, "act"):
                # Rule-based bot (e.g. FullOffensePolicy) â€” use .act(states)
                acts_opp = agent_opp.act(prev_states, side_opp, masks_opp)
                # Bot provides no value/log_probs; use zeros for buffer
                _bot_value    = torch.zeros(n_envs, 1, device=device)
                _bot_log_probs = torch.full((n_envs, 13), -100.0, device=device)
                _bot_log_probs = _bot_log_probs.masked_fill(masks_opp, -1e9)
                out_opp = type("_BotOut", (), {
                    "value": _bot_value, "log_probs": _bot_log_probs,
                })()
            else:
                out_opp  = agent_opp(*ins_opp,  masks_opp)

            acts_self = _sample_action(out_self.log_probs, masks_self)
            acts_opp  = _sample_action(out_opp.log_probs,  masks_opp) if not hasattr(agent_opp, "act") else acts_opp

            curr_states = []
            crashed: set[int] = set()
            skip_buffer: set[int] = set()
            chosen_actions_self: list[int] = []
            chosen_actions_opp:  list[int] = []
            subt_envs: dict[int, tuple] = {}  # env_idx -> (fresh_state, req_self, req_opp)

            for i in range(n_envs):
                # â”€â”€ Couche 1 : re-valider avec un Ã©tat frais â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                fresh_state = envs[i].get_state()
                req_self = fresh_state["sides"][side_self].get("request_state", "")
                req_opp = fresh_state["sides"][side_opp].get("request_state", "")

                # â”€â”€ DÃ©tection Switch sub-turn (aprÃ¨s KO / forced switch) â”€â”€â”€â”€
                if req_self == "None" or req_opp == "None":
                    # Les deux en None â†’ la bataille n'attend rien, on avance
                    if req_self != "Switch" and req_opp != "Switch":
                        curr_states.append(fresh_state)
                        skip_buffer.add(i)
                        chosen_actions_self.append(0)
                        chosen_actions_opp.append(0)
                        continue

                    # Sub-turn Switch : collecter pour forward pass aprÃ¨s la boucle
                    subt_envs[i] = (fresh_state, req_self, req_opp)
                    curr_states.append(fresh_state)  # placeholder, sera mis Ã  jour
                    chosen_actions_self.append(0)  # placeholder
                    chosen_actions_opp.append(0)
                    continue

                # â”€â”€ Tour normal : les deux sides ont des requests valides â”€â”€
                fresh_mask_self = build_action_mask(fresh_state, side_self)
                fresh_mask_opp  = build_action_mask(fresh_state, side_opp)

                a_self = acts_self[i].item()
                a_opp  = acts_opp[i].item()

                chosen_actions_self.append(a_self)
                chosen_actions_opp.append(a_opp)

                c_self = action_to_choice(a_self, fresh_state, side_self)
                c_opp  = action_to_choice(a_opp,  fresh_state, side_opp)
                p1, p2 = (c_self, c_opp) if side_self == 0 else (c_opp, c_self)

                ok = envs[i].make_choices(p1, p2)
                if ok:
                    log_entries = envs[i].get_new_log_entries()
                    curr_states.append(envs[i].get_state())
                    tracker_self.update(i, log_entries, curr_states[-1], side_opp)
                    tracker_opp.update(i, log_entries, curr_states[-1], side_self)
                else:
                    # Choice invalide â†’ reset env, skip transition.
                    print(f"INVALID choice env={i} p1={p1!r} p2={p2!r}", file=sys.stderr)
                    envs[i] = PyBattle(format_id, seed=next_seed)
                    next_seed += 1
                    wins_self[i].reset()
                    wins_opp[i].reset()
                    tracker_self.reset(i)
                    tracker_opp.reset(i)
                    curr_states.append(envs[i].get_state())
                    crashed.add(i)

            # â”€â”€ Traitement des Switch sub-turns : forward pass dÃ©diÃ© â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if subt_envs:
                # Grouper par side qui doit switcher
                self_need_switch = {i: v for i, v in subt_envs.items() if v[1] == "Switch"}
                opp_need_switch  = {i: v for i, v in subt_envs.items() if v[2] == "Switch"}

                # --- side_self sub-turns (agent_self choisit le switch) ---
                if self_need_switch:
                    indices = list(self_need_switch.keys())
                    sub_states = [self_need_switch[i][0] for i in indices]
                    sub_wins = [wins_self[i] for i in indices]
                    sub_ins = _build_agent_inputs(sub_wins, sub_states, side_self, device)
                    sub_masks = torch.stack(
                        [build_action_mask(s, side_self) for s in sub_states]
                    ).to(device)
                    sub_out = agent_self(*sub_ins, sub_masks)
                    sub_acts = _sample_action(sub_out.log_probs, sub_masks)

                    for j, i in enumerate(indices):
                        fresh_state = sub_states[j]
                        a = sub_acts[j].item()

                        # Construire le choix switch
                        c_switch = action_to_choice(a, fresh_state, side_self)
                        p1, p2 = (c_switch, "") if side_self == 0 else ("", c_switch)
                        ok = envs[i].make_choices(p1, p2)

                        if ok:
                            log_entries = envs[i].get_new_log_entries()
                            new_state = envs[i].get_state()
                            curr_states[i] = new_state
                            tracker_self.update(i, log_entries, new_state, side_opp)
                            tracker_opp.update(i, log_entries, new_state, side_self)

                            # Enregistrer la transition
                            done = envs[i].ended
                            winner = envs[i].winner
                            won = (winner == "p1") if side_self == 0 else (winner == "p2")
                            reward, components = compute_step_reward(
                                fresh_state, new_state, done,
                                won if done else None, side_self, dense_scale
                            )
                            pb = sub_ins[0]
                            buffer.add(Transition(
                                species_idx       = pb.species_idx[j].cpu(),
                                type1_idx         = pb.type1_idx[j].cpu(),
                                type2_idx         = pb.type2_idx[j].cpu(),
                                tera_idx_emb      = pb.tera_idx[j].cpu(),
                                item_idx          = pb.item_idx[j].cpu(),
                                ability_idx       = pb.ability_idx[j].cpu(),
                                move_idx_emb      = pb.move_idx[j].cpu(),
                                scalars           = pb.scalars[j].cpu(),
                                field_tensor      = sub_ins[1][j].cpu(),
                                move_idx          = sub_ins[2][j].cpu(),
                                pp_ratio          = sub_ins[3][j].cpu(),
                                move_disabled     = sub_ins[4][j].cpu(),
                                mechanic_id       = sub_ins[5][j].item(),
                                mechanic_type_idx = sub_ins[6][j].item(),
                                action            = a,
                                log_prob_old      = sub_out.log_probs[j, a].item(),
                                action_mask       = sub_masks[j].cpu(),
                                reveal_state      = tracker_self.get_state(i),
                                reward            = reward,
                                reward_components = components,
                                done              = done,
                                value_old         = sub_out.value[j, 0].item(),
                            ))
                            if done:
                                tracker_self.reset(i)
                                tracker_opp.reset(i)
                                envs[i] = PyBattle(format_id, seed=next_seed)
                                next_seed += 1
                                wins_self[i].reset()
                                wins_opp[i].reset()
                                curr_states[i] = envs[i].get_state()
                        else:
                            print(f"SUB-TURN SWITCH FAIL env={i} p1={p1!r} p2={p2!r}", file=sys.stderr)
                            envs[i] = PyBattle(format_id, seed=next_seed)
                            next_seed += 1
                            wins_self[i].reset()
                            wins_opp[i].reset()
                            tracker_self.reset(i)
                            tracker_opp.reset(i)
                            curr_states[i] = envs[i].get_state()
                            crashed.add(i)

                # --- side_opp sub-turns (agent_opp choisit le switch) ---
                if opp_need_switch:
                    indices = list(opp_need_switch.keys())
                    sub_states = [opp_need_switch[i][0] for i in indices]
                    sub_masks = torch.stack(
                        [build_action_mask(s, side_opp) for s in sub_states]
                    ).to(device)

                    if hasattr(agent_opp, "act"):
                        # Bot policy
                        acts_opp_list = agent_opp.act(sub_states, side_opp, sub_masks)
                    else:
                        sub_wins = [wins_opp[i] for i in indices]
                        sub_ins = _build_agent_inputs(sub_wins, sub_states, side_opp, device)
                        sub_out = agent_opp(*sub_ins, sub_masks)
                        acts_opp_list = _sample_action(sub_out.log_probs, sub_masks)

                    for j, i in enumerate(indices):
                        fresh_state = sub_states[j]
                        a = acts_opp_list[j].item() if hasattr(agent_opp, "act") else acts_opp_list[j].item()
                        c_switch = action_to_choice(a, fresh_state, side_opp)
                        p1, p2 = ("", c_switch) if side_self == 0 else (c_switch, "")
                        ok = envs[i].make_choices(p1, p2)

                        if ok:
                            log_entries = envs[i].get_new_log_entries()
                            new_state = envs[i].get_state()
                            curr_states[i] = new_state
                            tracker_self.update(i, log_entries, new_state, side_opp)
                            tracker_opp.update(i, log_entries, new_state, side_self)
                        else:
                            print(f"SUB-TURN SWITCH FAIL (opp) env={i} p1={p1!r} p2={p2!r}", file=sys.stderr)
                            envs[i] = PyBattle(format_id, seed=next_seed)
                            next_seed += 1
                            wins_self[i].reset()
                            wins_opp[i].reset()
                            tracker_self.reset(i)
                            tracker_opp.reset(i)
                            curr_states[i] = envs[i].get_state()
                            crashed.add(i)

                skip_buffer.update(subt_envs.keys())

            # Crash limit: if too many envs fail, abort this rollout.
            total_crashes += len(crashed)
            if total_crashes > max_crashes:
                print(f"ABORT rollout: {total_crashes} crashes > {max_crashes} limit, "
                      f"collected {len(buffer)} steps", file=sys.stderr)
                break

            for i in range(n_envs):
                if i in crashed or i in skip_buffer:
                    continue  # transition skip â€” switch sub-turn ou donnÃ©es corrompues

                done   = envs[i].ended
                winner = envs[i].winner
                won    = (winner == "p1") if side_self == 0 else (winner == "p2")
                reward, components = compute_step_reward(prev_states[i], curr_states[i], done, won if done else None, side_self, dense_scale)

                pb    = unmasked_pb_self
                ft    = ins_self[1]
                midx  = ins_self[2]
                ppr   = ins_self[3]
                mdis  = ins_self[4]
                mech  = ins_self[5][i].item()
                mtype = ins_self[6][i].item()
                action = chosen_actions_self[i]   # action rÃ©ellement jouÃ©e (peut diffÃ©rer de acts_self)

                buffer.add(Transition(
                    species_idx       = pb.species_idx[i].cpu(),
                    type1_idx         = pb.type1_idx[i].cpu(),
                    type2_idx         = pb.type2_idx[i].cpu(),
                    tera_idx_emb      = pb.tera_idx[i].cpu(),
                    item_idx          = pb.item_idx[i].cpu(),
                    ability_idx       = pb.ability_idx[i].cpu(),
                    move_idx_emb      = pb.move_idx[i].cpu(),
                    scalars           = pb.scalars[i].cpu(),
                    field_tensor      = ft[i].cpu(),
                    move_idx          = midx[i].cpu(),
                    pp_ratio          = ppr[i].cpu(),
                    move_disabled     = mdis[i].cpu(),
                    mechanic_id       = mech,
                    mechanic_type_idx = mtype,
                    action            = action,
                    log_prob_old      = out_self.log_probs[i, action].item(),
                    action_mask       = masks_self[i].cpu(),
                    reveal_state      = tracker_self.get_state(i),
                    reward            = reward,
                    reward_components = components,
                    done              = done,
                    value_old         = out_self.value[i, 0].item(),
                ))

                if done:
                    tracker_self.reset(i)
                    tracker_opp.reset(i)
                    envs[i] = PyBattle(format_id, seed=next_seed)
                    next_seed += 1
                    wins_self[i].reset()
                    wins_opp[i].reset()
                    curr_states[i] = envs[i].get_state()

            for i, s in enumerate(curr_states):
                pf, ff = encode_state(s, side_self); wins_self[i].push(pf, ff)
                pf, ff = encode_state(s, side_opp);  wins_opp[i].push(pf, ff)

            prev_states = curr_states

    buffer.compute_gae(gamma, lam)
    return buffer
