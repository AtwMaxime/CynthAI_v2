"""
CynthAI_v2 Rollout — Trajectory collection for PPO self-play.

Reward design (zero-sum):
  Win / Loss          : +1.0 / -1.0    — terminal, sparse
  Opponent KO         : +0.05          — clear positive event
  Own KO              : -0.05          — symmetric penalty
  Delta HP advantage  : 0.01 * Δadv    — dense proxy for battle progress

  HP advantage = (own total hp / own total maxhp) - (opp total hp / opp total maxhp)
  Δadv = adv_current - adv_previous   (bounded ≈ [-1, 1] per turn)

Collection strategy:
  - Complete episodes (not fixed-length rollout): cleaner with terminal win/loss reward
  - N parallel battles (n_envs=16 default): fills buffer fast, GIL released in Rust sim
  - GAE with γ=0.99, λ=0.95 computed after collection

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
from collections import deque
from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from model.backbone import K_TURNS, D_MODEL
from model.embeddings import (
    PokemonBatch, FieldBatch,
    collate_features, collate_field_features,
    FIELD_DIM,
)
from env.state_encoder import (
    PokemonFeatures, FieldFeatures,
    encode_pokemon, encode_field,
    MOVE_INDEX, TYPE_INDEX, UNK,
)
from env.action_space import MECH_NONE, MECH_TERA

if TYPE_CHECKING:
    from model.agent import CynthAIAgent


# ── Reward hyperparameters ────────────────────────────────────────────────────

WIN_REWARD     =  1.0
LOSS_REWARD    = -1.0
KO_REWARD      =  0.05
OWN_KO_PENALTY = -0.05
HP_ADV_SCALE   =  0.01


# ── Random policy (for evaluation) ───────────────────────────────────────────

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


# ── Transition ────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """One (s, a, r, done) step for one player, stored on CPU."""

    # ── State for the backbone (K turns already stacked) ─────────────────────
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

    # ── ActionEncoder inputs (current turn, active Pokémon) ───────────────────
    move_idx:          torch.Tensor   # [4]  int64
    pp_ratio:          torch.Tensor   # [4]  float32
    move_disabled:     torch.Tensor   # [4]  float32
    mechanic_id:       int
    mechanic_type_idx: int

    # ── Action / mask ─────────────────────────────────────────────────────────
    action:       int
    log_prob_old: float
    action_mask:  torch.Tensor   # [13] bool

    # ── Reward / episode ──────────────────────────────────────────────────────
    reward:    float
    done:      bool
    value_old: float

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


# ── Reward ────────────────────────────────────────────────────────────────────

def _hp_ratio(pokemon_list: list[dict]) -> float:
    total_hp    = sum(p["hp"]    for p in pokemon_list)
    total_maxhp = sum(p["maxhp"] for p in pokemon_list)
    return total_hp / total_maxhp if total_maxhp > 0 else 0.0


def compute_step_reward(
    prev_state: dict,
    curr_state: dict,
    done:       bool,
    won:        bool | None,
    side_idx:   int = 0,
) -> float:
    reward  = 0.0
    opp_idx = 1 - side_idx

    own_prev = prev_state["sides"][side_idx]
    opp_prev = prev_state["sides"][opp_idx]
    own_curr = curr_state["sides"][side_idx]
    opp_curr = curr_state["sides"][opp_idx]

    new_opp_ko = opp_curr["total_fainted"] - opp_prev["total_fainted"]
    new_own_ko = own_curr["total_fainted"] - own_prev["total_fainted"]
    reward += KO_REWARD      * max(new_opp_ko, 0)
    reward += OWN_KO_PENALTY * max(new_own_ko, 0)

    hp_adv_prev = _hp_ratio(own_prev["pokemon"]) - _hp_ratio(opp_prev["pokemon"])
    hp_adv_curr = _hp_ratio(own_curr["pokemon"]) - _hp_ratio(opp_curr["pokemon"])
    reward += HP_ADV_SCALE * (hp_adv_curr - hp_adv_prev)

    if done:
        reward += WIN_REWARD if won else LOSS_REWARD
    return reward


# ── State encoding helpers ────────────────────────────────────────────────────

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
    For v1 self-play, both sides see full state — fine for random battle training.
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

    # Fainted Pokémon must switch — no move slots legal
    if not fainted:
        # Move slots 0-3
        if not force_sw:
            for i, mv in enumerate(active.get("moves", [])[:4]):
                if not mv.get("disabled", False):
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
        bench = [
            j for j, p in enumerate(side["pokemon"])
            if j not in active_set and not p.get("fainted", False)
        ]
        team_pos = bench[action - 8]   # 0-indexed team position
        return f"switch {team_pos + 1}"  # PS is 1-indexed
    raise ValueError(f"Invalid action index: {action}")


# ── Sliding window per env ────────────────────────────────────────────────────

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


# ── Rollout buffer + GAE ──────────────────────────────────────────────────────

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
        self._returns    = ret

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
        }


# ── Batch input builder ───────────────────────────────────────────────────────

def _build_agent_inputs(
    windows:    list[BattleWindow],
    states:     list[dict],
    side_idx:   int,
    device:     torch.device,
) -> tuple:
    """
    Build batched model inputs for N parallel envs from their BattleWindows.

    Returns (poke_batch, field_tensor, move_idx, pp_ratio, move_disabled,
             mechanic_id, mechanic_type_idx) — all on device.
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
    probs[~action_mask] = log_probs[~action_mask].exp()
    return torch.multinomial(probs, num_samples=1).squeeze(1)   # [B]


# ── Main collection function ──────────────────────────────────────────────────

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

    for i, s in enumerate(prev_states):
        pf, ff = encode_state(s, side_self); wins_self[i].push(pf, ff)
        pf, ff = encode_state(s, side_opp);  wins_opp[i].push(pf, ff)

    next_seed = n_envs

    with torch.no_grad():
        while len(buffer) < min_steps:

            ins_self = _build_agent_inputs(wins_self, prev_states, side_self, device)
            ins_opp  = _build_agent_inputs(wins_opp,  prev_states, side_opp,  device)

            masks_self = torch.stack([build_action_mask(s, side_self) for s in prev_states]).to(device)
            masks_opp  = torch.stack([build_action_mask(s, side_opp)  for s in prev_states]).to(device)

            out_self = agent_self(*ins_self, masks_self)
            out_opp  = agent_opp(*ins_opp,  masks_opp)

            acts_self = _sample_action(out_self.log_probs, masks_self)
            acts_opp  = _sample_action(out_opp.log_probs,  masks_opp)

            curr_states = []
            for i in range(n_envs):
                c_self = action_to_choice(acts_self[i].item(), prev_states[i], side_self)
                c_opp  = action_to_choice(acts_opp[i].item(),  prev_states[i], side_opp)
                p1, p2 = (c_self, c_opp) if side_self == 0 else (c_opp, c_self)
                try:
                    envs[i].make_choices(p1, p2)
                except Exception:
                    # Diagnostic: log the state that caused the crash
                    import sys, traceback as _tb
                    print(f"CRASH in make_choices env={i} p1={p1!r} p2={p2!r}", file=sys.stderr)
                    s = prev_states[i]
                    for si in [0, 1]:
                        side = s["sides"][si]
                        active_set = [a for a in side["active"] if a is not None]
                        if active_set:
                            p = side["pokemon"][active_set[0]]
                            print(f"  side{si}: {p.get('species_id','?')} hp={p.get('hp','?')}/{p.get('maxhp','?')} "
                                  f"fainted={p.get('fainted')} force_sw={p.get('force_switch_flag')} "
                                  f"trapped={p.get('trapped')} status={p.get('status','')}", file=sys.stderr)
                    _tb.print_exc()
                    raise
                curr_states.append(envs[i].get_state())

            for i in range(n_envs):
                done   = envs[i].ended
                winner = envs[i].winner
                won    = (winner == "p1") if side_self == 0 else (winner == "p2")
                reward = compute_step_reward(prev_states[i], curr_states[i], done, won if done else None, side_self)

                pb    = ins_self[0]
                ft    = ins_self[1]
                midx  = ins_self[2]
                ppr   = ins_self[3]
                mdis  = ins_self[4]
                mech  = ins_self[5][i].item()
                mtype = ins_self[6][i].item()
                action = acts_self[i].item()

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
                    reward            = reward,
                    done              = done,
                    value_old         = out_self.value[i, 0].item(),
                ))

                if done:
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
