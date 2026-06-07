"""
experiments/estimate_ev_max.py — Expérience Monte Carlo EVπ,max

Estime EVπ,max = Var(Vπ(s)) / Var(R) sous la politique courante π.
Mesure aussi R²(Vθ, Vπ̂) : fraction de la variance prédictible capturée par le critic.

Usage:
    python experiments/estimate_ev_max.py \\
        --checkpoint checkpoints/cheater_v7/agent_000300.pt \\
        --n-per-bucket 20 \\
        --k-rollouts 50 \\
        --output-dir experiments/results/ev_max_300
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import multiprocessing as mp

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.agent import CynthAIAgent
from model.embeddings import collate_features, collate_field_features, FIELD_DIM
from model.backbone import K_TURNS
from env.state_encoder import MOVE_INDEX, TYPE_INDEX, UNK
from env.action_space import MECH_NONE, MECH_TERA
from env.team_pool import sample_teams
from training.rollout import (
    BattleWindow, build_action_mask, action_to_choice,
    encode_state, compute_step_reward,
)


# ── Constants ──────────────────────────────────────────────────────────────────

GAMMA = 0.99

TURN_BUCKETS = [
    (1,  5,   "turn_01-05"),
    (6,  10,  "turn_06-10"),
    (11, 20,  "turn_11-20"),
    (21, 999, "turn_21+"),
]

FORMAT_ID = "gen9randombattle"


# ── State snapshot ─────────────────────────────────────────────────────────────

@dataclass
class StateSnapshot:
    battle_fork: object          # PyBattle clone (ready for rollouts)
    window: BattleWindow         # deep-copied K-turn window for the forked state
    v_theta: float               # critic prediction at this state
    turn: int
    game_idx: int = -1
    rollout_returns: list[float] = dc_field(default_factory=list)


# ── Agent input builder (single state) ────────────────────────────────────────

def _get_agent_inputs(window: BattleWindow, state: dict, side_idx: int, device: torch.device):
    poke_feats, field_feat = encode_state(state, side_idx)
    window.push(poke_feats, field_feat)
    poke_turns, field_turns = window.as_padded()

    flat = []
    for turn in poke_turns:
        flat.extend(turn)

    pb = collate_features([flat]).to(device)
    ft = collate_field_features(field_turns).field.reshape(1, K_TURNS, FIELD_DIM).to(device)

    side = state["sides"][side_idx]
    active_set = {i for i in side["active"] if i is not None}
    active_pos = next(iter(active_set)) if active_set else 0
    active = side["pokemon"][active_pos]

    midx, ppr, mdis = [], [], []
    for mv in (active.get("moves") or [])[:4]:
        midx.append(MOVE_INDEX.get(mv["id"], UNK))
        maxpp = mv["maxpp"]
        ppr.append(mv["pp"] / maxpp if maxpp > 0 else 0.0)
        mdis.append(1.0 if mv.get("disabled") else 0.0)
    while len(midx) < 4:
        midx.append(UNK); ppr.append(0.0); mdis.append(0.0)

    tera_used = any(p.get("terastallized") is not None for p in side["pokemon"])
    mech_id = MECH_NONE if tera_used else MECH_TERA
    tera_str = active.get("tera_type") or ""
    mech_type = TYPE_INDEX.get(tera_str.lower(), UNK) if tera_str else UNK

    mask = build_action_mask(state, side_idx).unsqueeze(0).to(device)

    return (
        pb,
        ft,
        torch.tensor([midx],   dtype=torch.long,    device=device),
        torch.tensor([ppr],    dtype=torch.float32, device=device),
        torch.tensor([mdis],   dtype=torch.float32, device=device),
        torch.tensor([mech_id],   dtype=torch.long, device=device),
        torch.tensor([mech_type], dtype=torch.long, device=device),
        mask,
    )


def get_value(agent: CynthAIAgent, window: BattleWindow, state: dict,
              side_idx: int, device: torch.device) -> float:
    inputs = _get_agent_inputs(window, state, side_idx, device)
    with torch.no_grad():
        out = agent(*inputs)
    return out.value.item()


def sample_action(agent: CynthAIAgent, window: BattleWindow, state: dict,
                  side_idx: int, device: torch.device) -> int:
    """Sample an action stochastically from the policy (never argmax)."""
    inputs = _get_agent_inputs(window, state, side_idx, device)
    mask = inputs[-1].squeeze(0)  # [13] bool

    with torch.no_grad():
        out = agent(*inputs)

    # Stochastic sampling — essential to estimate Vπ(s), not V_greedy(s)
    dist = torch.distributions.Categorical(logits=out.action_logits.squeeze(0))
    return dist.sample().item()


# ── Phase 1: State collection ──────────────────────────────────────────────────

def collect_states_batched(
    agent: CynthAIAgent,
    n_per_bucket: int,
    device: torch.device,
    batch_size: int = 64,
    seed_start: int = 0,
) -> list[StateSnapshot]:
    """Vectorised Phase 1: run `batch_size` self-play battles in parallel with a
    single batched forward per step. Each battle contributes at most ONE snapshot
    (into its assigned target bucket) and is then reset to a fresh battle, so all
    source games are distinct (games == states, 0%% duplication).
    """
    from training.rollout import _build_agent_inputs, _sample_action
    from simulator import PyBattle

    agent.eval()
    buckets: dict[str, list[StateSnapshot]] = {b[2]: [] for b in TURN_BUCKETS}
    full: set[str] = set()

    def _bucket_for_turn(t):
        for lo, hi, name in TURN_BUCKETS:
            if lo <= t <= hi:
                return name
        return None

    def _pick_target():
        opens = [b[2] for b in TURN_BUCKETS if b[2] not in full]
        return min(opens, key=lambda nm: len(buckets[nm])) if opens else None

    def _all_full():
        return all(len(buckets[b[2]]) >= n_per_bucket for b in TURN_BUCKETS)

    state_seed = [seed_start]
    game_counter = [0]

    def _new_env():
        t1, t2 = sample_teams()
        b = PyBattle.from_packed_teams(FORMAT_ID, state_seed[0], t1, t2)
        state_seed[0] += 1
        game_counter[0] += 1
        ws, wo = BattleWindow(), BattleWindow()
        st = b.get_state()
        pf, ff = encode_state(st, 0); ws.push(pf, ff)
        pf, ff = encode_state(st, 1); wo.push(pf, ff)
        return {"battle": b, "ws": ws, "wo": wo, "gid": game_counter[0],
                "target": _pick_target(), "steps": 0}

    total_needed = n_per_bucket * len(TURN_BUCKETS)
    pbar = tqdm(total=total_needed, desc="Phase 1 collect", unit="state")

    envs = [_new_env() for _ in range(batch_size)]
    max_steps = 200

    while not _all_full():
        # reset dead / over-long / stale-target envs
        for i in range(len(envs)):
            e = envs[i]
            if (e["battle"].ended or e["steps"] >= max_steps
                    or e["target"] is None or e["target"] in full):
                envs[i] = _new_env()

        states_now = [e["battle"].get_state() for e in envs]

        dec = []
        for i, st in enumerate(states_now):
            if st["sides"][0].get("request_state", "") == "None" and \
               st["sides"][1].get("request_state", "") == "None":
                continue
            dec.append(i)
        if not dec:
            for i in range(len(envs)):
                envs[i] = _new_env()
            continue

        sub_s  = [states_now[i] for i in dec]
        ws_sub = [envs[i]["ws"] for i in dec]
        wo_sub = [envs[i]["wo"] for i in dec]
        masks_self = torch.stack([build_action_mask(st, 0) for st in sub_s]).to(device)
        masks_opp  = torch.stack([build_action_mask(st, 1) for st in sub_s]).to(device)
        ins_self = _build_agent_inputs(ws_sub, sub_s, 0, device)
        ins_opp  = _build_agent_inputs(wo_sub, sub_s, 1, device)
        with torch.no_grad():
            out_self = agent(*ins_self, masks_self)
            out_opp  = agent(*ins_opp,  masks_opp)
        acts_self = _sample_action(out_self.log_probs, masks_self)
        acts_opp  = _sample_action(out_opp.log_probs,  masks_opp)
        vals = out_self.value.reshape(-1)

        snapped = set()
        for k, i in enumerate(dec):
            e = envs[i]; st = sub_s[k]; turn = st.get("turn", 0); tb = e["target"]
            if tb is not None and tb not in full and _bucket_for_turn(turn) == tb \
               and len(buckets[tb]) < n_per_bucket:
                snap = StateSnapshot(
                    battle_fork=e["battle"].clone_battle(),
                    window=copy.deepcopy(e["ws"]),
                    v_theta=float(vals[k].item()),
                    turn=turn,
                    game_idx=e["gid"],
                )
                buckets[tb].append(snap)
                pbar.update(1)
                pbar.set_postfix({b[2]: len(buckets[b[2]]) for b in TURN_BUCKETS})
                if len(buckets[tb]) >= n_per_bucket:
                    full.add(tb)
                envs[i] = _new_env()
                snapped.add(i)

        for k, i in enumerate(dec):
            if i in snapped:
                continue
            e = envs[i]; st = sub_s[k]
            if st["sides"][0].get("request_state", "") == "None" or bool(masks_self[k].all().item()):
                a_self = "default"
            else:
                a_self = action_to_choice(int(acts_self[k].item()), st, 0)
            if st["sides"][1].get("request_state", "") == "None" or bool(masks_opp[k].all().item()):
                a_opp = "default"
            else:
                a_opp = action_to_choice(int(acts_opp[k].item()), st, 1)
            ok = e["battle"].make_choices(a_self, a_opp)
            if not ok:
                e["battle"].make_choices("default", "default")
            curr = e["battle"].get_state()
            pf, ff = encode_state(curr, 0); e["ws"].push(pf, ff)
            pf, ff = encode_state(curr, 1); e["wo"].push(pf, ff)
            e["steps"] += 1

    pbar.close()
    all_states = [s for b in buckets.values() for s in b]
    print(f"[Phase 1] Done (batched). {len(all_states)} states; {game_counter[0]} games spawned.\n")
    return all_states


def collect_states(
    agent: CynthAIAgent,
    n_per_bucket: int,
    device: torch.device,
    seed_start: int = 0,
) -> list[StateSnapshot]:
    """
    Run fresh battles (self-play with π) and snapshot states at each turn.
    Fills turn buckets until each has n_per_bucket snapshots.
    """
    from simulator import PyBattle

    # bucket -> list[StateSnapshot]
    buckets: dict[str, list[StateSnapshot]] = {b[2]: [] for b in TURN_BUCKETS}
    full_buckets: set[str] = set()

    def _bucket_for_turn(t: int) -> Optional[str]:
        for lo, hi, name in TURN_BUCKETS:
            if lo <= t <= hi:
                return name
        return None

    def _all_full() -> bool:
        return all(len(buckets[b[2]]) >= n_per_bucket for b in TURN_BUCKETS)

    seed = seed_start
    total_games = 0
    total_needed = n_per_bucket * len(TURN_BUCKETS)

    pbar = tqdm(total=total_needed, desc="Phase 1 collect", unit="state")

    while not _all_full():
        t1, t2 = sample_teams()
        battle = PyBattle.from_packed_teams(FORMAT_ID, seed, t1, t2)
        seed += 1
        total_games += 1

        win_self = BattleWindow()
        win_opp  = BattleWindow()
        prev_state = battle.get_state()
        pf, ff = encode_state(prev_state, 0); win_self.push(pf, ff)
        pf, ff = encode_state(prev_state, 1); win_opp.push(pf, ff)

        # One snapshot per battle -> independent source games. Aim for the
        # least-filled still-open bucket so every bucket fills with distinct games.
        open_buckets = [b[2] for b in TURN_BUCKETS if b[2] not in full_buckets]
        if not open_buckets:
            break
        target_bucket = min(open_buckets, key=lambda nm: len(buckets[nm]))

        max_turns = 200
        for _ in range(max_turns):
            if battle.ended:
                break

            state = battle.get_state()
            turn  = state.get("turn", 0)

            mask_self = build_action_mask(state, 0)
            mask_opp  = build_action_mask(state, 1)

            req_self = state["sides"][0].get("request_state", "")
            req_opp  = state["sides"][1].get("request_state", "")
            if req_self == "None" and req_opp == "None":
                prev_state = state
                continue

            # One snapshot per battle: only when we reach the target bucket
            bname = _bucket_for_turn(turn)
            if bname == target_bucket and len(buckets[bname]) < n_per_bucket:
                win_copy = copy.deepcopy(win_self)
                v_theta = get_value(agent, win_copy, state, side_idx=0, device=device)

                fork = battle.clone_battle()
                snap = StateSnapshot(
                    battle_fork=fork,
                    window=copy.deepcopy(win_self),
                    v_theta=v_theta,
                    turn=turn,
                    game_idx=total_games,
                )
                buckets[bname].append(snap)
                pbar.update(1)
                pbar.set_postfix({b[2]: len(buckets[b[2]]) for b in TURN_BUCKETS})
                if len(buckets[bname]) >= n_per_bucket:
                    full_buckets.add(bname)
                break  # stop this battle after one snapshot -> distinct source games

            # Advance battle: sample actions stochastically
            # Need to handle sub-turn switches gracefully
            if req_self == "None":
                a_self_str = "default"
            elif mask_self.all():
                a_self_str = "default"
            else:
                a_self = sample_action(agent, win_self, state, 0, device)
                if mask_self[a_self]:
                    legal = [i for i in range(13) if not mask_self[i]]
                    a_self = random.choice(legal) if legal else 0
                a_self_str = action_to_choice(a_self, state, 0)

            if req_opp == "None":
                a_opp_str = "default"
            elif mask_opp.all():
                a_opp_str = "default"
            else:
                a_opp = sample_action(agent, win_opp, state, 1, device)
                if mask_opp[a_opp]:
                    legal = [i for i in range(13) if not mask_opp[i]]
                    a_opp = random.choice(legal) if legal else 0
                a_opp_str = action_to_choice(a_opp, state, 1)

            ok = battle.make_choices(a_self_str, a_opp_str)
            if not ok:
                # Fallback: try default
                battle.make_choices("default", "default")

            curr_state = battle.get_state()
            pf, ff = encode_state(curr_state, 0); win_self.push(pf, ff)
            pf, ff = encode_state(curr_state, 1); win_opp.push(pf, ff)
            prev_state = curr_state

    pbar.close()
    all_states = [s for b in buckets.values() for s in b]
    print(f"[Phase 1] Done. {len(all_states)} states from {total_games} games.\n")
    return all_states


# ── Globals for fork-based parallel rollouts ──────────────────────────────────
# Set in the main process before Pool creation; inherited by workers via fork.
_g_states: list = []
_g_agent_sd: dict = {}
_g_k: int = 0
_g_agent = None  # created in each worker via _worker_init


def _worker_init() -> None:
    """Pool initializer: build agent on CPU once per worker process."""
    global _g_agent
    _g_agent = CynthAIAgent(critic_n_layers=2, critic_detach=True)
    _g_agent.load_state_dict(_g_agent_sd)
    _g_agent.eval()
    _g_agent.to(torch.device("cpu"))


def _worker_single(idx: int):
    """Run _g_k rollouts for _g_states[idx]. Returns (idx, list[float])."""
    snap = _g_states[idx]
    returns = []
    for i in range(_g_k):
        # Per-rollout seed for reproducibility (torch PRNG for action sampling)
        torch.manual_seed(idx * 100_000 + i)
        random.seed(idx * 100_000 + i)
        ret = run_mc_rollout(snap.battle_fork, snap.window, _g_agent, torch.device("cpu"))
        returns.append(ret)
    return idx, returns


# ── Phase 2: MC rollouts ───────────────────────────────────────────────────────

def run_mc_rollout(
    battle_fork: object,
    window_snap: BattleWindow,
    agent: CynthAIAgent,
    device: torch.device,
    gamma: float = GAMMA,
) -> float:
    """
    Run one complete MC rollout from a forked battle state.
    Returns discounted return R = Σ γ^t * r_t from the fork point.
    Actions are sampled stochastically from π (not greedy).
    """
    # Clone the fork so we can run multiple independent rollouts
    battle = battle_fork.clone_battle()

    win_self = copy.deepcopy(window_snap)
    win_opp  = BattleWindow()  # opponent window — reset (we only track self)

    prev_state = battle.get_state()
    # Push initial opp window
    pf, ff = encode_state(prev_state, 1); win_opp.push(pf, ff)

    total_return = 0.0
    discount = 1.0
    max_turns = 300

    for _ in range(max_turns):
        if battle.ended:
            break

        state = battle.get_state()

        req_self = state["sides"][0].get("request_state", "")
        req_opp  = state["sides"][1].get("request_state", "")

        if req_self == "None" and req_opp == "None":
            prev_state = state
            continue

        mask_self = build_action_mask(state, 0)
        mask_opp  = build_action_mask(state, 1)

        # Sample actions stochastically for both sides
        if req_self == "None" or mask_self.all():
            a_self_str = "default"
        else:
            a_self = sample_action(agent, win_self, state, 0, device)
            if mask_self[a_self]:
                legal = [i for i in range(13) if not mask_self[i]]
                a_self = random.choice(legal) if legal else 0
            a_self_str = action_to_choice(a_self, state, 0)

        if req_opp == "None" or mask_opp.all():
            a_opp_str = "default"
        else:
            a_opp = sample_action(agent, win_opp, state, 1, device)
            if mask_opp[a_opp]:
                legal = [i for i in range(13) if not mask_opp[i]]
                a_opp = random.choice(legal) if legal else 0
            a_opp_str = action_to_choice(a_opp, state, 1)

        ok = battle.make_choices(a_self_str, a_opp_str)
        if not ok:
            battle.make_choices("default", "default")

        curr_state = battle.get_state()
        done = curr_state.get("ended", False)
        winner = curr_state.get("winner")
        won = (winner == "p1") if done else None

        reward, _ = compute_step_reward(prev_state, curr_state, done, won, side_idx=0)
        total_return += discount * reward
        discount *= gamma

        pf, ff = encode_state(curr_state, 0); win_self.push(pf, ff)
        pf, ff = encode_state(curr_state, 1); win_opp.push(pf, ff)
        prev_state = curr_state

        if done:
            break

    return total_return


def run_all_rollouts_batched(
    states: list[StateSnapshot],
    agent: CynthAIAgent,
    k_rollouts: int,
    device: torch.device,
    batch_size: int = 256,
    gamma: float = GAMMA,
) -> None:
    """Fill state.rollout_returns in-place with a vectorised multi-env rollout.

    All (state, rollout) copies are stepped as parallel envs with a single
    batched agent forward per step (mirrors training's collect_rollout), on CPU.
    Per-env stepping semantics match run_mc_rollout exactly.
    """
    from training.rollout import _build_agent_inputs, _sample_action

    agent.eval()
    jobs = [si for si, _ in enumerate(states) for _ in range(k_rollouts)]
    total = len(jobs)
    returns_by_state: list[list[float]] = [[] for _ in states]

    print(f"[Phase 2] {total} rollouts ({k_rollouts}/state) | batched CPU | batch_size={batch_size}")
    pbar = tqdm(total=total, desc="Phase 2 batched", unit="rollout")
    max_turns = 300

    def _choices(active, states_now, windows, side):
        sub_s = [states_now[j] for j in active]
        sub_w = [windows[j] for j in active]
        masks = torch.stack([build_action_mask(st, side) for st in sub_s]).to(device)
        ins = _build_agent_inputs(sub_w, sub_s, side, device)
        with torch.no_grad():
            out = agent(*ins, masks)
        acts = _sample_action(out.log_probs, masks)
        ch = {}
        for k, j in enumerate(active):
            st = sub_s[k]
            if st["sides"][side].get("request_state", "") == "None" or bool(masks[k].all().item()):
                ch[j] = "default"
            else:
                ch[j] = action_to_choice(int(acts[k].item()), st, side)
        return ch

    for start in range(0, total, batch_size):
        chunk = jobs[start:start + batch_size]
        B = len(chunk)

        battles   = [states[si].battle_fork.clone_battle() for si in chunk]
        wins_self = [copy.deepcopy(states[si].window) for si in chunk]
        wins_opp  = [BattleWindow() for _ in chunk]
        prev      = [b.get_state() for b in battles]
        for j in range(B):
            pf, ff = encode_state(prev[j], 1); wins_opp[j].push(pf, ff)

        ret      = [0.0] * B
        disc     = [1.0] * B
        finished = [False] * B

        for _ in range(max_turns):
            active = [j for j in range(B) if not finished[j] and not battles[j].ended]
            if not active:
                break
            states_now = {j: battles[j].get_state() for j in active}

            dec = []
            for j in active:
                st = states_now[j]
                if st["sides"][0].get("request_state", "") == "None" and \
                   st["sides"][1].get("request_state", "") == "None":
                    prev[j] = st
                    continue
                dec.append(j)
            if not dec:
                continue

            ch_self = _choices(dec, states_now, wins_self, 0)
            ch_opp  = _choices(dec, states_now, wins_opp, 1)

            for j in dec:
                ok = battles[j].make_choices(ch_self[j], ch_opp[j])
                if not ok:
                    battles[j].make_choices("default", "default")
                curr   = battles[j].get_state()
                done   = curr.get("ended", False)
                winner = curr.get("winner")
                won    = (winner == "p1") if done else None
                reward, _ = compute_step_reward(prev[j], curr, done, won, side_idx=0)
                ret[j]  += disc[j] * reward
                disc[j] *= gamma
                pf, ff = encode_state(curr, 0); wins_self[j].push(pf, ff)
                pf, ff = encode_state(curr, 1); wins_opp[j].push(pf, ff)
                prev[j] = curr
                if done:
                    finished[j] = True

        for j, si in enumerate(chunk):
            returns_by_state[si].append(ret[j])
            pbar.update(1)

    pbar.close()
    for si, snap in enumerate(states):
        snap.rollout_returns = returns_by_state[si]


def run_all_rollouts(
    states: list[StateSnapshot],
    agent: CynthAIAgent,
    k_rollouts: int,
    device: torch.device,
    n_workers: int = 0,
) -> None:
    """Fill state.rollout_returns for every state in-place using parallel CPU workers.

    Uses multiprocessing fork: child processes inherit the battle objects from the
    parent without pickling. The agent runs on CPU in workers (CUDA is not fork-safe).
    """
    global _g_states, _g_agent_sd, _g_k

    if n_workers <= 0:
        n_workers = min(32, mp.cpu_count())
    n_workers = min(n_workers, len(states))

    total = len(states) * k_rollouts
    est_min = total * 15 / n_workers / 60
    print(f"[Phase 2] {total} rollouts ({k_rollouts}/state) | {n_workers} CPU workers | est. ~{est_min:.0f} min")

    # Set globals before fork so workers inherit them
    _g_states   = states
    _g_agent_sd = {k: v.cpu() for k, v in agent.state_dict().items()}
    _g_k        = k_rollouts

    if n_workers <= 1:
        # Sequential fallback (safe with PyO3 objects, no fork/GIL issues)
        pbar = tqdm(total=len(states), desc="Phase 2 rollouts", unit="state")
        for idx, snap in enumerate(states):
            returns = []
            for i in range(k_rollouts):
                torch.manual_seed(idx * 100_000 + i)
                random.seed(idx * 100_000 + i)
                ret = run_mc_rollout(snap.battle_fork, snap.window, agent, device)
                returns.append(ret)
            snap.rollout_returns = returns
            pbar.update(1)
            pbar.set_postfix({"turn": snap.turn, "mean_R": f"{sum(returns)/len(returns):.3f}"})
        pbar.close()
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
            results = list(tqdm(
                pool.imap_unordered(_worker_single, range(len(states))),
                total=len(states),
                desc="Phase 2 rollouts",
                unit="state",
            ))
        for idx, returns in results:
            states[idx].rollout_returns = returns

    print("[Phase 2] Done.\n")


# ── Phase 3: Analysis ──────────────────────────────────────────────────────────

def _dist_summary(arr) -> dict:
    """Quantiles + 20-bin histogram for a 1-D array (e.g. V_pi_hat distribution)."""
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {}
    qs = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    counts, edges = np.histogram(a, bins=20)
    return {
        "n":    int(a.size),
        "mean": round(float(a.mean()), 4),
        "std":  round(float(a.std()), 4),
        "quantiles": {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs},
        "histogram": {
            "counts":    counts.tolist(),
            "bin_edges": [round(float(e), 4) for e in edges],
        },
    }


def analyse(states: list[StateSnapshot], k_rollouts: int) -> dict:
    """Compute EVπ,max, R², and per-bucket breakdowns."""

    v_pi_hats = np.array([np.mean(s.rollout_returns) for s in states])
    cond_vars  = np.array([np.var(s.rollout_returns)  for s in states])
    v_thetas   = np.array([s.v_theta for s in states])

    # Flat returns for unbiased var_R (law of total variance)
    all_returns = np.array([r for s in states for r in s.rollout_returns])
    var_R = float(np.var(all_returns))

    # EVπ,max with bias correction
    bias = float(np.mean(cond_vars)) / k_rollouts
    ev_pi_max_biased = float(np.var(v_pi_hats)) / var_R if var_R > 0 else 0.0
    ev_pi_max        = float(np.var(v_pi_hats) - bias) / var_R if var_R > 0 else 0.0
    ev_pi_max_alt    = 1.0 - float(np.mean(cond_vars)) / var_R if var_R > 0 else 0.0

    # R²(Vθ, Vπ̂)
    ss_res = float(np.sum((v_thetas - v_pi_hats) ** 2))
    ss_tot = float(np.sum((v_pi_hats - np.mean(v_pi_hats)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Pearson correlation
    if np.std(v_thetas) > 0 and np.std(v_pi_hats) > 0:
        corr = float(np.corrcoef(v_thetas, v_pi_hats)[0, 1])
    else:
        corr = float("nan")

    # Regression slope Vθ ~ Vπ̂
    if np.var(v_pi_hats) > 0:
        slope = float(np.cov(v_thetas, v_pi_hats)[0, 1] / np.var(v_pi_hats))
    else:
        slope = float("nan")

    mse = float(np.mean((v_thetas - v_pi_hats) ** 2))

    # Per-bucket analysis
    buckets_out = {}
    bucket_map: dict[str, list[int]] = defaultdict(list)
    for idx, snap in enumerate(states):
        for lo, hi, name in TURN_BUCKETS:
            if lo <= snap.turn <= hi:
                bucket_map[name].append(idx)
                break

    for lo, hi, name in TURN_BUCKETS:
        idxs = bucket_map[name]
        if not idxs:
            continue
        b_v_pi = v_pi_hats[idxs]
        b_cvar  = cond_vars[idxs]
        b_vth   = v_thetas[idxs]
        b_flat  = np.array([r for i in idxs for r in states[i].rollout_returns])
        b_var_R = float(np.var(b_flat))

        b_bias = float(np.mean(b_cvar)) / k_rollouts
        b_ev   = float(np.var(b_v_pi) - b_bias) / b_var_R if b_var_R > 0 else 0.0

        b_ss_res = float(np.sum((b_vth - b_v_pi) ** 2))
        b_ss_tot = float(np.sum((b_v_pi - np.mean(b_v_pi)) ** 2))
        b_r2 = 1.0 - b_ss_res / b_ss_tot if b_ss_tot > 0 else float("nan")

        buckets_out[name] = {
            "n_states":     len(idxs),
            "ev_pi_max":    round(b_ev, 4),
            "r2":           round(b_r2, 4),
            "var_R":        round(b_var_R, 4),
            "mean_v_pi":    round(float(np.mean(b_v_pi)), 4),
            "mean_v_theta": round(float(np.mean(b_vth)), 4),
            "mse":          round(float(np.mean((b_vth - b_v_pi)**2)), 4),
            "mean_cond_var_R": round(float(np.mean(b_cvar)), 4),
            "v_pi_dist":     _dist_summary(b_v_pi),
        }

    return {
        "global": {
            "n_states":         len(states),
            "k_rollouts":       k_rollouts,
            "var_R":            round(var_R, 4),
            "ev_pi_max":        round(ev_pi_max, 4),
            "ev_pi_max_biased": round(ev_pi_max_biased, 4),
            "ev_pi_max_alt":    round(ev_pi_max_alt, 4),
            "r2":               round(r2, 4),
            "corr":             round(corr, 4) if not np.isnan(corr) else None,
            "slope":            round(slope, 4) if not np.isnan(slope) else None,
            "mse":              round(mse, 4),
            "mean_v_pi_hat":    round(float(np.mean(v_pi_hats)), 4),
            "std_v_pi_hat":     round(float(np.std(v_pi_hats)), 4),
            "mean_v_theta":     round(float(np.mean(v_thetas)), 4),
            "std_v_theta":      round(float(np.std(v_thetas)), 4),
            "bias_correction":  round(bias, 4),
            "mean_cond_var_R":  round(float(np.mean(cond_vars)), 4),
            "v_pi_dist":        _dist_summary(v_pi_hats),
        },
        "by_bucket": buckets_out,
        "raw": {
            "v_pi_hats": v_pi_hats.tolist(),
            "v_thetas":  v_thetas.tolist(),
            "turns":     [s.turn for s in states],
            "cond_vars": cond_vars.tolist(),
        },
    }


def print_summary(results: dict) -> None:
    g = results["global"]
    print("=" * 60)
    print("GLOBAL RESULTS")
    print("=" * 60)
    print(f"  States:         {g['n_states']}   (K={g['k_rollouts']} rollouts each)")
    print(f"  Var(R):         {g['var_R']:.4f}")
    print(f"  EVpi_max:        {g['ev_pi_max']:.4f}  (bias-corrected)")
    print(f"  EVpi_max (alt):  {g['ev_pi_max_alt']:.4f}  (1 - mean(cond_var)/Var(R))")
    print(f"  R2(Vth,Vpi):     {g['r2']:.4f}")
    print(f"  corr(Vth,Vpi):   {g['corr']}")
    print(f"  slope:          {g['slope']}")
    print(f"  MSE(Vth,Vpi):    {g['mse']:.4f}")
    print()
    print("  Interpretation:")
    ev = g["ev_pi_max"]
    r2 = g["r2"]
    if ev < 0.15 and r2 > 0.6:
        print("  ->Critic is near-optimal for current policy.")
    elif ev > 0.25 and r2 < 0.3:
        print("  -> Significant predictable structure not captured - critic has room to improve.")
    elif ev > 0.25 and r2 > 0.6:
        print("  ->Critic captures most predictable structure.")
    else:
        print("  -> Mixed signal - check per-bucket breakdown.")

    print()
    print("PER-BUCKET RESULTS")
    print("-" * 60)
    print(f"  {'Bucket':<14}  {'N':>4}  {'EVpi_max':>8}  {'R2':>8}  {'Var(R)':>8}  {'MSE':>8}")
    for lo, hi, name in TURN_BUCKETS:
        b = results["by_bucket"].get(name)
        if b is None:
            continue
        print(f"  {name:<14}  {b['n_states']:>4}  {b['ev_pi_max']:>8.4f}  "
              f"{b['r2']:>8.4f}  {b['var_R']:>8.4f}  {b['mse']:>8.4f}")
    print()


def make_plots(results: dict, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plots] matplotlib not available, skipping.")
        return

    raw = results["raw"]
    vth = np.array(raw["v_thetas"])
    vpi = np.array(raw["v_pi_hats"])
    turns = np.array(raw["turns"])

    # 1. Scatter Vθ vs Vπ̂
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(vpi, vth, c=turns, cmap="viridis", alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax, label="Turn")
    lims = [min(vpi.min(), vth.min()) - 0.1, max(vpi.max(), vth.max()) + 0.1]
    ax.plot(lims, lims, "r--", lw=1, label="y=x")
    ax.set_xlabel("Vpi_hat(s)  [MC estimate]")
    ax.set_ylabel("Vtheta(s)  [critic]")
    g = results["global"]
    ax.set_title(f"Critic vs MC   R2={g['r2']:.3f}  EVpi_max={g['ev_pi_max']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_v_theta_vs_v_pi.png", dpi=150)
    plt.close(fig)

    # 2. EVπ,max and R² per bucket
    bnames = [b[2] for b in TURN_BUCKETS if b[2] in results["by_bucket"]]
    ev_vals = [results["by_bucket"][b]["ev_pi_max"] for b in bnames]
    r2_vals = [results["by_bucket"][b]["r2"]        for b in bnames]
    x = np.arange(len(bnames))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, ev_vals, width, label="EVpi_max", color="steelblue")
    ax.bar(x + width/2, r2_vals, width, label="R2",       color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(bnames, rotation=20)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("EVpi_max and R2 per turn bucket")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ev_r2_by_bucket.png", dpi=150)
    plt.close(fig)

    print(f"[Plots] Saved to {output_dir}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo EVπ,max experiment")
    parser.add_argument("--checkpoint",    required=True,      help=".pt checkpoint file")
    parser.add_argument("--n-per-bucket",  type=int, default=20)
    parser.add_argument("--k-rollouts",    type=int, default=50)
    parser.add_argument("--output-dir",    default="experiments/results/ev_max")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--n-workers",     type=int, default=0,
                        help="Parallel workers for Phase 2 rollouts (0=auto, uses all CPUs up to 32)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Run only Phase 1 and report collection/duplication stats, then exit")
    parser.add_argument("--batched", action="store_true",
                        help="Use vectorised multi-env batched rollout (CPU) for Phase 2")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of parallel envs per batch in batched rollout")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"N per bucket: {args.n_per_bucket}  |  K rollouts: {args.k_rollouts}")
    print()

    # Load agent
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    agent = CynthAIAgent(critic_n_layers=2, critic_detach=True)
    agent.load_state_dict(ckpt["model"], strict=False)
    agent.eval()
    agent.to(device)
    print(f"Agent loaded from {args.checkpoint}\n")

    # Phase 1: collect states
    if args.batched:
        states = collect_states_batched(agent, args.n_per_bucket, device, batch_size=64)
    else:
        states = collect_states(agent, args.n_per_bucket, device)

    if args.collect_only:
        from collections import defaultdict as _dd
        by_b = _dd(list)
        for s in states:
            for lo, hi, name in TURN_BUCKETS:
                if lo <= s.turn <= hi:
                    by_b[name].append(s)
                    break
        report = {}
        print("\n=== Collection diagnostics ===")
        print(f"{'Bucket':<14}{'states':>8}{'distinct':>10}{'games':>8}{'dup%':>8}{'sameBattle%':>13}")
        tot_states = tot_distinct = 0
        all_games = set()
        for lo, hi, name in TURN_BUCKETS:
            ss = by_b.get(name, [])
            n = len(ss)
            distinct = len({(x.game_idx, x.turn) for x in ss})
            games = len({x.game_idx for x in ss})
            dup = 100.0 * (n - distinct) / n if n else 0.0
            same_battle = 100.0 * (n - games) / n if n else 0.0
            report[name] = {"states": n, "distinct_snapshots": distinct,
                            "distinct_games": games,
                            "dup_pct": round(dup, 2),
                            "same_battle_pct": round(same_battle, 2)}
            print(f"{name:<14}{n:>8}{distinct:>10}{games:>8}{dup:>8.1f}{same_battle:>13.1f}")
            tot_states += n
            tot_distinct += distinct
            all_games |= {x.game_idx for x in ss}
        g_dup = 100.0 * (tot_states - tot_distinct) / tot_states if tot_states else 0.0
        g_sb = 100.0 * (tot_states - len(all_games)) / tot_states if tot_states else 0.0
        print(f"{'GLOBAL':<14}{tot_states:>8}{tot_distinct:>10}{len(all_games):>8}{g_dup:>8.1f}{g_sb:>13.1f}")
        report["GLOBAL"] = {"states": tot_states, "distinct_snapshots": tot_distinct,
                            "distinct_games": len(all_games),
                            "dup_pct": round(g_dup, 2),
                            "same_battle_pct": round(g_sb, 2)}
        out = output_dir / "collection_diag.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved {out}")
        return

    # Phase 2: MC rollouts
    if args.batched:
        roll_device = torch.device("cpu")
        agent.to(roll_device)
        run_all_rollouts_batched(states, agent, args.k_rollouts, roll_device,
                                 batch_size=args.batch_size)
        agent.to(device)
    else:
        run_all_rollouts(states, agent, args.k_rollouts, device, n_workers=args.n_workers)

    # Phase 3: analyse
    results = analyse(states, args.k_rollouts)
    print_summary(results)

    # Save JSON (without raw arrays to keep it readable; raw is included separately)
    summary = {k: v for k, v in results.items() if k != "raw"}
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "results_full.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")

    # Plots
    make_plots(results, output_dir)


if __name__ == "__main__":
    main()
