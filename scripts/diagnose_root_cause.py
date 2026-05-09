"""
Diagnostic ciblé : comprendre pourquoi Rust rejette des choix valides selon notre mask.

Hypothèse : après un KO sur le tour N-1, le Rust sim attend un switch pour le side concerné
au tour N, mais notre make_choices() envoie deux moves.

Stratégie : avant chaque make_choices, loguer l'état complet pour comparer
ce qu'on envoie vs ce que le Rust attend.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import torch
from simulator import PyBattle
from training.rollout import (
    build_action_mask, action_to_choice, RandomPolicy,
    _build_agent_inputs, BattleWindow, encode_state, _sample_action,
)
from model.backbone import K_TURNS

device = torch.device("cpu")
N_ENVS = 8
MAX_STEPS = 500
format_id = "gen9randombattle"
next_seed = 10000

envs = [PyBattle(format_id, seed=s) for s in range(N_ENVS)]
wins = [BattleWindow() for _ in range(N_ENVS)]
prev_states = [e.get_state() for e in envs]
for i, s in enumerate(prev_states):
    pf, ff = encode_state(s, 0); wins[i].push(pf, ff)

rand = RandomPolicy()
n_crashes = 0

def dump_poke(poke: dict, idx: int) -> str:
    mvs = [(m.get("id","?"), m.get("pp","?"), m.get("disabled",False)) for m in poke.get("moves",[])[:4]]
    sid = str(poke.get('species_id','?'))
    return (f"    poke[{idx}]: {sid:20s} "
            f"hp={str(poke.get('hp','?')):>4s}/{str(poke.get('maxhp','?')):>4s} "
            f"fainted={bool(poke.get('fainted',False))} "
            f"force_sw={bool(poke.get('force_switch_flag',False))} "
            f"trapped={bool(poke.get('trapped',False))} "
            f"status={poke.get('status','') or '-'} "
            f"moves={mvs}")

def dump_side(side, label: str) -> list[str]:
    lines = [f"  {label}: active={[a for a in side['active'] if a is not None]}  "
             f"req_state={side.get('request_state','?')}"]
    for pi, poke in enumerate(side["pokemon"]):
        lines.append(dump_poke(poke, pi))
    return lines

with torch.no_grad():
    for step in range(1, MAX_STEPS + 1):
        ins = _build_agent_inputs(wins, prev_states, 0, device)
        masks = torch.stack([build_action_mask(s, 0) for s in prev_states]).to(device)
        out = rand(*ins, masks)
        acts = _sample_action(out.log_probs, masks)

        curr_states = []
        for i in range(N_ENVS):
            fresh_state = envs[i].get_state()

            # Side 0 (self): re-valider
            fresh_mask = build_action_mask(fresh_state, 0)
            a = acts[i].item()
            if fresh_mask[a]:
                legal = (~fresh_mask).nonzero().squeeze(-1)
                a = legal[torch.randint(0, len(legal), (1,))].item() if len(legal) > 0 else 0

            # Side 1 (opp): action aléatoire depuis mask frais
            fresh_mask_opp = build_action_mask(fresh_state, 1)
            legal_opp = (~fresh_mask_opp).nonzero().squeeze(-1)
            a_opp = legal_opp[torch.randint(0, len(legal_opp), (1,))].item() if len(legal_opp) > 0 else 0

            c = action_to_choice(a, fresh_state, 0)
            c_opp = action_to_choice(a_opp, fresh_state, 1)

            try:
                envs[i].make_choices(c, c_opp)
                curr_states.append(envs[i].get_state())
            except BaseException:
                n_crashes += 1
                print(f"\n=== CRASH #{n_crashes}  step={step} env={i} ===")
                print(f"  p1={c!r:30s} (slot {a})  |  p2={c_opp!r:30s} (slot {a_opp})")

                # État FRAIS (celui utilisé pour la re-validation)
                print(f"\n  --- fresh_state (get_state() avant make_choices) ---")
                for si in [0, 1]:
                    for line in dump_side(fresh_state["sides"][si], f"Side {si}"):
                        print(line)

                # État PRÉCÉDENT (prev_states, du tour d'avant)
                print(f"\n  --- prev_state (tour précédent) ---")
                for si in [0, 1]:
                    for line in dump_side(prev_states[i]["sides"][si], f"Side {si}"):
                        print(line)

                # Comparaison : qu'est-ce qui a changé entre prev et fresh ?
                print(f"\n  --- Diff prev -> fresh ---")
                for si in [0, 1]:
                    prev = prev_states[i]["sides"][si]
                    fresh = fresh_state["sides"][si]
                    prev_fnt = sum(1 for p in prev["pokemon"] if p.get("fainted"))
                    fresh_fnt = sum(1 for p in fresh["pokemon"] if p.get("fainted"))
                    if fresh_fnt > prev_fnt:
                        print(f"  Side {si}: +{fresh_fnt - prev_fnt} faint(s) depuis le tour précédent")
                    if fresh["total_fainted"] != prev["total_fainted"]:
                        print(f"  Side {si}: total_fainted {prev['total_fainted']} → {fresh['total_fainted']}")

                sys.stdout.flush()

                # Reset
                envs[i] = PyBattle(format_id, seed=next_seed)
                next_seed += 1
                wins[i].reset()
                curr_states.append(envs[i].get_state())

        for i, s in enumerate(curr_states):
            if envs[i].ended:
                envs[i] = PyBattle(format_id, seed=next_seed)
                next_seed += 1
                wins[i].reset()
                s = envs[i].get_state()
                curr_states[i] = s
            pf, ff = encode_state(s, 0); wins[i].push(pf, ff)
        prev_states = curr_states

        if step % 100 == 0:
            print(f"  step {step}  crashes={n_crashes}")
            sys.stdout.flush()

print(f"\nTotal crashes: {n_crashes} / {MAX_STEPS * N_ENVS} env-steps")