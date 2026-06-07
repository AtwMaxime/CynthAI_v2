"""
experiments/diag_critic_blowup.py — Diagnostic de la pathologie de "blowup" du critic

Contexte
--------
L'expérience estimate_ev_max (cheater_v7, agent_001000.pt) a révélé un R²
catastrophique (-89425) causé par ~2 états sur 320 où la tête value sort une
valeur géante (~2086) totalement décorrélée du retour MC. critic_value_bound
(tanh ±10) MASQUE le symptôme en sortie mais ne corrige pas la cause.

Ce script (lecture seule, déterministe, seed 42, SANS rollouts MC) :

  Phase 1 — Robustesse
    - reproduit EXACTEMENT les mêmes états que l'EV exp (collect_states, seed 42),
    - calcule v_theta brut (agent instancié avec critic_value_bound=0),
    - isole les états de "blowup" (|v_theta| > --threshold),
    - pour chacun : stats des tenseurs d'entrée (pb/ft : min/max/NaN/inf, midx,
      mech_id, mech_type, # de tours réels vs padding dans la fenêtre K-turns),
    - hooks forward sur chaque sous-module du critic -> max_abs / mean_abs / std
      de l'activation, pour localiser à QUELLE couche la magnitude explose.

  Phase 2.5 — Évolution à travers l'entraînement
    - sur les MÊMES états de blowup (figés, collectés une fois),
    - évalue une liste de checkpoints (--updates),
    - mesure raw_value (pré-tanh) et bounded_value = bound*tanh(raw/bound),
    - => montre si l'entraînement RÉSORBE la pathologie (raw redescend) ou si le
      bound ne fait que la MASQUER (raw reste géant, bounded plat à ±bound).

Usage
-----
    python experiments/diag_critic_blowup.py \\
        --ckpt-dir checkpoints/cheater_v7 \\
        --collect-update 1000 \\
        --updates 300,600,1000,1400,1700 \\
        --n-per-bucket 80 --threshold 5.0 --bound 10.0 \\
        --output-dir experiments/results/critic_blowup_v7
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # pour importer estimate_ev_max

from model.agent import CynthAIAgent
from model.backbone import K_TURNS

# Réutilise la collecte d'états et le builder d'inputs de l'expérience EV
from estimate_ev_max import (
    collect_states, _get_agent_inputs, sample_action, TURN_BUCKETS, FORMAT_ID,
)
from training.rollout import (
    BattleWindow, build_action_mask, action_to_choice, encode_state,
)
from env.team_pool import sample_teams


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_agent(checkpoint: str, device: torch.device, value_bound: float = 0.0) -> CynthAIAgent:
    """Charge un agent. value_bound=0 => la tête critic renvoie le raw (pré-tanh)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    agent = CynthAIAgent(critic_n_layers=2, critic_detach=True,
                         critic_value_bound=value_bound)
    agent.load_state_dict(ckpt["model"], strict=False)
    agent.eval()
    agent.to(device)
    return agent


def collect_states_fast(agent, n_per_bucket, device, seed_start=0, save_path=None):
    """
    Collecteur RAPIDE : plusieurs snapshots par bataille (vs 1 seul dans l'EV exp).
    Supprime le facteur ~10x de parties jetées. États NON indépendants (corrélés
    intra-bataille) — acceptable pour un diag de blowup. Sauvegarde incrémentale.

    Retourne une liste légère de dicts {idx, turn, game_idx, state, window}.
    """
    from simulator import PyBattle

    buckets = {b[2]: 0 for b in TURN_BUCKETS}
    def bucket_open(t):
        nm = _bucket_for_turn(t)
        return nm if (nm != "?" and buckets[nm] < n_per_bucket) else None
    def all_full():
        return all(v >= n_per_bucket for v in buckets.values())

    lw = []
    seed = seed_start
    n_games = 0

    while not all_full():
        t1, t2 = sample_teams()
        battle = PyBattle.from_packed_teams(FORMAT_ID, seed, t1, t2)
        seed += 1
        n_games += 1

        win_self, win_opp = BattleWindow(), BattleWindow()  # up-to-PREVIOUS state

        for _ in range(200):
            if battle.ended:
                break
            state = battle.get_state()
            turn = state.get("turn", 0)
            req_self = state["sides"][0].get("request_state", "")
            req_opp  = state["sides"][1].get("request_state", "")

            # Snapshot (côté 0) si décision réelle + bucket ouvert.
            # win_self = états jusqu'au tour PRÉCÉDENT ; raw_value poussera `state`.
            if req_self != "None":
                nm = bucket_open(turn)
                if nm is not None:
                    lw.append({"idx": len(lw), "turn": turn, "game_idx": n_games,
                               "state": state, "window": copy.deepcopy(win_self)})
                    buckets[nm] += 1

            # Échantillonne les actions (deepcopy -> ne pas muter win_self)
            mask_self = build_action_mask(state, 0)
            mask_opp  = build_action_mask(state, 1)
            if req_self == "None" or mask_self.all():
                a_self_str = "default"
            else:
                a = sample_action(agent, copy.deepcopy(win_self), state, 0, device)
                if mask_self[a]:
                    legal = [i for i in range(13) if not mask_self[i]]
                    a = random.choice(legal) if legal else 0
                a_self_str = action_to_choice(a, state, 0)
            if req_opp == "None" or mask_opp.all():
                a_opp_str = "default"
            else:
                a = sample_action(agent, copy.deepcopy(win_opp), state, 1, device)
                if mask_opp[a]:
                    legal = [i for i in range(13) if not mask_opp[i]]
                    a = random.choice(legal) if legal else 0
                a_opp_str = action_to_choice(a, state, 1)

            # Avance la fenêtre : pousse l'état courant une seule fois
            pf, ff = encode_state(state, 0); win_self.push(pf, ff)
            pf, ff = encode_state(state, 1); win_opp.push(pf, ff)

            if not battle.make_choices(a_self_str, a_opp_str):
                battle.make_choices("default", "default")

        if save_path is not None and (all_full() or n_games % 10 == 0):
            torch.save(lw, save_path)

    if save_path is not None:
        torch.save(lw, save_path)
    print(f"[Fast collect] {len(lw)} états depuis {n_games} parties "
          f"({len(lw)/max(n_games,1):.1f} états/partie). Buckets: {buckets}")
    return lw


def _bucket_for_turn(t: int) -> str:
    for lo, hi, name in TURN_BUCKETS:
        if lo <= t <= hi:
            return name
    return "?"


def raw_value(agent: CynthAIAgent, state: dict, window, device: torch.device) -> float:
    """v_theta brut (window non mutée — deepcopy à chaque appel)."""
    win = copy.deepcopy(window)
    inputs = _get_agent_inputs(win, state, side_idx=0, device=device)
    with torch.no_grad():
        out = agent(*inputs)
    return float(out.value.item())


def tensor_stats(t: torch.Tensor) -> dict:
    tf = t.detach().float()
    return {
        "shape": list(t.shape),
        "min": round(float(tf.min()), 4),
        "max": round(float(tf.max()), 4),
        "max_abs": round(float(tf.abs().max()), 4),
        "mean_abs": round(float(tf.abs().mean()), 4),
        "std": round(float(tf.std()), 4),
        "n_nan": int(torch.isnan(tf).sum()),
        "n_inf": int(torch.isinf(tf).sum()),
    }


def input_report(state: dict, window, turn: int, game_idx: int, device: torch.device) -> dict:
    """Stats des tenseurs d'entrée + fenêtre K-turns pour un état de blowup."""
    win = copy.deepcopy(window)
    pb, ft, midx, ppr, mdis, mech_id, mech_type, mask = \
        _get_agent_inputs(win, state, side_idx=0, device=device)

    # Tours réels vs padding : une "ligne" de field tout-à-zéro = padding (cf critic.py:86)
    field_turn_norm = ft.reshape(K_TURNS, -1).abs().sum(dim=-1)  # [K]
    real_turns = int((field_turn_norm >= 1e-6).sum())

    # pb est un PokemonBatch (dataclass) : on rapporte les scalars (floats non bornés)
    # + les max des index d'embedding (détection d'index hors-plage).
    idx_maxes = {
        nm: int(getattr(pb, nm).max())
        for nm in ("species_idx", "type1_idx", "type2_idx", "tera_idx",
                   "item_idx", "ability_idx", "move_idx")
    }

    return {
        "turn": turn,
        "bucket": _bucket_for_turn(turn),
        "game_idx": game_idx,
        "pb_scalars": tensor_stats(pb.scalars),
        "pb_idx_max": idx_maxes,
        "ft": tensor_stats(ft),
        "field_real_turns": real_turns,
        "field_padded_turns": K_TURNS - real_turns,
        "move_idx": midx.flatten().tolist(),
        "mech_id": int(mech_id.item()),
        "mech_type": int(mech_type.item()),
        "pp_ratio": [round(x, 3) for x in ppr.flatten().tolist()],
    }


def layer_activation_report(agent: CynthAIAgent, state: dict, window, device: torch.device) -> dict:
    """
    Hooks forward sur les sous-modules du critic. Pour chaque couche :
    max_abs / mean_abs / std de la sortie -> propagation de la magnitude.
    """
    vh = agent.value_head
    targets = {}
    if vh.n_layers > 0:
        targets["transformer"] = vh.transformer
    targets["value_head.0"] = vh.value_head[0]   # Linear D->D
    targets["value_head.1"] = vh.value_head[1]   # ReLU
    targets["value_head.2"] = vh.value_head[2]   # Linear D->1 (= raw value, pré-tanh)

    captured: dict[str, dict] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o):
                of = o.detach().float()
                captured[name] = {
                    "max_abs":  round(float(of.abs().max()), 4),
                    "mean_abs": round(float(of.abs().mean()), 4),
                    "std":      round(float(of.std()), 4),
                }
        return hook

    for name, mod in targets.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        win = copy.deepcopy(window)
        inputs = _get_agent_inputs(win, state, side_idx=0, device=device)
        with torch.no_grad():
            agent(*inputs)
    finally:
        for h in handles:
            h.remove()

    return captured


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Diagnostic critic blowup")
    ap.add_argument("--ckpt-dir", default="checkpoints/cheater_v7",
                    help="Répertoire des checkpoints agent_NNNNNN.pt")
    ap.add_argument("--collect-update", type=int, default=1000,
                    help="Update dont la politique sert à reproduire les états (= EV exp)")
    ap.add_argument("--updates", default="300,600,1000,1400,1700",
                    help="Liste d'updates à évaluer en Phase 2.5")
    ap.add_argument("--n-per-bucket", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="|v_theta| au-dessus duquel un état est un 'blowup'")
    ap.add_argument("--bound", type=float, default=10.0,
                    help="value_bound pour recalculer bounded_value = bound*tanh(raw/bound)")
    ap.add_argument("--output-dir", default="experiments/results/critic_blowup_v7")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-cache", action="store_true",
                    help="Recharge states_cache.pt au lieu de re-collecter (retouche analyse)")
    ap.add_argument("--fast", action="store_true",
                    help="Collecteur multi-snapshots/bataille (~10x moins de parties)")
    args = ap.parse_args()

    # Reproductibilité EXACTE de l'EV exp : seeds AVANT collect_states
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.ckpt_dir)
    def ckpt_path(u: int) -> Path:
        return ckpt_dir / f"agent_{u:06d}.pt"

    updates = [int(x) for x in args.updates.split(",") if x.strip()]

    print(f"Device: {device}")
    print(f"Collect policy: {ckpt_path(args.collect_update)} (update {args.collect_update})")
    print(f"Updates évalués (Phase 2.5): {updates}")
    print(f"n_per_bucket={args.n_per_bucket}  threshold={args.threshold}  bound={args.bound}\n")

    # ── Phase 1 : reproduire les états avec la politique de collect_update ──
    # Cache léger : on ne garde que (state dict, window, turn) — picklable,
    # contrairement au battle_fork. Permet de retoucher l'analyse sans re-simuler.
    cache_path = out_dir / "states_cache.pt"
    collect_agent = load_agent(str(ckpt_path(args.collect_update)), device, value_bound=0.0)

    if args.load_cache and cache_path.exists():
        print(f"Phase 1 — chargement du cache {cache_path} (pas de re-collecte)...")
        lw = torch.load(cache_path, weights_only=False)
    elif args.fast:
        print("Phase 1 — collecte RAPIDE (multi-snapshots/bataille, seed 42)...")
        lw = collect_states_fast(collect_agent, args.n_per_bucket, device,
                                 save_path=cache_path)
        print(f"  cache sauvegardé -> {cache_path}")
    else:
        print("Phase 1 — collecte des états (déterministe, seed 42)...")
        states = collect_states(collect_agent, args.n_per_bucket, device)
        lw = [{"idx": i, "turn": s.turn, "game_idx": s.game_idx,
               "state": s.battle_fork.get_state(), "window": s.window}
              for i, s in enumerate(states)]
        torch.save(lw, cache_path)
        print(f"  cache sauvegardé -> {cache_path}")

    # v_theta brut + isolation des blowups
    raws = np.array([raw_value(collect_agent, e["state"], e["window"], device) for e in lw])
    blow_idx = [i for i, v in enumerate(raws) if abs(v) > args.threshold]
    print(f"\n# états collectés : {len(lw)}")
    print(f"# états de blowup (|v_theta|>{args.threshold}) : {len(blow_idx)}")
    for i in blow_idx:
        e = lw[i]
        print(f"  idx={i:3d}  turn={e['turn']:3d}  bucket={_bucket_for_turn(e['turn']):<10}"
              f"  v_theta={raws[i]:12.2f}")

    # ── Phase 1 (suite) : inputs + activations couche par couche ──
    diag = {
        "config": {
            "collect_update": args.collect_update, "updates": updates,
            "n_per_bucket": args.n_per_bucket, "threshold": args.threshold,
            "bound": args.bound, "seed": args.seed,
            "n_states": len(lw), "n_blowup": len(blow_idx),
        },
        "blowup_states": [],
    }
    for i in blow_idx:
        e = lw[i]
        diag["blowup_states"].append({
            "idx": i,
            "v_theta_collect": round(float(raws[i]), 4),
            "input": input_report(e["state"], e["window"], e["turn"], e["game_idx"], device),
            "layer_activations": layer_activation_report(collect_agent, e["state"], e["window"], device),
        })

    # ── Phase 2.5 : évolution raw/bounded sur les mêmes états, par checkpoint ──
    print("\nPhase 2.5 — évolution à travers les checkpoints...")
    evolution = {}  # update -> list[{idx, turn, raw_value, bounded_value}]
    b = args.bound
    for u in updates:
        p = ckpt_path(u)
        if not p.exists():
            print(f"  [skip] {p} introuvable")
            continue
        ag = load_agent(str(p), device, value_bound=0.0)
        rows = []
        for i in blow_idx:
            r = raw_value(ag, lw[i]["state"], lw[i]["window"], device)
            bounded = b * float(np.tanh(r / b)) if b > 0 else r
            rows.append({"idx": i, "turn": lw[i]["turn"],
                         "raw_value": round(r, 4), "bounded_value": round(bounded, 4)})
        evolution[str(u)] = rows
        del ag

    diag["evolution"] = evolution

    # Table récap : raw_value par état × update
    print("\n=== raw_value par état de blowup × update ===")
    header = "  idx  turn | " + " | ".join(f"u{u:>5}" for u in updates)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in blow_idx:
        cells = []
        for u in updates:
            rows = evolution.get(str(u))
            val = next((r["raw_value"] for r in rows if r["idx"] == i), None) if rows else None
            cells.append(f"{val:>6.1f}" if val is not None else "   n/a")
        print(f"  {i:3d}  {lw[i]['turn']:4d} | " + " | ".join(cells))

    out_json = out_dir / "critic_blowup_diag.json"
    with open(out_json, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
