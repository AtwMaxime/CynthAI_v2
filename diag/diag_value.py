"""
diag_value.py — Diagnostic des predictions V(s) du critic.

Usage:
    python diag/diag_value.py --checkpoint checkpoints/cheater_v5/agent_000400.pt
    python diag/diag_value.py --checkpoint checkpoints/cheater_v5/agent_000400.pt --n_games 50 --threshold 5.0
"""

import sys
import random
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.agent import CynthAIAgent
from model.backbone import K_TURNS
from model.embeddings import collate_features, collate_field_features, FIELD_DIM
from env.state_encoder import MOVE_INDEX, TYPE_INDEX, UNK
from env.action_space import MECH_NONE, MECH_TERA
from training.rollout import BattleWindow, build_action_mask, action_to_choice, encode_state
import simulator


def get_value(agent, window, state, side_idx, device):
    poke_feats, field_feat = encode_state(state, side_idx=side_idx)
    window.push(poke_feats, field_feat)
    poke_turns, field_turns = window.as_padded()

    flat = []
    for turn in poke_turns:
        flat.extend(turn)

    pb = collate_features([flat]).to(device)
    ft = collate_field_features(field_turns).field.reshape(1, K_TURNS, FIELD_DIM).to(device)
    mask = build_action_mask(state, side_idx).unsqueeze(0).to(device)

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
    mech_id   = MECH_NONE if tera_used else MECH_TERA
    tera_str  = active.get("tera_type") or ""
    mech_type = TYPE_INDEX.get(tera_str.lower(), UNK) if tera_str else UNK

    with torch.no_grad():
        out = agent(
            poke_batch        = pb,
            field_tensor      = ft,
            move_idx          = torch.tensor([midx], dtype=torch.long,    device=device),
            pp_ratio          = torch.tensor([ppr],  dtype=torch.float32, device=device),
            move_disabled     = torch.tensor([mdis], dtype=torch.float32, device=device),
            mechanic_id       = torch.tensor([mech_id],  dtype=torch.long, device=device),
            mechanic_type_idx = torch.tensor([mech_type], dtype=torch.long, device=device),
            action_mask       = mask,
        )
    return out.value.item()


def _pick_choice(state, side_idx):
    """Pick a random legal action for side_idx, handling forced switches."""
    mask = build_action_mask(state, side_idx)
    legal = [i for i in range(13) if not mask[i]]
    if not legal:
        # Fallback: try switch slots explicitly
        legal = list(range(8, 13))
    return action_to_choice(random.choice(legal), state, side_idx)


def run_battle(agent, device):
    seed = torch.randint(0, 2**31, ()).item()
    battle = simulator.PyBattle("gen9randombattle", seed)
    window0 = BattleWindow()
    records = []
    turn = 0
    last_n_own, last_n_opp = 6, 6

    while True:
        state = battle.get_state()
        if state.get("ended") or battle.ended:
            break

        side0 = state["sides"][0]
        side1 = state["sides"][1]
        hp_own = sum(p["hp"] / max(p["maxhp"], 1) for p in side0["pokemon"]) / 6
        hp_opp = sum(p["hp"] / max(p["maxhp"], 1) for p in side1["pokemon"]) / 6
        n_own  = sum(1 for p in side0["pokemon"] if not p.get("fainted"))
        n_opp  = sum(1 for p in side1["pokemon"] if not p.get("fainted"))
        last_n_own, last_n_opp = n_own, n_opp

        value = get_value(agent, window0, state, 0, device)

        records.append({
            "turn": turn, "value": value,
            "hp_own": hp_own, "hp_opp": hp_opp,
            "n_own": n_own, "n_opp": n_opp,
        })

        try:
            choice_p1 = _pick_choice(state, 0)
            choice_p2 = _pick_choice(state, 1)
            battle.make_choices(choice_p1, choice_p2)
        except Exception:
            break

        turn += 1
        if turn > 300:
            break

    # Infer outcome from last known Pokémon counts
    if last_n_own > last_n_opp:
        outcome = 1
    elif last_n_opp > last_n_own:
        outcome = -1
    else:
        outcome = 0
    return records, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_games",   type=int,   default=10)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--tail",      type=int,   default=20,
                        help="Nb de derniers tours pour le plot trajectoire alignée")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt   = torch.load(args.checkpoint, map_location=device, weights_only=True)
    agent  = CynthAIAgent(use_independent_critic=True, critic_n_layers=2).to(device)
    agent.load_state_dict(ckpt["model"], strict=False)
    agent.eval()

    print(f"Checkpoint: {args.checkpoint}  (update {ckpt.get('update','?')})")
    print(f"Running {args.n_games} games...")

    all_records  = []
    game_records = []
    outcomes     = []

    for g in range(args.n_games):
        recs, outcome = run_battle(agent, device)
        game_records.append(recs)
        outcomes.append(outcome)
        all_records.extend(recs)
        vmin = min(r["value"] for r in recs)
        vmax = max(r["value"] for r in recs)
        tag = {1: "WIN", -1: "LOSS", 0: "DRAW"}[outcome]
        print(f"  game {g+1:2d}: {len(recs):3d} steps  V=[{vmin:.2f}, {vmax:.2f}]  {tag}")

    all_v = np.array([r["value"] for r in all_records])

    # ── Stats globales ────────────────────────────────────────────────────────
    n_win  = outcomes.count(1)
    n_loss = outcomes.count(-1)
    n_draw = outcomes.count(0)
    print(f"\nGlobal V(s) — mean={all_v.mean():.3f}  std={all_v.std():.3f}  "
          f"min={all_v.min():.3f}  max={all_v.max():.3f}")
    print(f"Outcomes — WIN={n_win}  LOSS={n_loss}  DRAW={n_draw}")
    aberrant = (np.abs(all_v) > args.threshold).sum()
    print(f"|V| > {args.threshold}: {aberrant}/{len(all_v)} steps ({100*aberrant/len(all_v):.1f}%)")

    bad = [r for r in all_records if abs(r["value"]) > args.threshold]
    if bad:
        print(f"\nAberrant states (|V| > {args.threshold}):")
        for r in bad[:20]:
            print(f"  turn={r['turn']:3d}  V={r['value']:8.2f}  "
                  f"hp_own={r['hp_own']:.2f}  hp_opp={r['hp_opp']:.2f}  "
                  f"n_own={r['n_own']}  n_opp={r['n_opp']}")

    # ── Corrélation V(s) moyen vs outcome ────────────────────────────────────
    mean_v_per_game = [np.mean([r["value"] for r in recs]) for recs in game_records]
    corr = np.corrcoef(mean_v_per_game, outcomes)[0, 1] if len(mean_v_per_game) > 1 else float("nan")
    print(f"\nCorrélation mean V(s) vs outcome: {corr:.3f}")

    # ── V(s) moyen par phase ─────────────────────────────────────────────────
    def phase(turn):
        if turn < 10:  return "early (0-9)"
        if turn < 25:  return "mid (10-24)"
        return "late (25+)"

    phases = {"early (0-9)": [], "mid (10-24)": [], "late (25+)": []}
    for r in all_records:
        phases[phase(r["turn"])].append(r["value"])
    print("\nV(s) moyen par phase:")
    for ph, vs in phases.items():
        if vs:
            print(f"  {ph}: mean={np.mean(vs):.3f}  std={np.std(vs):.3f}  n={len(vs)}")

    # ── Valeurs normales vs aberrantes ────────────────────────────────────────
    # Tous les plots sont clippés sur [-clip, +clip] pour être lisibles.
    # Les aberrants sont comptés séparément dans le texte.
    clip = args.threshold
    normal_mask = np.abs(all_v) <= clip
    all_v_clip  = np.clip(all_v, -clip, clip)
    n_aberrant  = (~normal_mask).sum()
    clip_note   = f"  [clippé à ±{clip}, {n_aberrant} aberrants exclus]"

    def clip_recs(recs):
        return [dict(r, value=np.clip(r["value"], -clip, clip)) for r in recs]

    # ── Figures ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Value diagnostic — {Path(args.checkpoint).stem}  "
                 f"(update {ckpt.get('update','?')})", fontsize=13)

    axes = []
    for i in range(1, 8):
        axes.append(fig.add_subplot(2, 4, i))

    # 1. Distribution V(s) — clippée
    ax = axes[0]
    v_normal = all_v[normal_mask]
    ax.hist(v_normal, bins=60, color="steelblue", edgecolor="white", lw=0.3)
    ax.axvline(0, color="red", ls="--", lw=1, label="V=0")
    ax.set_xlabel("V(s)"); ax.set_ylabel("Count")
    ax.set_title(f"Distribution V(s) normale\n(hors {n_aberrant} aberrants |V|>{clip})")
    ax.legend(fontsize=8)

    # 2. V(s) over time (first 5 games, clippé)
    ax = axes[1]
    colors_game = plt.cm.tab10.colors
    for g, recs in enumerate(game_records[:5]):
        col = colors_game[g % 10]
        tag = {1: "W", -1: "L", 0: "D"}[outcomes[g]]
        recs_c = clip_recs(recs)
        ax.plot([r["turn"] for r in recs_c], [r["value"] for r in recs_c],
                color=col, label=f"g{g+1} ({tag})", alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Turn"); ax.set_ylabel("V(s)")
    ax.set_title(f"V(s) au fil du temps (5 premières parties){clip_note}")
    ax.legend(fontsize=8)

    # 3. V(s) vs HP advantage (clippé)
    ax = axes[2]
    hp_adv = np.array([r["hp_own"] - r["hp_opp"] for r in all_records])
    ax.scatter(hp_adv[normal_mask], all_v[normal_mask], alpha=0.2, s=5,
               color="steelblue", label="normal")
    ax.scatter(hp_adv[~normal_mask], np.zeros(n_aberrant), alpha=0.6, s=15,
               color="orange", marker="x", label=f"aberrant (n={n_aberrant})")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("HP adv (own - opp)"); ax.set_ylabel("V(s)")
    ax.set_title("V(s) vs avantage HP")
    ax.legend(fontsize=8)

    # 4. Trajectoire alignée fin de partie (clippée, colorée win/loss)
    ax = axes[3]
    tail = args.tail
    win_trajs, loss_trajs = [], []
    for recs, outcome in zip(game_records, outcomes):
        vs = np.clip([r["value"] for r in recs], -clip, clip).tolist()
        aligned = vs[-tail:]
        x = list(range(-len(aligned), 0))
        if outcome == 1:
            win_trajs.append((x, aligned))
        elif outcome == -1:
            loss_trajs.append((x, aligned))

    for x, y in win_trajs:
        ax.plot(x, y, color="green", alpha=0.25, lw=1)
    for x, y in loss_trajs:
        ax.plot(x, y, color="red", alpha=0.25, lw=1)

    def _mean_traj(trajs):
        if not trajs:
            return None, None
        max_len = max(len(y) for _, y in trajs)
        acc = np.zeros(max_len); cnt = np.zeros(max_len)
        for _, y in trajs:
            off = max_len - len(y)
            for i, v in enumerate(y):
                acc[off + i] += v; cnt[off + i] += 1
        return range(-max_len, 0), np.where(cnt > 0, acc / cnt, np.nan)

    xs, ys = _mean_traj(win_trajs)
    if xs is not None:
        ax.plot(xs, ys, color="green", lw=2.5, label=f"WIN moy. (n={len(win_trajs)})")
    xs, ys = _mean_traj(loss_trajs)
    if xs is not None:
        ax.plot(xs, ys, color="red", lw=2.5, label=f"LOSS moy. (n={len(loss_trajs)})")

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Tours avant la fin de partie"); ax.set_ylabel("V(s)")
    ax.set_title(f"Trajectoire alignée fin de partie (last {tail} turns){clip_note}")
    ax.legend(fontsize=8)

    # 5. V(s) moyen par avantage en Pokémon (hors aberrants)
    ax = axes[4]
    poke_adv = np.array([r["n_own"] - r["n_opp"] for r in all_records])
    v_norm   = all_v[normal_mask]
    pa_norm  = poke_adv[normal_mask]
    unique_adv = sorted(set(pa_norm))
    means  = [v_norm[pa_norm == a].mean() for a in unique_adv]
    stds   = [v_norm[pa_norm == a].std()  for a in unique_adv]
    counts = [int((pa_norm == a).sum())   for a in unique_adv]
    bar_colors = ["green" if a > 0 else "red" if a < 0 else "steelblue" for a in unique_adv]
    bars = ax.bar(unique_adv, means, yerr=stds, color=bar_colors,
                  alpha=0.7, capsize=4, error_kw={"lw": 1}, width=0.6)
    for bar, c in zip(bars, counts):
        ypos = bar.get_height() + (max(stds) * 0.05 if stds else 0.02)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Avantage Pokémon (n_own - n_opp)"); ax.set_ylabel("Mean V(s)")
    ax.set_title(f"V(s) moyen par avantage Pokémon\n(hors {n_aberrant} aberrants)")

    # 6. V(s) moyen par phase de jeu (hors aberrants)
    ax = axes[5]
    phases_norm = {"early (0-9)": [], "mid (10-24)": [], "late (25+)": []}
    for r in all_records:
        if abs(r["value"]) <= clip:
            phases_norm[phase(r["turn"])].append(r["value"])
    phase_labels  = list(phases_norm.keys())
    phase_means   = [np.mean(phases_norm[p]) if phases_norm[p] else 0 for p in phase_labels]
    phase_stds    = [np.std(phases_norm[p])  if phases_norm[p] else 0 for p in phase_labels]
    phase_counts  = [len(phases_norm[p]) for p in phase_labels]
    bars = ax.bar(range(len(phase_labels)), phase_means, yerr=phase_stds,
                  color=["#4fa8e0", "#f0a040", "#e05050"],
                  capsize=4, error_kw={"lw": 1}, width=0.5, alpha=0.85)
    for bar, c in zip(bars, phase_counts):
        ypos = bar.get_height() + max(phase_stds) * 0.03 if phase_stds else 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"n={c}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(phase_labels)))
    ax.set_xticklabels(["Early\n(tours 0-9)", "Mid\n(tours 10-24)", "Late\n(tours 25+)"], fontsize=8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Mean V(s)")
    ax.set_title(f"V(s) moyen par phase\n(hors {n_aberrant} aberrants)")

    # 7. Mean V(s) par partie, colorée par outcome (hors aberrants)
    ax = axes[6]
    mean_v_clip = [np.clip(np.mean([r["value"] for r in recs]), -clip, clip)
                   for recs in game_records]
    colors_out = ["green" if o == 1 else "red" if o == -1 else "gray" for o in outcomes]
    ax.bar(range(len(mean_v_clip)), mean_v_clip, color=colors_out, alpha=0.75)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(mean_v_clip)))
    ax.set_xticklabels([f"g{i+1}" for i in range(len(game_records))], fontsize=6, rotation=45)
    ax.set_ylabel("Mean V(s) clippé")
    # Légende manuelle
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="green", label="WIN"),
                       Patch(color="red",   label="LOSS"),
                       Patch(color="gray",  label="DRAW")], fontsize=8)
    corr_clip = np.corrcoef(mean_v_clip, outcomes)[0, 1] if len(set(outcomes)) > 1 else float("nan")
    ax.set_title(f"Mean V(s) par partie  corr={corr_clip:.2f}\n(vert=WIN  rouge=LOSS  gris=DRAW)")

    plt.tight_layout()
    out = Path("diag") / f"value_diag_{Path(args.checkpoint).stem}.png"
    plt.savefig(out, dpi=120)
    print(f"\nPlot saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
