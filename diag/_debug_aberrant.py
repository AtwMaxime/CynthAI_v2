import sys, torch, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.agent import CynthAIAgent
from model.backbone import K_TURNS
from model.embeddings import collate_features, collate_field_features, FIELD_DIM
from env.state_encoder import MOVE_INDEX, TYPE_INDEX, UNK
from env.action_space import MECH_NONE, MECH_TERA
from training.rollout import BattleWindow, build_action_mask, action_to_choice, encode_state
import simulator

device = torch.device("cpu")
ckpt = torch.load("checkpoints/cheater_v5/agent_000400.pt", map_location=device, weights_only=True)
agent = CynthAIAgent(use_independent_critic=True, critic_n_layers=2).to(device)
agent.load_state_dict(ckpt["model"], strict=False)
agent.eval()

found = 0
for trial in range(500):
    seed = torch.randint(0, 2**31, ()).item()
    battle = simulator.PyBattle("gen9randombattle", seed)
    window0 = BattleWindow()
    for turn in range(10):
        state = battle.get_state()
        if state.get("ended") or battle.ended:
            break

        poke_feats, field_feat = encode_state(state, side_idx=0)
        window0.push(poke_feats, field_feat)
        poke_turns, field_turns = window0.as_padded()
        flat = []
        for t in poke_turns:
            flat.extend(t)
        pb = collate_features([flat]).to(device)
        ft = collate_field_features(field_turns).field.reshape(1, K_TURNS, FIELD_DIM).to(device)
        mask = build_action_mask(state, 0).unsqueeze(0).to(device)

        side = state["sides"][0]
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
            midx.append(UNK)
            ppr.append(0.0)
            mdis.append(0.0)

        tera_used = any(p.get("terastallized") is not None for p in side["pokemon"])
        mech_id = MECH_NONE if tera_used else MECH_TERA
        tera_str = active.get("tera_type") or ""
        mech_type = TYPE_INDEX.get(tera_str.lower(), UNK) if tera_str else UNK

        with torch.no_grad():
            out = agent(pb, ft,
                torch.tensor([midx], dtype=torch.long, device=device),
                torch.tensor([ppr], dtype=torch.float32, device=device),
                torch.tensor([mdis], dtype=torch.float32, device=device),
                torch.tensor([mech_id], dtype=torch.long, device=device),
                torch.tensor([mech_type], dtype=torch.long, device=device),
                mask)
        v = out.value.item()

        if abs(v) > 5:
            found += 1
            name = active.get("name", "?")
            moves = [m["id"] for m in (active.get("moves") or [])]
            print(f"trial={trial} turn={turn} V={v:.2f}")
            print(f"  active={name}  tera_type={tera_str!r}")
            print(f"  moves={moves}")
            print(f"  midx={midx}  mech_id={mech_id}  mech_type={mech_type}")
            print(f"  mask (legal slots): {[i for i in range(13) if not mask[0][i].item()]}")
            if found >= 5:
                print("(arrêt après 5 cas)")
                sys.exit(0)
            break

        mask0 = build_action_mask(state, 0)
        legal = [i for i in range(13) if not mask0[i]]
        mask1 = build_action_mask(state, 1)
        legal1 = [i for i in range(13) if not mask1[i]]
        if not legal or not legal1:
            break
        battle.make_choices(
            action_to_choice(legal[0], state, 0),
            action_to_choice(random.choice(legal1), state, 1)
        )

print(f"\nTotal aberrant: {found}/500 trials")
