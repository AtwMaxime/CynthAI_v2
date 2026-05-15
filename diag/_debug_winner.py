import sys, torch, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.rollout import BattleWindow, build_action_mask, action_to_choice, encode_state
import simulator

for trial in range(10):
    seed = torch.randint(0, 2**31, ()).item()
    battle = simulator.PyBattle("gen9randombattle", seed)
    turn = 0
    while True:
        state = battle.get_state()
        if state.get("ended") or battle.ended:
            break
        mask0 = build_action_mask(state, 0)
        mask1 = build_action_mask(state, 1)
        legal  = [i for i in range(13) if not mask0[i]]
        legal1 = [i for i in range(13) if not mask1[i]]
        if not legal or not legal1:
            break
        battle.make_choices(
            action_to_choice(random.choice(legal), state, 0),
            action_to_choice(random.choice(legal1), state, 1),
        )
        turn += 1
        if turn > 200:
            break

    ended = battle.ended
    winner = battle.winner
    state = battle.get_state()
    state_ended = state.get("ended")
    alive0 = sum(1 for p in state["sides"][0]["pokemon"] if not p.get("fainted"))
    alive1 = sum(1 for p in state["sides"][1]["pokemon"] if not p.get("fainted"))
    print(f"trial={trial} turns={turn} battle.ended={ended} battle.winner={winner!r} "
          f"state.ended={state_ended} alive0={alive0} alive1={alive1}")
