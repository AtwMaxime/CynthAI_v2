"""Quick test: check request_state in simulator state dict"""
import sys
sys.path.insert(0, "C:\\Users\\Sun\\Desktop\\PokemonAI\\CynthAI_v2")
from simulator import PyBattle

b = PyBattle("gen9randombattle", 42)
s = b.get_state()
for i, side in enumerate(s["sides"]):
    print(f"Side {i}: request_state = {side.get('request_state', 'KEY NOT FOUND')}")