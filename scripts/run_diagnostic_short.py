"""Short diagnostic runner: 250 steps with request_state visible"""
import sys
sys.path.insert(0, "C:\\Users\\Sun\\Desktop\\PokemonAI\\CynthAI_v2")
with open("C:\\Users\\Sun\\Desktop\\PokemonAI\\CynthAI_v2\\scripts\\diagnose_root_cause.py") as f:
    code = f.read()
code = code.replace("MAX_STEPS = 500", "MAX_STEPS = 250")
exec(code)