"""Quick test: run collect_rollout() to verify no crashes from Switch sub-turns."""
import sys
sys.path.insert(0, ".")

import torch
from training.rollout import collect_rollout

# Minimal agent stub that outputs valid actions
class StubAgent:
    def __init__(self, n_envs, device):
        self.n_envs = n_envs
        self.device = device

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return []

    def state_dict(self):
        return {}

    def __call__(self, poke_batch, field_tensor, move_idx, pp_ratio, move_disabled,
                 mechanic_id, mechanic_type_idx, action_mask):
        B = action_mask.shape[0]
        # Uniform logits over legal actions
        logits = torch.randn(B, 13, device=self.device)
        logits = logits.masked_fill(action_mask, -1e9)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Value head: random value
        value = torch.randn(B, 1, device=self.device) * 0.1
        from training.rollout import RandomPolicyOutput
        return RandomPolicyOutput(value=value, log_probs=log_probs)

for trial in range(10):
    device = torch.device("cpu")
    agent = StubAgent(16, device)

    buf = collect_rollout(
        agent_self=agent,
        agent_opp=agent,
        n_envs=4,          # small
        min_steps=128,     # quick
        device=device,
        max_crashes=20,
    )
    n_crashes = 0
    print(f"Trial {trial}: buffer={len(buf)} steps")
    if len(buf) == 0:
        print("  EMPTY BUFFER!")
        break
print("Done")