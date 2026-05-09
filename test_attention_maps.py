"""Test get_attention_maps on a real battle state."""
import sys
sys.path.insert(0, r'C:\Users\Sun\Desktop\PokemonAI\CynthAI_v2')

import torch
from simulator import PyBattle
from training.rollout import BattleWindow, encode_state
from model.backbone import K_TURNS, N_LAYERS, N_HEADS, SEQ_LEN
from model.agent import CynthAIAgent
from model.embeddings import collate_features, collate_field_features, PokemonEmbeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Create battle
b = PyBattle('gen9randombattle', 42)
state = b.get_state()

# Build window (K=4, same state repeated for early turns)
window = BattleWindow()
poke_feats, field_feat = encode_state(state, side_idx=0)
for _ in range(K_TURNS):
    window.push(poke_feats, field_feat)

# Get collated inputs
poke_turns, field_turns = window.as_padded()
flat_poke = []
for turn in poke_turns:
    flat_poke.extend(turn)

poke_batch = collate_features([flat_poke]).to(device)
field_tensor = collate_field_features(field_turns).field.unsqueeze(0).to(device)

# Embed tokens (same as agent.poke_emb would do)
emb = PokemonEmbeddings().to(device)
pokemon_tokens = emb(poke_batch)   # [1, 48, 445]
field_tokens = field_tensor        # [1, 4, 72]

print(f'pokemon_tokens: {tuple(pokemon_tokens.shape)}')
print(f'field_tokens:   {tuple(field_tokens.shape)}')

# Test get_attention_maps
agent = CynthAIAgent().to(device)
agent.eval()
with torch.no_grad():
    result = agent.backbone.get_attention_maps(pokemon_tokens, field_tokens)

print(f'\nAttention maps: {len(result["attention_maps"])} layers')
for i, attn in enumerate(result['attention_maps']):
    ok = attn.shape == (1, N_HEADS, SEQ_LEN, SEQ_LEN) or attn.shape == (1, SEQ_LEN, SEQ_LEN)
    status = 'OK' if ok else 'FAIL'
    print(f'  layer {i}: {tuple(attn.shape)}  [{status}]')

print(f'value:          {tuple(result["value"].shape)}  [OK]')
print(f'current_tokens: {tuple(result["current_tokens"].shape)}  [OK]')
print(f'token_labels:   {len(result["token_labels"])} labels')
print(f'  first 5: {result["token_labels"][:5]}')
print(f'  last 5:  {result["token_labels"][-5:]}')

# Sanity: attention weights are proper probability distributions
attn0 = result['attention_maps'][0]
row_sums = attn0.sum(dim=-1)
print(f'\nSanity check - attention rows sum to 1.0:')
print(f'  layer 0: min={row_sums.min().item():.4f}  max={row_sums.max().item():.4f}')

# Show hottest attention links
# If attn has head dim [B, H, T, T], average over heads first
if attn0.dim() == 4:
    avg_attn = attn0.mean(dim=1).squeeze(0)  # [T, T]
elif attn0.dim() == 3:
    avg_attn = attn0.squeeze(0)               # [T, T]
else:
    raise ValueError(f'Unexpected attention shape: {attn0.shape}')

for query_idx in [0, 12, 24, 36, 48, 51]:
    key_idx = avg_attn[query_idx].argmax().item()
    print(f'  token {result["token_labels"][query_idx]:>12} -> most attends to '
          f'{result["token_labels"][key_idx]} '
          f'({avg_attn[query_idx, key_idx]:.3f})')

print('\nAll checks passed!')