"""
Visualise les attention maps du Transformer — heatmap par tête + top-5 tokens scrutés.

Usage:
    python scripts/viz_attention.py [--checkpoint checkpoints/agent_best.pt] [--seed 42]

Output:
    saves plots/attention_L{layer}_H{head}.png for each (layer, head)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator import PyBattle
from training.rollout import BattleWindow, encode_state, build_action_mask
from model.backbone import K_TURNS, N_LAYERS, N_HEADS, SEQ_LEN
from model.agent import CynthAIAgent
from model.embeddings import collate_features, collate_field_features, PokemonEmbeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Run a few battle steps to get varied state ---
b = PyBattle("gen9randombattle", 42)
state = b.get_state()
for step in range(6):
    mask = build_action_mask(state, side_idx=0)
    if mask[0:4].all():   # all move slots illegal -> must switch
        break
    try:
        b.make_choices("move 1", "move 1")
        state = b.get_state()
    except Exception:
        break

window = BattleWindow()
poke_feats, field_feat = encode_state(state, side_idx=0)
for _ in range(K_TURNS):
    window.push(poke_feats, field_feat)

poke_turns, field_turns = window.as_padded()
flat_poke = []
for turn in poke_turns:
    flat_poke.extend(turn)

poke_batch = collate_features([flat_poke]).to(device)
field_tensor = collate_field_features(field_turns).field.unsqueeze(0).to(device)

emb = PokemonEmbeddings().to(device)
pokemon_tokens = emb(poke_batch)
field_tokens = field_tensor

agent = CynthAIAgent().to(device)
agent.eval()
with torch.no_grad():
    result = agent.backbone.get_attention_maps(pokemon_tokens, field_tokens)

token_labels = result["token_labels"]
attn_maps    = result["attention_maps"]  # list of [1, H, 52, 52]

out_dir = Path(__file__).parent.parent / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Heatmap per layer + head ---
FIGSIZE = 14
for layer_idx, attn in enumerate(attn_maps):
    # attn: [1, H, 52, 52]
    attn_np = attn.squeeze(0).numpy()  # [H, 52, 52]

    for head_idx in range(N_HEADS):
        mat = attn_np[head_idx]  # [52, 52]

        fig, (ax_heat, ax_bar) = plt.subplots(
            1, 2, figsize=(FIGSIZE, 6),
            gridspec_kw={"width_ratios": [1, 0.03]}
        )

        vmax = mat.max()
        im = ax_heat.imshow(mat, cmap="viridis", aspect="equal", vmin=0.0, vmax=vmax)
        ax_heat.set_title(f"Layer {layer_idx} — Head {head_idx}", fontsize=13)
        ax_heat.set_xlabel("Key token", fontsize=10)
        ax_heat.set_ylabel("Query token", fontsize=10)

        ax_heat.set_xticks(range(SEQ_LEN))
        ax_heat.set_yticks(range(SEQ_LEN))
        ax_heat.set_xticklabels(token_labels, fontsize=5, rotation=90)
        ax_heat.set_yticklabels(token_labels, fontsize=5)

        fig.colorbar(im, cax=ax_bar)

        # --- Top-5 per current-turn query ---
        ax_side = fig.add_axes([0.92, 0.12, 0.15, 0.76])
        ax_side.axis("off")

        cur_start = SEQ_LEN - 13
        topk = 5
        y_pos = 0
        ax_side.text(0, 1.02, "Top-5 attended keys\n(current turn queries)", fontsize=9,
                     fontweight="bold", transform=ax_side.transAxes)

        for query_offset in range(13):
            query_idx = cur_start + query_offset
            q_label   = token_labels[query_idx]
            row       = mat[query_idx]
            top_indices = np.argsort(row)[-topk:][::-1]

            short_q = q_label.split("_", 1)[1] if "_" in q_label else q_label
            keys_str = " ".join(
                lbl.split("_", 1)[1] if "_" in lbl else lbl
                for lbl in [token_labels[i] for i in top_indices]
            )
            ax_side.text(0, y_pos, f"{short_q:>10} -> {keys_str}", fontsize=5.5,
                         family="monospace", transform=ax_side.transAxes)
            y_pos -= 0.065

        fname = out_dir / f"attention_L{layer_idx}_H{head_idx}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> saved {fname.name}")

# --- Per-head average attention per turn ---
print("\n--- Average attention weight per turn (per head) ---")
for layer_idx in range(N_LAYERS):
    attn_np = attn_maps[layer_idx].squeeze(0).numpy()  # [H, 52, 52]
    print(f"\n  Layer {layer_idx}:")
    turn_labels = [f"T{t}" for t in range(K_TURNS)]
    header = "        " + "  ".join(f"{l:>6}" for l in turn_labels)
    for head_idx in range(N_HEADS):
        mat = attn_np[head_idx]
        turn_attn = np.zeros((K_TURNS, K_TURNS))
        for q_t in range(K_TURNS):
            for k_t in range(K_TURNS):
                q_slice = slice(q_t * 13, (q_t + 1) * 13)
                k_slice = slice(k_t * 13, (k_t + 1) * 13)
                turn_attn[q_t, k_t] = mat[q_slice, k_slice].mean()
        print(f"    Head {head_idx}:")
        for q_t in range(K_TURNS):
            row_vals = "  ".join(f"{turn_attn[q_t, k_t]:.4f}" for k_t in range(K_TURNS))
            print(f"      {turn_labels[q_t]}  {row_vals}")

# --- Head specialisation summary ---
print("\n--- Head specialisation: strongest token pair per head ---")
for layer_idx in range(N_LAYERS):
    attn_np = attn_maps[layer_idx].squeeze(0).numpy()
    print(f"\n  Layer {layer_idx}:")
    for head_idx in range(N_HEADS):
        mat = attn_np[head_idx]
        flat_idx = mat.reshape(-1).argmax()
        q_i, k_i = divmod(flat_idx, SEQ_LEN)
        print(f"    Head {head_idx}: max={mat[q_i, k_i]:.4f}  "
              f"{token_labels[q_i]} -> {token_labels[k_i]}")

print("\nDone!")