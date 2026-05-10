"""
CynthAI_v2 Attention Maps — visualize Transformer attention patterns.

Usage:
    python -m training.attention_viz --checkpoint checkpoints/curriculum_max_20260510_1234/agent_000050.pt
    python -m training.attention_viz --checkpoint checkpoints/curriculum_max_20260510_1234/agent_000050.pt \
        --layer 0 --head 2 --save plots/attn_layer0_head2.png

Reads a checkpoint, runs a forward pass on random battle data, and plots
attention heatmaps showing which tokens each position attends to.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.agent import CynthAIAgent
from model.backbone import K_TURNS, N_SLOTS, SEQ_LEN
from env.state_encoder import N_SPECIES, N_MOVES, N_ITEMS, N_ABILITIES, UNK


def _dummy_batch(B: int = 4, device: str = "cpu") -> dict:
    """Create a random dummy batch for a forward pass."""
    K12 = K_TURNS * 12
    return {
        "poke_batch": type("PB", (), {
            "species_idx":  torch.randint(0, N_SPECIES,  (B, K12), device=device),
            "type1_idx":    torch.randint(0, 18,        (B, K12), device=device),
            "type2_idx":    torch.randint(0, 18,        (B, K12), device=device),
            "tera_idx":     torch.randint(0, 18,        (B, K12), device=device),
            "item_idx":     torch.randint(0, N_ITEMS,   (B, K12), device=device),
            "ability_idx":  torch.randint(0, N_ABILITIES, (B, K12), device=device),
            "move_idx":     torch.randint(0, N_MOVES,   (B, K12, 4), device=device),
            "scalars":      torch.rand(B, K12, 222, device=device),
            "to":           lambda d: _dummy_batch(B, d)["poke_batch"],
        })(),
        "field_tensor": torch.randn(B, K_TURNS, 72, device=device),
        "move_idx":     torch.randint(0, N_MOVES, (B, 4), device=device),
        "pp_ratio":     torch.rand(B, 4, device=device),
        "move_disabled": torch.zeros(B, 4, device=device),
        "mechanic_id":  torch.zeros(B, dtype=torch.long, device=device),
        "mechanic_type_idx": torch.zeros(B, dtype=torch.long, device=device),
        "action_mask":  torch.zeros(B, 13, dtype=torch.bool, device=device),
    }


def _token_labels() -> list[str]:
    """Return human-readable labels for the 52 sequence positions."""
    labels = []
    for t in range(K_TURNS):
        for s in range(N_SLOTS):
            if s < 6:
                labels.append(f"T{t} OWN{s}")
            elif s < 12:
                labels.append(f"T{t} OPP{s - 6}")
            else:
                labels.append(f"T{t} FIELD")
    return labels


def plot_attention_maps(
    checkpoint: str,
    layer: int | None = None,
    head: int | None = None,
    save_path: str | None = None,
    show: bool = False,
    batch_idx: int = 0,
    device: str = "cpu",
) -> None:
    """
    Load a checkpoint, run forward pass, and plot attention maps.

    If layer and head are specified, plots a single heatmap.
    Otherwise, plots a grid of all layers × heads.
    """
    dev = torch.device(device)
    agent = CynthAIAgent().to(dev)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=True)
    agent.load_state_dict(ckpt["model"])
    agent.eval()

    batch = _dummy_batch(B=4, device=device)
    pb = batch["poke_batch"]

    # Run forward through the backbone to get attention maps
    with torch.no_grad():
        result = agent.backbone.get_attention_maps(
            agent.poke_emb(pb),
            batch["field_tensor"],
        )

    attn_maps = result["attention_maps"]  # list of [B, H, 52, 52] per layer
    labels = _token_labels()
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[1]

    if layer is not None and head is not None:
        # Single heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        _plot_single_heatmap(ax, attn_maps[layer][batch_idx, head].numpy(),
                             labels, f"Layer {layer}, Head {head}")
        fig.tight_layout()
        _save_or_show(fig, save_path, show)
        return

    # Grid: layers × heads
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads, 4 * n_layers))
    fig.suptitle(f"Attention Maps — {Path(checkpoint).name}", fontsize=14, fontweight="bold")

    for li in range(n_layers):
        for hi in range(n_heads):
            ax = axes[li, hi] if n_layers > 1 else axes[hi]
            attn = attn_maps[li][batch_idx, hi].numpy()
            im = ax.imshow(attn, cmap="viridis", aspect="auto", vmin=0, vmax=attn.max())
            if li == 0:
                ax.set_title(f"Head {hi}", fontsize=9)
            if hi == 0:
                ax.set_ylabel(f"Layer {li}", fontsize=9)
            # Tick labels only for edges
            if li == n_layers - 1:
                ax.set_xticks(range(0, SEQ_LEN, 13))
                ax.set_xticklabels([labels[i] for i in range(0, SEQ_LEN, 13)], rotation=45, fontsize=5)
            if hi == 0:
                ax.set_yticks(range(0, SEQ_LEN, 13))
                ax.set_yticklabels([labels[i] for i in range(0, SEQ_LEN, 13)], fontsize=5)

    fig.tight_layout()
    _save_or_show(fig, save_path, show)


def _plot_single_heatmap(ax, attn: np.ndarray, labels: list[str], title: str) -> None:
    """Plot a single attention heatmap with token labels."""
    im = ax.imshow(attn, cmap="viridis", aspect="auto", vmin=0, vmax=attn.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    # Show every 13th label (1 per turn)
    step = 13
    ax.set_xticks(range(0, SEQ_LEN, step))
    ax.set_xticklabels([labels[i] for i in range(0, SEQ_LEN, step)], rotation=45, fontsize=7)
    ax.set_yticks(range(0, SEQ_LEN, step))
    ax.set_yticklabels([labels[i] for i in range(0, SEQ_LEN, step)], fontsize=7)
    ax.set_xlabel("Keys (attended to)")
    ax.set_ylabel("Queries (attending)")


def _save_or_show(fig, save_path: str | None, show: bool) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CynthAI_v2 Attention Map Visualizer")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (0-based)")
    parser.add_argument("--head", type=int, default=None, help="Head index (0-based)")
    parser.add_argument("--save", default=None, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    parser.add_argument("--batch-idx", type=int, default=0, help="Batch index to visualize")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    plot_attention_maps(
        checkpoint=args.checkpoint,
        layer=args.layer,
        head=args.head,
        save_path=args.save,
        show=args.show,
        batch_idx=args.batch_idx,
        device=args.device,
    )