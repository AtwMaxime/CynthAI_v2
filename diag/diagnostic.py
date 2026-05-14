"""
Diagnostic: Analyse cross-attention et autres métriques.
Charge un checkpoint et extrait les poids d'attention pour diagnostiquer le collapse.
"""
import sys, json, math, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np

CKPT_DIR = Path("checkpoints/cheater")
EVAL_DIR = CKPT_DIR / "eval_data"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── 1. Analyse métriques CSV ──────────────────────────────────────────────────
print("=" * 70)
print("1. MÉTRIQUES GÉNÉRALES")
print("=" * 70)

metrics = np.genfromtxt(CKPT_DIR / "metrics.csv", delimiter=",", names=True)
print(f"Updates: {metrics['update'][0]:.0f} -> {metrics['update'][-1]:.0f}")

# Sampling
samples = [1, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, int(metrics['update'][-1])]
print(f"\n{'update':>6} {'wr':>5} {'policy':>8} {'entropy':>8} {'kl':>6} {'lr':>9} {'gn':>7} {'cf':>5} {'ev':>6} {'ia':>5} {'aa':>5} {'ta':>5} {'mr':>5} {'attn_e':>8}")
print("-" * 100)

# Find row by update number
names = metrics.dtype.names
for s in samples:
    mask = metrics['update'] == s
    if not mask.any():
        idx = np.abs(metrics['update'] - s).argmin()
        row = metrics[idx]
    else:
        row = metrics[mask][0]

    ia = row['item_acc'] if 'item_acc' in names else 0.0
    aa = row['ability_acc'] if 'ability_acc' in names else 0.0
    ta = row['tera_acc'] if 'tera_acc' in names else 0.0
    mr = row['move_recall'] if 'move_recall' in names else 0.0
    ae = row['attn_entropy'] if 'attn_entropy' in names else 0.0
    kl = row['kl'] if 'kl' in names else row['pred']

    print(f"{row['update']:6.0f} {row['win_rate']:5.2f} {row['policy']:8.4f} {row['entropy']:8.4f} {kl:6.4f} {row['lr']:9.2e} {row['grad_norm']:7.1f} {row['clip_frac']:5.3f} {row['explained_variance']:6.3f} {ia:5.2f} {aa:5.2f} {ta:5.2f} {mr:5.2f} {ae:8.4f}")

# ── 2. Analyse eval.csv (win rate per opponent) ───────────────────────────────
print("\n" + "=" * 70)
print("2. WIN RATE PAR OPPOSANT")
print("=" * 70)

eval_data = np.genfromtxt(CKPT_DIR / "eval.csv", delimiter=",", names=True)
print(f"\n{'update':>6} {'random':>6} {'fulloff':>6} {'ema':>6} {'pool':>6}")
print("-" * 35)
for row in eval_data:
    print(f"{row['update']:6.0f} {row['random_wr']:6.2f} {row['fulloff_wr']:6.2f} {row['ema_wr']:6.2f} {row['pool_wr']:6.2f}")

# ── 3. Analyse attention maps ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. ANALYSE CROSS-ATTENTION (via eval_data battle_lengths)")
print("=" * 70)

# Check if images can be analyzed
from PIL import Image
cross_attn_dir = CKPT_DIR / "cross_attn"
for fn in sorted(cross_attn_dir.glob("*.png")):
    img = Image.open(fn).convert("L")  # grayscale
    arr = np.array(img)
    print(f"\n{fn.stem}: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")

# ── 4. Load checkpoint and analyze cross-attention directly ────────────────────
print("\n" + "=" * 70)
print("4. ANALYSE CROSS-ATTENTION DEPUIS CHECKPOINT")
print("=" * 70)

from model.backbone import BattleBackbone, D_MODEL, N_HEADS, N_SLOTS

checkpoints = sorted(CKPT_DIR.glob("agent_*.pt"))
print(f"Checkpoints found: {len(checkpoints)}")
print(f"  D_MODEL={D_MODEL}, N_HEADS={N_HEADS}, N_SLOTS={N_SLOTS}")

for ckpt_path in checkpoints:
    tag = ckpt_path.stem  # e.g. "agent_001400"
    update = int(tag.split("_")[1])

    if update not in [100, 500, 1000, 1400]:
        continue

    print(f"\n--- {tag} (update {update}) ---")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())}")

    # Build backbone with correct config
    bb = BattleBackbone()
    # Extract backbone weights from flat dict
    bb_state = {k.replace("backbone.", ""): v for k, v in ckpt["model"].items()
                if k.startswith("backbone.")}
    bb.load_state_dict(bb_state)
    bb.eval()

    # Extract cross-attention weights
    # action_cross_attn is nn.MultiheadAttention with in_proj_weight [3*D_MODEL, D_MODEL]
    # and in_proj_bias [3*D_MODEL]

    # Get the QKV projection weights
    w = bb.action_cross_attn.in_proj_weight  # [3*256, 256]
    b = bb.action_cross_attn.in_proj_bias     # [3*256]

    # Split into Q, K, V
    d = D_MODEL
    w_q, w_k, w_v = w[:d], w[d:2*d], w[2*d:]
    b_q, b_k, b_v = b[:d], b[d:2*d], b[2*d:]

    # Compute attention patterns analytically
    # For any pair of tokens (query_i, key_j), the attention logit is
    # q_i^T k_j / sqrt(d_k) where d_k = D_MODEL / N_HEADS = 64

    d_k = D_MODEL // N_HEADS

    # Reshape to per-head
    w_q_h = w_q.reshape(N_HEADS, d_k, d)  # [H, d_k, D]
    w_k_h = w_k.reshape(N_HEADS, d_k, d)   # [H, d_k, D]

    # For a uniform input distribution, the expected attention pattern
    # is determined by QK^T / sqrt(d_k). The dominant pattern can be found
    # by looking at the SVD of each head's QK^T.

    # Compute QK^T matrix: [H, D, D] where each entry is
    # (w_q_h[h]^T w_k_h[h]) / sqrt(d_k) — the pairwise affinity in input space
    qk = torch.bmm(w_q_h, w_k_h.transpose(1, 2)) / math.sqrt(d_k)  # [H, D, D]

    # Per-head, find which output dimensions (token positions) have high self-affinity
    # If QK^T has a rank-1 structure, it means attention is collapsing

    for h in range(N_HEADS):
        u, s, v = torch.svd(qk[h])
        explained = (s[0]**2 / (s**2).sum()).item()
        print(f"  Head {h}: QK^T top singular value explains {explained:.3f} of variance "
              f"(s=[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}, ...])")

    # Also get output projection weights
    out_w = bb.action_cross_attn.out_proj.weight  # [D, D]
    out_b = bb.action_cross_attn.out_proj.bias     # [D]

    # Final output score head
    score_w = bb.action_score.weight  # [1, D]
    score_b = bb.action_score.bias     # [1]

    # Combine to find which token positions the action score head reads from
    # attn_out = softmax(QK^T / sqrt(d_k)) @ V
    # For a uniform query, this depends on the combination of all projections
    # Equivalent: check out_proj @ score_head to see which token features matter

    combined = score_w @ out_w  # [1, D]
    print(f"  Score head reading from: mean|w|={combined.abs().mean().item():.4f}, "
          f"max|w|={combined.abs().max().item():.4f}")

    # Run actual forward pass with dummy data to get attention distribution
    print(f"\n  Running forward pass with random inputs...")

    B = 256  # batch size for stats
    with torch.no_grad():
        # Create random tokens
        dummy_current = torch.randn(B, N_SLOTS, D_MODEL)
        dummy_actions = torch.randn(B, N_SLOTS, D_MODEL)
        dummy_mask = torch.zeros(B, N_SLOTS, dtype=torch.bool)

        # Store attention
        bb._store_cross_attn = True

        # Run act()
        logits, attn_entropy, attn_rank = bb.act(dummy_actions, dummy_current, dummy_mask)

        # Get stats
        stats = bb.get_cross_attention_stats()
        bb._store_cross_attn = False

        if stats is not None:
            mean_w = stats["mean"]  # [H, 13, 13]
            std_w = stats["std"]
            n = stats["n"]

            # Average over heads
            avg_w = mean_w.mean(dim=0)  # [13, 13]

            print(f"  N={n}, mean attn entropy = {attn_entropy.item():.6f}")

            # For each query (action), find the top-1 attended token position
            top1_per_query = avg_w.argmax(dim=-1)  # [13]

            print(f"\n  Top-1 token per action query:")
            state_labels = ["OWN0","OWN1","OWN2","OWN3","OWN4","OWN5",
                            "OPP0","OPP1","OPP2","OPP3","OPP4","OPP5","FIELD"]
            action_labels = ["M1","M2","M3","M4","T1","T2","T3","T4","S0","S1","S2","S3","S4"]

            for q in range(13):
                t = top1_per_query[q].item()
                attn_val = avg_w[q, t].item()
                print(f"    {action_labels[q]:>3} -> {state_labels[t]:>5} ({attn_val:.3f})")

            # Distribution of attention across token GROUPS
            # Own: 0-5, Opp: 6-11, Field: 12
            own_attn = avg_w[:, 0:6].sum(dim=-1)   # [13]
            opp_attn = avg_w[:, 6:12].sum(dim=-1)  # [13]
            fld_attn = avg_w[:, 12:13].sum(dim=-1)  # [13]

            print(f"\n  Attention distribution by token group (avg over queries):")
            print(f"    Own Pokémon: {own_attn.mean().item():.3f} "
                  f"(per-query: {', '.join(f'{x:.2f}' for x in own_attn.tolist())})")
            print(f"    Opp Pokémon: {opp_attn.mean().item():.3f} "
                  f"(per-query: {', '.join(f'{x:.2f}' for x in opp_attn.tolist())})")
            print(f"    Field:       {fld_attn.mean().item():.3f} "
                  f"(per-query: {', '.join(f'{x:.2f}' for x in fld_attn.tolist())})")

            # Per-head analysis
            print(f"\n  Per-head top-1 distribution:")
            for h in range(N_HEADS):
                h_avg = mean_w[h]  # [13, 13]
                h_top1 = h_avg.argmax(dim=-1)
                top1_counts = torch.zeros(13, dtype=torch.int32)
                for q in range(13):
                    top1_counts[h_top1[q]] += 1
                top_fields = [(i, top1_counts[i].item()) for i in top1_counts.argsort(descending=True) if top1_counts[i] > 0]
                fields_str = ", ".join(f"{state_labels[t]}:{c}" for t, c in top_fields)
                print(f"    Head {h}: {fields_str}")

            # Measure entropy per head
            print(f"\n  Per-head entropy (mean over queries):")
            for h in range(N_HEADS):
                h_w = mean_w[h]  # [13, 13]
                h_ent = -(h_w * torch.log(h_w.clamp(min=1e-8))).sum(dim=-1)  # [13]
                print(f"    Head {h}: mean={h_ent.mean().item():.4f}, "
                      f"min={h_ent.min().item():.4f}, max={h_ent.max().item():.4f}")

print("\n✅ Diagnostic terminé.")