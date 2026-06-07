"""
DEAD CODE — references removed IndependentCritic with action-aware cross-attention.
This script will NOT work after the critic unification refactoring (v12+).

Diagnose NaN in critic cross-attention (cheater_v9 crash at update 15).

Root cause hypothesis:
  - IndependentCritic cross-attention uses key_padding_mask = action_mask
  - action_mask convention: True = illegal (mask out)
  - PyTorch key_padding_mask: True = ignore position
  - Semantics match! BUT: when ALL 13 actions are illegal (all True),
    softmax is computed over zero valid keys → NaN
  - NaN propagates: attn_out → cls_out → value → loss → backward → weights corrupted
  - Next forward: model outputs NaN log_probs → _sample_action crash

This script reproduces the bug in isolation, then tests the fix.

Usage (on auriga):
    cd /local_scratch/mattwood/projects/rl_agent
    python CynthAI_v2/debug/diagnose_nan_cross_attn.py
"""

import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import D_MODEL, N_HEADS, DROPOUT

def make_cross_attn():
    return nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)

def test_all_masked_nan():
    """Reproduce: all-True key_padding_mask → NaN in attention output."""
    print("=" * 70)
    print("TEST 1: All actions masked → NaN in cross-attention output")
    print("=" * 70)

    B = 4
    cross_attn = make_cross_attn().eval()
    ln = nn.LayerNorm(D_MODEL)

    cls_q = torch.randn(B, 1, D_MODEL)
    action_embeds = torch.randn(B, 13, D_MODEL)

    # Case A: some legal actions → should be fine
    mask_partial = torch.zeros(B, 13, dtype=torch.bool)
    mask_partial[:, 4:] = True  # 4 legal, 9 illegal
    with torch.no_grad():
        out_a, w_a = cross_attn(cls_q, action_embeds, action_embeds,
                                key_padding_mask=mask_partial, need_weights=True)
    has_nan_a = torch.isnan(out_a).any().item()
    print(f"  Partial mask (4 legal):  has_nan={has_nan_a}  ✓" if not has_nan_a
          else f"  Partial mask (4 legal):  has_nan={has_nan_a}  ✗ UNEXPECTED")

    # Case B: ALL actions masked → NaN expected (the bug)
    mask_all = torch.ones(B, 13, dtype=torch.bool)
    with torch.no_grad():
        out_b, w_b = cross_attn(cls_q, action_embeds, action_embeds,
                                key_padding_mask=mask_all, need_weights=True)
    has_nan_b = torch.isnan(out_b).any().item()
    print(f"  All masked (0 legal):    has_nan={has_nan_b}  {'✓ BUG REPRODUCED' if has_nan_b else '✗ NOT REPRODUCED'}")

    # Show NaN propagation through LayerNorm + value head
    cls_out = ln(cls_q + out_b).squeeze(1)
    value_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 1))
    with torch.no_grad():
        v = value_head(cls_out)
    print(f"  Value after NaN prop:    has_nan={torch.isnan(v).any().item()}")
    print()
    return has_nan_b


def test_gradient_contamination():
    """Show that NaN value → NaN loss → NaN gradients → corrupted weights."""
    print("=" * 70)
    print("TEST 2: NaN value contaminates gradients and weights")
    print("=" * 70)

    B = 8
    cross_attn = make_cross_attn()
    ln = nn.LayerNorm(D_MODEL)
    value_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 1))

    # Simulate a batch where 1 sample has all actions masked
    cls_q = torch.randn(B, 1, D_MODEL, requires_grad=True)
    action_embeds = torch.randn(B, 13, D_MODEL)

    mask = torch.zeros(B, 13, dtype=torch.bool)
    mask[3, :] = True  # sample 3: all illegal

    out, _ = cross_attn(cls_q, action_embeds, action_embeds,
                        key_padding_mask=mask, need_weights=True)
    cls_out = ln(cls_q + out).squeeze(1)
    v = value_head(cls_out)

    print(f"  Values (sample 3 is NaN): {v.detach().squeeze().tolist()}")

    # Compute a simple loss
    returns = torch.randn(B, 1)
    loss = F.mse_loss(v, returns)
    print(f"  MSE loss: {loss.item()}")

    loss.backward()
    # Check if gradients are NaN
    nan_grads = sum(1 for p in value_head.parameters() if torch.isnan(p.grad).any())
    print(f"  Value head params with NaN grads: {nan_grads}/{sum(1 for _ in value_head.parameters())}")

    # Simulate optimizer step
    with torch.no_grad():
        for p in value_head.parameters():
            p -= 0.001 * p.grad
    nan_weights = sum(1 for p in value_head.parameters() if torch.isnan(p).any())
    print(f"  Value head params with NaN weights after step: {nan_weights}/{sum(1 for _ in value_head.parameters())}")
    print()


def test_how_often_all_masked():
    """Check if all-masked scenario can actually happen during rollout.

    Even if rare, a single occurrence per rollout is enough to corrupt the model.
    The all-masked case happens when:
      - All Pokemon fainted (game should be over, but env might send one more step)
      - Edge case in Revival Blessing handling
    """
    print("=" * 70)
    print("TEST 3: Frequency analysis — how likely is all-masked?")
    print("=" * 70)

    # We can't run the full env here, but we can check _sample_action's fallback
    print("  _sample_action handles all-masked with fallback to action 0")
    print("  BUT the critic forward runs BEFORE _sample_action")
    print("  So even if _sample_action survives, the critic already produced NaN")
    print("  → NaN stored in buffer → NaN during training → model corrupted")
    print()
    print("  The fix must be in the critic forward pass, not in _sample_action.")
    print()


def test_fix_nan_to_num():
    """Test fix: use nan_to_num on attention output before residual connection."""
    print("=" * 70)
    print("TEST 4: Fix — nan_to_num on attn_out before residual")
    print("=" * 70)

    B = 4
    cross_attn = make_cross_attn().eval()
    ln = nn.LayerNorm(D_MODEL)
    value_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 1))

    cls_q = torch.randn(B, 1, D_MODEL)
    action_embeds = torch.randn(B, 13, D_MODEL)
    mask_all = torch.ones(B, 13, dtype=torch.bool)

    with torch.no_grad():
        out, w = cross_attn(cls_q, action_embeds, action_embeds,
                            key_padding_mask=mask_all, need_weights=True)
        # FIX: replace NaN with 0 — effectively skip cross-attention contribution
        out = torch.nan_to_num(out, nan=0.0)
        cls_out = ln(cls_q + out).squeeze(1)
        v = value_head(cls_out)

    has_nan = torch.isnan(v).any().item()
    print(f"  With nan_to_num fix: has_nan={has_nan}  {'✓ FIXED' if not has_nan else '✗ STILL BROKEN'}")
    print()


def test_fix_skip_when_all_masked():
    """Test fix: skip cross-attention entirely when all actions are masked."""
    print("=" * 70)
    print("TEST 5: Fix — skip cross-attn for samples with all-masked actions")
    print("=" * 70)

    B = 8
    cross_attn = make_cross_attn().eval()
    ln = nn.LayerNorm(D_MODEL)
    value_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 1))

    cls_q = torch.randn(B, 1, D_MODEL)
    action_embeds = torch.randn(B, 13, D_MODEL)

    mask = torch.zeros(B, 13, dtype=torch.bool)
    mask[3, :] = True  # sample 3: all illegal
    mask[7, :] = True  # sample 7: all illegal

    # Identify which samples have all actions masked
    all_masked = mask.all(dim=1)  # [B]
    print(f"  All-masked samples: {all_masked.tolist()}")

    # Fix: for all-masked samples, unmask action 0 so softmax has at least 1 valid key
    safe_mask = mask.clone()
    safe_mask[all_masked, 0] = False  # ensure at least 1 key is valid

    with torch.no_grad():
        out, w = cross_attn(cls_q, action_embeds, action_embeds,
                            key_padding_mask=safe_mask, need_weights=True)
        cls_out = ln(cls_q + out).squeeze(1)
        v = value_head(cls_out)

    has_nan = torch.isnan(v).any().item()
    print(f"  With safe_mask fix:  has_nan={has_nan}  {'✓ FIXED' if not has_nan else '✗ STILL BROKEN'}")
    print(f"  Values: {v.detach().squeeze().tolist()}")
    print()


def test_fix_combined_robust():
    """Test the recommended production fix: nan_to_num + gradient check."""
    print("=" * 70)
    print("TEST 6: Production fix — nan_to_num + loss NaN guard")
    print("=" * 70)

    B = 8
    cross_attn = make_cross_attn()
    ln = nn.LayerNorm(D_MODEL)
    value_head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.ReLU(), nn.Linear(D_MODEL, 1))

    cls_q = torch.randn(B, 1, D_MODEL, requires_grad=True)
    action_embeds = torch.randn(B, 13, D_MODEL)

    mask = torch.zeros(B, 13, dtype=torch.bool)
    mask[3, :] = True  # all masked

    # Forward with fix
    out, w = cross_attn(cls_q, action_embeds, action_embeds,
                        key_padding_mask=mask, need_weights=True)
    out = torch.nan_to_num(out, nan=0.0)  # FIX 1
    cls_out = ln(cls_q + out).squeeze(1)
    v = value_head(cls_out)

    returns = torch.randn(B, 1)
    loss = F.mse_loss(v, returns)

    # FIX 2: skip backward if loss is NaN (safety net)
    if torch.isnan(loss):
        print("  Loss is NaN — would skip backward (safety net)")
    else:
        loss.backward()
        nan_grads = sum(1 for p in value_head.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        print(f"  Loss: {loss.item():.4f}")
        print(f"  NaN grads: {nan_grads}  {'✓ CLEAN' if nan_grads == 0 else '✗ CONTAMINATED'}")

    print()


def test_full_critic_simulation():
    """Simulate the full IndependentCritic forward with the fix applied."""
    print("=" * 70)
    print("TEST 7: Full IndependentCritic forward simulation")
    print("=" * 70)

    from model.critic import IndependentCritic

    critic = IndependentCritic(
        n_layers=2,
        value_bound=10.0,
        use_victory_head=False,
        action_aware=True,
        n_cross_layers=1,
    ).eval()

    B = 4
    from model.embeddings import TOKEN_DIM, FIELD_DIM
    from model.backbone import K_TURNS

    pokemon_tokens = torch.randn(B, K_TURNS * 12, TOKEN_DIM)
    field_tokens = torch.randn(B, K_TURNS, FIELD_DIM)
    action_embeds = torch.randn(B, 13, D_MODEL)

    # Normal case
    mask_normal = torch.zeros(B, 13, dtype=torch.bool)
    mask_normal[:, 4:] = True
    with torch.no_grad():
        v_normal, _ = critic(pokemon_tokens, field_tokens,
                             action_embeds=action_embeds, action_mask=mask_normal)
    print(f"  Normal mask:     values={v_normal.squeeze().tolist()}")
    print(f"                   has_nan={torch.isnan(v_normal).any().item()}")

    # All-masked case (triggers bug)
    mask_all = torch.ones(B, 13, dtype=torch.bool)
    with torch.no_grad():
        v_all, _ = critic(pokemon_tokens, field_tokens,
                          action_embeds=action_embeds, action_mask=mask_all)
    print(f"  All-masked:      values={v_all.squeeze().tolist()}")
    print(f"                   has_nan={torch.isnan(v_all).any().item()}")

    # Mixed case (1 sample all-masked)
    mask_mixed = torch.zeros(B, 13, dtype=torch.bool)
    mask_mixed[:, 4:] = True
    mask_mixed[2, :] = True  # sample 2 all masked
    with torch.no_grad():
        v_mixed, _ = critic(pokemon_tokens, field_tokens,
                            action_embeds=action_embeds, action_mask=mask_mixed)
    print(f"  Mixed (sample 2): values={v_mixed.squeeze().tolist()}")
    print(f"                    has_nan={torch.isnan(v_mixed).any().item()}")
    print()


if __name__ == "__main__":
    torch.manual_seed(42)

    bug_reproduced = test_all_masked_nan()
    test_gradient_contamination()
    test_how_often_all_masked()
    test_fix_nan_to_num()
    test_fix_skip_when_all_masked()
    test_fix_combined_robust()

    if bug_reproduced:
        print("Testing with actual IndependentCritic...")
        test_full_critic_simulation()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Root cause: IndependentCritic cross-attention crashes when ALL 13 actions
are masked (key_padding_mask all-True → softmax over 0 valid keys → NaN).

The NaN propagates: attn_out → cls_out → value → loss → backward → weights
corrupted → next forward produces NaN log_probs → _sample_action crash.

This happens even if the all-masked case is rare (1 sample in 1 batch is
enough to corrupt the entire model via shared gradient update).

Recommended fixes (apply both):
  1. critic.py: torch.nan_to_num(attn_out, nan=0.0) after cross-attention
  2. self_play.py: skip optimizer.step() if loss is NaN (safety net)
  3. rollout.py: clamp log_probs with nan_to_num before _sample_action (defense in depth)
""")
