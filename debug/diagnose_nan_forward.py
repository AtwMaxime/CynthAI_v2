"""
Diagnose NaN in the full agent forward+backward with action-aware critic.

Reuses the dummy data setup from test_full_pipeline.py.
Tests that the critic nan_to_num fix prevents NaN contamination in all scenarios.

Usage:
    cd CynthAI_v2 && python debug/diagnose_nan_forward.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from model.agent import CynthAIAgent, AgentOutput
from model.embeddings import (
    collate_features, collate_field_features,
    TOKEN_DIM, FIELD_DIM,
)
from model.backbone import K_TURNS, D_MODEL, N_SLOTS
from training.rollout import encode_state
from training.losses import compute_losses

# ---------- dummy data (same as test_full_pipeline) ----------
from tests.test_full_pipeline import (
    pb, field_t, move_idx, pp_ratio, move_disabled,
    mechanic_id, mech_type_idx, action_mask, B,
)


def make_agent():
    return CynthAIAgent(
        critic_n_layers=2,
        critic_detach=True,
        critic_value_bound=10.0,
    )


def forward_agent(agent, mask_override=None):
    m = mask_override if mask_override is not None else action_mask
    return agent(
        poke_batch=pb, field_tensor=field_t,
        move_idx=move_idx, pp_ratio=pp_ratio, move_disabled=move_disabled,
        mechanic_id=mechanic_id, mechanic_type_idx=mech_type_idx,
        action_mask=m,
    )


def test_normal():
    print("=" * 70)
    print("TEST A: Normal forward (partial mask)")
    print("=" * 70)
    agent = make_agent().eval()
    with torch.no_grad():
        out = forward_agent(agent)
    print(f"  value has_nan: {torch.isnan(out.value).any().item()}")
    print(f"  logits has_nan: {torch.isnan(out.action_logits).any().item()}")
    print(f"  log_probs has_nan: {torch.isnan(out.log_probs).any().item()}")
    ok = not torch.isnan(out.value).any().item()
    print(f"  {'PASS' if ok else 'FAIL'}")
    print()
    return agent


def test_all_masked(agent):
    print("=" * 70)
    print("TEST B: All 13 actions masked (triggers cross-attn NaN)")
    print("=" * 70)
    all_mask = torch.ones(B, 13, dtype=torch.bool)
    with torch.no_grad():
        out = forward_agent(agent, mask_override=all_mask)
    v_nan = torch.isnan(out.value).any().item()
    lp_nan = torch.isnan(out.log_probs).any().item()
    print(f"  value has_nan: {v_nan}")
    print(f"  logits has_nan: {torch.isnan(out.action_logits).any().item()}")
    print(f"  log_probs has_nan: {lp_nan}")
    ok = not v_nan
    print(f"  {'PASS -- nan_to_num fix works' if ok else 'FAIL -- critic still produces NaN'}")
    print()
    return ok


def test_mixed_mask(agent):
    print("=" * 70)
    print("TEST C: Mixed batch -- 1 sample all-masked, rest normal")
    print("=" * 70)
    mixed_mask = action_mask.clone()
    mixed_mask[1, :] = True  # sample 1 all-masked
    with torch.no_grad():
        out = forward_agent(agent, mask_override=mixed_mask)
    v_nan = torch.isnan(out.value).any().item()
    print(f"  values: {out.value.squeeze().tolist()}")
    print(f"  value has_nan: {v_nan}")
    ok = not v_nan
    print(f"  {'PASS' if ok else 'FAIL'}")
    print()


def test_gradient_flow():
    print("=" * 70)
    print("TEST D: Forward + backward with mixed mask")
    print("=" * 70)
    agent = make_agent()
    agent.train()

    mixed_mask = action_mask.clone()
    mixed_mask[1, :] = True

    out = forward_agent(agent, mask_override=mixed_mask)
    print(f"  value has_nan: {torch.isnan(out.value).any().item()}")

    returns = torch.randn(B, 1)
    loss = F.mse_loss(out.value, returns) + out.log_probs.mean()
    print(f"  loss: {loss.item():.4f}  has_nan: {torch.isnan(loss).item()}")

    if not torch.isnan(loss):
        loss.backward()
        nan_params = [n for n, p in agent.named_parameters()
                      if p.grad is not None and torch.isnan(p.grad).any()]
        total = sum(1 for _ in agent.parameters())
        print(f"  NaN grad params: {len(nan_params)}/{total}")
        if nan_params:
            for n in nan_params[:5]:
                print(f"    {n}")
        ok = len(nan_params) == 0
    else:
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'}")
    print()


def test_full_ppo_step():
    print("=" * 70)
    print("TEST E: Full PPO loss computation with mixed mask")
    print("=" * 70)
    agent = make_agent()
    agent.train()

    mixed_mask = action_mask.clone()
    mixed_mask[1, :] = True

    out = forward_agent(agent, mask_override=mixed_mask)

    # Simulate stored rollout data
    actions = torch.randint(0, 4, (B,))
    log_prob_old = torch.randn(B)
    advantages = torch.randn(B)
    returns = torch.randn(B)
    pred_loss = torch.tensor(0.5)

    losses = compute_losses(
        logits_new=out.action_logits,
        log_prob_old=log_prob_old,
        actions=actions,
        advantages=advantages,
        returns=returns,
        values=out.value,
        action_mask=mixed_mask,
        pred_loss=pred_loss,
    )

    total = losses["total"]
    print(f"  total loss: {total.item():.4f}  has_nan: {torch.isnan(total).item()}")

    if not torch.isnan(total):
        total.backward()
        nan_params = [n for n, p in agent.named_parameters()
                      if p.grad is not None and torch.isnan(p.grad).any()]
        print(f"  NaN grad params: {len(nan_params)}")
        ok = len(nan_params) == 0
    else:
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def test_repeated_steps():
    """Simulate multiple training steps to check NaN doesn't accumulate."""
    print("=" * 70)
    print("TEST F: 20 consecutive forward+backward (NaN accumulation check)")
    print("=" * 70)
    agent = make_agent()
    agent.train()
    optimizer = torch.optim.AdamW(agent.parameters(), lr=2.5e-4)

    nan_count = 0
    for step in range(20):
        optimizer.zero_grad()
        # Every 5th step, inject all-masked sample
        mask = action_mask.clone()
        if step % 5 == 0:
            mask[0, :] = True

        out = forward_agent(agent, mask_override=mask)
        loss = F.mse_loss(out.value, torch.randn(B, 1)) + out.log_probs.mean()

        if torch.isnan(loss):
            nan_count += 1
            print(f"  step {step}: NaN loss!")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

        if any(torch.isnan(p).any() for p in agent.parameters()):
            nan_count += 1
            print(f"  step {step}: NaN in weights after optimizer step!")
            break

    if nan_count == 0:
        print(f"  20 steps completed, no NaN -- PASS")
    else:
        print(f"  {nan_count} NaN occurrences -- FAIL")
    print()


if __name__ == "__main__":
    torch.manual_seed(42)
    agent = test_normal()
    test_all_masked(agent)
    test_mixed_mask(agent)
    test_gradient_flow()
    test_full_ppo_step()
    test_repeated_steps()
