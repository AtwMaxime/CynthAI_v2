"""
CynthAI_v2 Training Losses — PPO-clip + auxiliary predictor.

Single entry point: compute_losses() → dict of scalar losses.

PPO components:
  policy_loss  — PPO-clip surrogate (maximise advantage, clip ratio)
  value_loss   — MSE between V(s) estimate and discounted returns
  entropy_loss — negative entropy over legal actions (encourages exploration)

Auxiliary:
  pred_loss    — cross-entropy on opponent hidden state (item/ability/tera/moves)
                 pre-computed by PredictionHeads.compute_loss(), passed as scalar

Total:
  total = policy_loss + c_value * value_loss + c_entropy * entropy_loss + c_pred * pred_loss

Notes:
  - Advantage normalisation is applied here (mean=0, std=1 per batch).
  - Value loss uses simple MSE — returns from Pokemon are bounded in [-1, 1]
    after discounting so the critic scale is stable without running normalisation.
  - Entropy is computed only over legal actions (action_mask=False slots).
    Illegal action probs ≈ 0 from the backbone's -1e9 masking; explicit
    masked_fill(action_mask, 0) before sum() is a safety net against NaN.
  - Gradient clipping (clip_grad_norm_ 0.5) lives in self_play.py, not here.
  - log_prob_old is stored per-step in the rollout buffer (one scalar, not all
    13 logits) to minimise buffer memory.
"""

import torch
import torch.nn.functional as F


def compute_losses(
    logits_new:   torch.Tensor,   # [B, 13]  current model logits (backbone-masked)
    log_prob_old: torch.Tensor,   # [B]      log P(a|s) at collection time
    actions:      torch.Tensor,   # [B]      int64 — chosen action indices
    advantages:   torch.Tensor,   # [B]      GAE advantages (unnormalised)
    returns:      torch.Tensor,   # [B]      discounted returns (value targets)
    values:       torch.Tensor,   # [B] or [B,1]  current value estimates
    action_mask:  torch.Tensor,   # [B, 13]  bool — True = illegal
    pred_loss:    torch.Tensor,   # scalar   PredictionHeads total loss
    *,
    clip_eps:  float = 0.2,
    c_value:   float = 1.0,       # P5: increased from 0.5 for critic stability
    c_entropy: float = 0.01,
    c_pred:    float = 0.5,
) -> dict[str, torch.Tensor]:
    """
    Compute all PPO + auxiliary losses for one minibatch.

    Returns a dict with keys:
        "policy"   — PPO-clip loss              (minimise)
        "value"    — MSE value loss             (minimise)
        "entropy"  — negative entropy           (minimise = maximise entropy)
        "pred"     — predictor auxiliary        (minimise)
        "clip_frac"— fraction of clipped ratios (monitoring)
        "total"    — weighted sum               (call .backward() on this)
    """

    # ── Advantage normalisation ───────────────────────────────────────────────
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)   # [B]

    # ── Policy loss (PPO-clip) ────────────────────────────────────────────────
    log_probs_new = F.log_softmax(logits_new, dim=-1)                     # [B, 13]
    log_prob_new  = log_probs_new.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

    ratio = torch.exp(log_prob_new - log_prob_old)                        # [B]
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    # ── Clip fraction (monitoring) ───────────────────────────────────────────
    clip_frac = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean()

    # ── Value loss (MSE) ──────────────────────────────────────────────────────
    # Returns are pre-normalised globally in RolloutBuffer.compute_gae()
    # so no per-batch normalisation is needed here.
    values  = values.squeeze(-1)
    value_loss = F.mse_loss(values, returns)

    # ── Explained variance (monitoring) ──────────────────────────────────────
    ev = 1.0 - ((returns - values)**2).sum() / ((returns - returns.mean())**2).sum().clamp(min=1e-8)

    # ── Entropy bonus (legal actions only) ───────────────────────────────────
    # backbone already applied -1e9 to illegal slots → probs ≈ 0 for those.
    # clamp prevents log(0); masked_fill zeros illegal contributions before sum.
    probs     = F.softmax(logits_new, dim=-1)                             # [B, 13]
    log_probs = torch.log(probs.clamp(min=1e-8))                          # [B, 13]
    entropy   = -(probs * log_probs).masked_fill(action_mask, 0.0).sum(dim=-1)  # [B]
    entropy_loss = -entropy.mean()   # negative: we minimise loss → maximise entropy

    # ── Diagnostics (P16d) ─────────────────────────────────────────────────
    with torch.no_grad():
        adv_mean  = advantages.mean().item()
        adv_std   = advantages.std().item()
        ratio_dev = (ratio - 1.0).abs().mean().item()

    # ── Total ─────────────────────────────────────────────────────────────────
    total = (
        policy_loss
        + c_value   * value_loss
        + c_entropy * entropy_loss
        + c_pred    * pred_loss
    )

    return {
        "policy":     policy_loss,
        "value":      value_loss,
        "entropy":    entropy_loss,
        "pred":       pred_loss,
        "clip_frac":  clip_frac.detach(),
        "explained_variance": ev.detach(),
        "adv_mean":   adv_mean,
        "adv_std":    adv_std,
        "ratio_dev":  ratio_dev,
        "total":      total,
    }
