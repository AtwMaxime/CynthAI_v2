"""
diag.probing — Unified probing package for CynthAI_v2.

Modules:
  _common         Shared probe helpers, constants, cache loading
  actor_probes    Backbone 52-token probes + CLS + cross-token matrix
  critic_probes   IndependentCritic CLS/seq probes + PCA + erank
  detr_probes     DETR query probes (action, win, dHP, KO)
  svd_probes      PCA + SVD per-state analysis

Usage:
    python -m diag.probing --cache diag/seq_all_cache.pt --analyses all
    python -m diag.probing --cache diag/seq_all_cache.pt --analyses actor,critic
    python -m diag.probing --checkpoint checkpoints/cheater_v5/agent_001000.pt
"""
