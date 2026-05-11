"""Debug: Curriculum config on GPU, 50 updates to verify no more blocking."""
import sys, torch, os
sys.path.insert(0, ".")

from training.self_play import TrainingConfig, train

cfg = TrainingConfig(
    run_name        = "debug_gpu_final",
    total_updates   = 50,
    n_envs          = 16,
    min_steps       = 512,
    n_epochs        = 4,
    batch_size      = 128,
    lr              = 2.5e-4,
    lr_min          = 1e-5,
    warmup_steps    = 20,
    c_value         = 1.0,
    c_entropy       = 0.02,
    c_pred          = 0.8,
    max_grad_norm   = 0.5,
    weight_decay    = 1e-4,
    pool_size       = 30,
    pool_snapshot_threshold = 0.55,
    pool_cooldown   = 5,
    mask_schedule           = "phase",
    mask_phase_breakpoints  = (600, 2500),
    mask_phase_values       = (0.0, 0.5, 1.0),
    dense_schedule          = "phase",
    dense_phase_breakpoints = (600, 2500),
    dense_phase_values      = (1.0, 0.5, 0.1),
    checkpoint_freq = 50,
    eval_freq       = 30,
    eval_n_games    = 500,
    log_every       = 1,
    win_rate_window = 100,
    device          = "cuda",
)
print(f"Device: {cfg.device}  n_envs={cfg.n_envs}  updates={cfg.total_updates}")
train(cfg)
print("DONE - 50 updates completed without blocking")