"""Debug: run 3 training updates to test for blocking."""
import sys, torch, time
sys.path.insert(0, ".")

from training.self_play import TrainingConfig, train

cfg = TrainingConfig(
    run_name        = "debug_test",
    total_updates   = 3,
    n_envs          = 4,
    min_steps       = 64,
    n_epochs        = 2,
    batch_size      = 32,
    lr              = 2.5e-4,
    warmup_steps    = 20,
    c_value         = 1.0,
    c_entropy       = 0.02,
    c_pred          = 0.8,
    max_grad_norm   = 0.5,
    weight_decay    = 1e-4,
    pool_size       = 30,
    checkpoint_freq = 50,
    eval_freq       = 10,
    eval_n_games    = 50,
    log_every       = 1,
    win_rate_window = 100,
    device          = "cpu",
)
print("Starting debug training...")
train(cfg)