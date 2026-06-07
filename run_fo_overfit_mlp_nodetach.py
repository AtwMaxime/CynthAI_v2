"""
FO overfit diagnostic — MLP critic, no detach.
critic_n_layers=0, critic_detach=False → MLP on backbone CLS with full gradient flow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.self_play import TrainingConfig, train
import torch

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TrainingConfig(
        run_name      = "fo_mlp_nodetach",
        total_updates = 500,
        n_envs=64, min_steps=4096,
        n_epochs=4, batch_size=256,
        lr=2.5e-4, lr_min=1e-5, warmup_steps=20,
        c_value=2.0, c_entropy=0.01, c_pred=0.6,
        c_attn_entropy=0.01, c_attn_rank=0.005,
        max_grad_norm=0.5, weight_decay=1e-4,

        critic_n_layers=0, critic_detach=False,
        critic_lr=5e-4, critic_wd=1e-4, critic_grad_norm=1.0,

        use_victory_head=True, c_victory=0.1,
        value_dump_threshold=20.0, critic_value_bound=10.0,

        opp_fo_frac=1.0, opp_ema_frac=0.0,
        pool_snapshot_freq=9999, pool_snapshot_threshold=1.0, pool_cooldown=9999,
        ema_warmup=99999,

        mask_schedule="phase", mask_phase_breakpoints=(), mask_phase_values=(),
        dense_schedule="phase", dense_phase_breakpoints=(), dense_phase_values=(),

        checkpoint_freq=100, eval_freq=50, eval_n_games=200,
        probe_freq=0, log_every=1, win_rate_window=50,
        device=args.device,
    )
    train(cfg)
