import os
import random

import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_optimizer(cfg, model: torch.nn.Module):
    param_groups = list(model.parameters())
    if cfg["type"] == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"],
            ),
            eps=cfg["eps"],
        )
    elif cfg["type"] == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"],
            ),
            eps=cfg["eps"],
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg['type']}")


class WSD_LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio, decay_ratio, peak_lr, last_epoch=-1):
        assert 0.0 <= warmup_ratio < 1.0
        assert 0.0 <= decay_ratio < 1.0
        assert warmup_ratio + decay_ratio <= 1.0

        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.decay_steps = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.peak_lr = peak_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for _ in self.optimizer.param_groups:
            if step < self.warmup_steps:
                # Linear warmup
                lr = self.peak_lr * step / self.warmup_steps
            elif step < self.warmup_steps + self.stable_steps:
                # Constant stable phase
                lr = self.peak_lr
            elif step < self.total_steps:
                # Linear decay
                decay_progress = (step - self.warmup_steps - self.stable_steps) / self.decay_steps
                lr = self.peak_lr * (1.0 - decay_progress)
            else:
                # After total_steps
                lr = 0.0
            lrs.append(lr)

        return lrs


def get_scheduler(cfg, optimizer: torch.optim.Optimizer):
    if cfg["type"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg["factor"],
            patience=cfg["patience"],
            min_lr=cfg["min_lr"],
        )
    elif cfg["type"] == "wsd": # TODO: i think this is buggy, revisit later
        return WSD_LRScheduler(
            optimizer,
            total_steps=cfg["total_steps"],
            warmup_ratio=cfg["warmup_ratio"],
            decay_ratio=cfg["decay_ratio"],
            peak_lr=cfg["peak_lr"],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg['type']}")


def sum_weighted_losses(losses: dict[str, torch.Tensor], weights: dict[str, float] | None):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss: torch.Tensor = torch.zeros_like(list(losses.values())[0])
    if weights is None:
        weights = {k: 1.0 for k in losses.keys()}
    for k in losses.keys():
        loss = loss + weights[k] * losses[k]
    return loss


def worker_init_fn(worker_id: int):
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    global_rank = rank_zero_only.rank
    process_seed = torch.initial_seed()

    base_seed = process_seed - worker_id
    print(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)
