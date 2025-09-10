import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from diffusion_backend import NoisePredictor, SmileyDataset
from diffusion_backend.noise_schedule import LinearNoiseSchedule


def detect_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    steps: int = 10000
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    timesteps: int = 300
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    ckpt_dir: str = "checkpoints"
    ckpt_freq: int = 1000
    seed: Optional[int] = 0
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    ema_decay: float = 0.999
    # Resume options
    resume: Optional[str] = None  # None, "auto", or explicit ckpt path


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    torch.manual_seed(seed)
    try:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass


def sample_batch(ds: SmileyDataset, batch_size: int, device: torch.device) -> torch.Tensor:
    # Sample points on the fly each step
    with torch.no_grad():
        x0 = torch.stack([ds.sample() for _ in range(batch_size)], dim=0)
    return x0.to(device)


def save_checkpoint(path: Path, step: int, model: NoisePredictor, model_ema: NoisePredictor, opt: optim.Optimizer, cfg: TrainConfig, schedule: LinearNoiseSchedule):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict(),
        "optimizer": opt.state_dict(),
        "config": asdict(cfg),
        "schedule": {
            "timesteps": schedule.T,
            "beta_start": float(schedule.betas[0].item()),
            "beta_end": float(schedule.betas[-1].item()),
        },
    }
    torch.save(payload, path)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = detect_device(cfg.device)

    # Optionally load checkpoint metadata first to align timesteps
    start_step = 0
    payload = None
    resume_path: Optional[Path] = None
    if cfg.resume is not None:
        resume_path = Path(cfg.resume) if cfg.resume != "auto" else Path(cfg.ckpt_dir) / "latest.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested but checkpoint not found: {resume_path}")
        payload = torch.load(resume_path, map_location=device)
        sched_meta = payload.get("schedule", {}) if isinstance(payload, dict) else {}
        # Align timesteps to checkpoint to ensure model shapes match
        ckpt_T = int(sched_meta.get("timesteps", cfg.timesteps))
        if ckpt_T != cfg.timesteps:
            cfg.timesteps = ckpt_T
        # Align schedule endpoints for exact resume behavior
        if "beta_start" in sched_meta:
            cfg.beta_start = float(sched_meta["beta_start"])  # type: ignore[index]
        if "beta_end" in sched_meta:
            cfg.beta_end = float(sched_meta["beta_end"])  # type: ignore[index]
        start_step = int(payload.get("step", 0))

    # Data, model, schedule
    ds = SmileyDataset(seed=cfg.seed, device=str(device))
    model = NoisePredictor(num_timesteps=cfg.timesteps).to(device)
    # EMA model (for sampling and checkpoints)
    model_ema = NoisePredictor(num_timesteps=cfg.timesteps).to(device)
    model_ema.load_state_dict(model.state_dict())
    for p in model_ema.parameters():
        p.requires_grad_(False)
    schedule = LinearNoiseSchedule(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=device,
        seed=(cfg.seed + 1) if cfg.seed is not None else None,
    )

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # If resuming, load weights and optimizer state
    if payload is not None:
        model.load_state_dict(payload["model"])  # type: ignore[index]
        if "model_ema" in payload:
            model_ema.load_state_dict(payload["model_ema"])  # type: ignore[index]
        if "optimizer" in payload:
            try:
                opt.load_state_dict(payload["optimizer"])  # type: ignore[index]
            except Exception:
                # Optimizer shapes may differ if config changed; continue without optimizer state
                pass

    # Progress bar from start_step+1 to cfg.steps inclusive
    if start_step >= cfg.steps:
        print(f"Checkpoint step {start_step} >= target steps {cfg.steps}; nothing to do.")
        return

    pbar = tqdm(range(start_step + 1, cfg.steps + 1), desc="Training", dynamic_ncols=True)
    for step in pbar:
        model.train()

        # Sample batch and timesteps
        x0 = sample_batch(ds, cfg.batch_size, device)
        t = torch.randint(low=0, high=cfg.timesteps, size=(cfg.batch_size,), device=device, dtype=torch.long)

        # Forward diffusion and noise prediction
        x_t, eps = schedule.forward(x0, t)
        eps_pred = model(x_t, t)

        loss = F.mse_loss(eps_pred, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # EMA update
        with torch.no_grad():
            d = cfg.ema_decay
            for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

        pbar.set_postfix(loss=f"{loss.item():.4f}", device=str(device))

        # Checkpointing
        if step % cfg.ckpt_freq == 0 or step == cfg.steps:
            ckpt_dir = Path(cfg.ckpt_dir)
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", step, model, model_ema, opt, cfg, schedule)
            save_checkpoint(ckpt_dir / "latest.pt", step, model, model_ema, opt, cfg, schedule)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a 2D diffusion noise predictor on the SmileyDataset")
    p.add_argument("--steps", type=int, default=10000, help="Total training steps")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size")
    p.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    p.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping max norm (None to disable)")
    p.add_argument("--timesteps", type=int, default=300, help="Number of diffusion steps in the schedule")
    p.add_argument("--beta-start", type=float, default=1e-4, help="Linear schedule beta start")
    p.add_argument("--beta-end", type=float, default=2e-2, help="Linear schedule beta end")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--ckpt-freq", type=int, default=1000, help="Checkpoint save frequency (steps)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (None for random)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for model weights")
    p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training. If given without a path, loads <ckpt-dir>/latest.pt; else provide a checkpoint path.",
    )
    return p


def main():
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        ckpt_dir=args.ckpt_dir,
        ckpt_freq=args.ckpt_freq,
        seed=None if args.seed is None else int(args.seed),
        device=args.device,
        ema_decay=args.ema_decay,
        resume=args.resume,
    )
    train(cfg)


if __name__ == "__main__":
    main()
