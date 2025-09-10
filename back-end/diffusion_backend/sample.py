import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from diffusion_backend import NoisePredictor
from diffusion_backend.noise_schedule import LinearNoiseSchedule


def detect_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> tuple[NoisePredictor, dict]:
    payload = torch.load(ckpt_path, map_location=device)
    sched_meta = payload.get("schedule", {}) if isinstance(payload, dict) else {}
    num_timesteps = int(sched_meta.get("timesteps", 300))
    model = NoisePredictor(num_timesteps=num_timesteps).to(device)
    state_key = "model_ema" if isinstance(payload, dict) and "model_ema" in payload else "model"
    model.load_state_dict(payload[state_key])  # type: ignore[index]
    model.eval()
    return model, payload


def build_schedule(meta: dict, device: torch.device) -> LinearNoiseSchedule:
    t = int(meta.get("timesteps", 1000))
    b0 = float(meta.get("beta_start", 1e-4))
    b1 = float(meta.get("beta_end", 2e-2))
    return LinearNoiseSchedule(timesteps=t, beta_start=b0, beta_end=b1, device=device)


def sample_points(
    n: int,
    model: NoisePredictor,
    schedule: LinearNoiseSchedule,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()

    # Start from standard normal at time T-1
    T = schedule.T
    x_t = torch.randn((n, 2), device=device, dtype=torch.float32, generator=g)

    model.eval()
    with torch.no_grad():
        for t_idx in tqdm(range(T - 1, -1, -1), desc="Sampling", dynamic_ncols=True):
            t_batch = torch.full((n,), t_idx, dtype=torch.long, device=device)
            eps_pred = model(x_t, t_batch)
            x_t = schedule.backward(x_t, eps_pred, t_batch)

    return x_t


def main():
    p = argparse.ArgumentParser(description="Reverse diffusion sampler and plotter")
    p.add_argument("--ckpt", type=str, required=True, help="Path to a training checkpoint .pt file")
    p.add_argument("-n", "--num", type=int, default=5000, help="Number of points to sample")
    p.add_argument("--seed", type=int, default=None, help="Seed for initial noise sampling")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    p.add_argument("--out", type=str, default=None, help="Optional image path to save plot")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI for saving")
    p.add_argument("--point-size", type=float, default=2.0, help="Scatter marker size")
    p.add_argument("--alpha", type=float, default=0.8, help="Point alpha")
    p.add_argument("--color", type=str, default="#1f77b4", help="Point color")
    args = p.parse_args()

    device = detect_device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, payload = load_model(ckpt_path, device)
    sched_meta = payload.get("schedule", {}) if isinstance(payload, dict) else {}
    schedule = build_schedule(sched_meta, device)

    x0 = sample_points(args.num, model, schedule, device, seed=args.seed)
    x = x0.detach().cpu().numpy()

    # Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=args.point_size, c=args.color, alpha=args.alpha, linewidths=0, marker=".")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.tight_layout(pad=0.1)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    else:
        plt.show()


if __name__ == "__main__":
    main()
