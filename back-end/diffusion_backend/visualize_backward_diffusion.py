import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider
from tqdm import tqdm

from diffusion_backend.model import NoisePredictor
from diffusion_backend.noise_schedule import LinearNoiseSchedule


def detect_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_schedule(ckpt_path: Path, device: torch.device, sched_seed: Optional[int] = None) -> tuple[NoisePredictor, LinearNoiseSchedule, dict]:
    payload = torch.load(ckpt_path, map_location=device)
    sched_meta = payload.get("schedule", {}) if isinstance(payload, dict) else {}
    timesteps = int(sched_meta.get("timesteps", 300))
    beta_start = float(sched_meta.get("beta_start", 1e-4))
    beta_end = float(sched_meta.get("beta_end", 2e-2))

    model = NoisePredictor(num_timesteps=timesteps).to(device)
    state_key = "model_ema" if isinstance(payload, dict) and "model_ema" in payload else "model"
    model.load_state_dict(payload[state_key])  # type: ignore[index]
    model.eval()

    schedule = LinearNoiseSchedule(
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        seed=sched_seed,
    )
    return model, schedule, payload


def precompute_trajectory(
    n: int,
    model: NoisePredictor,
    schedule: LinearNoiseSchedule,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # Start from standard normal at t = T-1
    T = schedule.T
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    x_t = torch.randn((n, 2), device=device, dtype=torch.float32, generator=g)

    traj = torch.empty((T, n, 2), dtype=torch.float32)

    with torch.no_grad():
        for t_idx in tqdm(range(T - 1, -1, -1), desc="Precompute", dynamic_ncols=True):
            traj[t_idx] = x_t.detach().cpu()
            t_batch = torch.full((n,), t_idx, dtype=torch.long, device=device)
            eps_pred = model(x_t, t_batch)
            # Use the full DDPM reverse step (adds sigma_t * z)
            x_t = schedule.backward(x_t, eps_pred, t_batch)

    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive backward diffusion visualization (reverse sampling)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a training checkpoint .pt file")
    parser.add_argument("-n", "--num", type=int, default=3000, help="Number of points to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Seed for initial noise")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed for reverse noise (DDPM stochastic steps)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Compute device")
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter marker size")
    parser.add_argument("--alpha", type=float, default=0.8, help="Point alpha (transparency)")
    parser.add_argument("--color", type=str, default="#1f77b4", help="Point color")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI for display")
    parser.add_argument("--margin", type=float, default=0.15, help="Relative padding around data for autoscaling")
    args = parser.parse_args()

    device = detect_device(args.device)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model, schedule, _payload = load_model_and_schedule(ckpt, device, sched_seed=args.noise_seed)

    # Precompute reverse trajectory for smooth scrubbing
    traj = precompute_trajectory(
        args.num,
        model,
        schedule,
        device,
        seed=args.seed,
    )
    T = traj.shape[0]

    # Set up interactive plot
    plt.rcParams["figure.dpi"] = args.dpi
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(bottom=0.14)

    xt = traj[T - 1].numpy()
    sc = ax.scatter(xt[:, 0], xt[:, 1], s=args.point_size, c=args.color, alpha=args.alpha, linewidths=0, marker=".")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Precompute per-timestep square bounds for autoscaling
    mn = traj.amin(dim=1)  # [T, 2]
    mx = traj.amax(dim=1)  # [T, 2]
    center = 0.5 * (mn + mx)  # [T, 2]
    extent = (mx - mn)  # [T, 2]
    side = torch.maximum(extent[:, 0], extent[:, 1]) * (1.0 + args.margin)  # [T]
    side = torch.clamp(side, min=1e-3)
    cx_np = center[:, 0].numpy()
    cy_np = center[:, 1].numpy()
    half_np = (0.5 * side).numpy()

    # Initialize limits for t = T-1
    cx0, cy0, h0 = cx_np[T - 1], cy_np[T - 1], half_np[T - 1]
    ax.set_xlim(cx0 - h0, cx0 + h0)
    ax.set_ylim(cy0 - h0, cy0 + h0)

    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.04])
    slider = Slider(ax_slider, "t", 0, T - 1, valinit=T - 1, valfmt="%0.0f")

    title = ax.set_title(f"Backward diffusion: t={T-1} → 0")

    def update(_val):
        idx = int(round(slider.val))
        xt = traj[idx].numpy()
        sc.set_offsets(xt)
        cx, cy, h = cx_np[idx], cy_np[idx], half_np[idx]
        ax.set_xlim(cx - h, cx + h)
        ax.set_ylim(cy - h, cy + h)
        title.set_text(f"Backward diffusion: t={idx} → 0")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Keyboard shortcuts
    def on_key(event):
        if event.key in ("right", "]"):
            new_val = min(T - 1, int(round(slider.val)) + 1)
            slider.set_val(new_val)
        elif event.key in ("left", "["):
            new_val = max(0, int(round(slider.val)) - 1)
            slider.set_val(new_val)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
