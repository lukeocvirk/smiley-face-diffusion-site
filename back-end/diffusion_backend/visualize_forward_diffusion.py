import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

from diffusion_backend import SmileyDataset
from diffusion_backend.noise_schedule import LinearNoiseSchedule


def build_points(n: int, seed: Optional[int]) -> torch.Tensor:
    ds = SmileyDataset(seed=seed)
    pts = torch.stack([ds.sample() for _ in range(n)])  # [N, 2]
    return pts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive forward diffusion visualization with a slider",
    )
    parser.add_argument("-n", "--num", type=int, default=5000, help="Number of points to sample")
    parser.add_argument("-T", "--timesteps", type=int, default=300, help="Number of diffusion steps (schedule length)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for dataset sampling")
    parser.add_argument("--noise-seed", type=int, default=None, help="Seed for fixed forward noise eps")
    parser.add_argument("--sched-seed", type=int, default=None, help="Seed for schedule RNG (not critical here)")
    parser.add_argument("--point-size", type=float, default=2.0, help="Scatter marker size")
    parser.add_argument("--alpha", type=float, default=0.8, help="Point alpha (transparency)")
    parser.add_argument("--color", type=str, default="#1f77b4", help="Point color")
    parser.add_argument("--dpi", type=int, default=120, help="Figure DPI for display")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Data and schedule
    x0 = build_points(args.num, args.seed).to(device)
    schedule = LinearNoiseSchedule(timesteps=args.timesteps, device=device, seed=args.sched_seed)

    # Fixed noise per point for consistent scrubbing using the closed-form q(x_t|x_0)
    g = torch.Generator(device=device)
    if args.noise_seed is not None:
        g.manual_seed(args.noise_seed)
    else:
        g.seed()
    eps = torch.randn(x0.shape, generator=g, device=device, dtype=x0.dtype)

    def compute_xt(t_idx: int) -> np.ndarray:
        t_tensor = torch.full((x0.shape[0],), int(t_idx), dtype=torch.long, device=device)
        x_t, _ = schedule.forward(x0, t_tensor, noise=eps)
        return x_t.cpu().numpy()

    # Initial plot
    plt.rcParams["figure.dpi"] = args.dpi
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(bottom=0.14)

    xt0 = compute_xt(0)
    sc = ax.scatter(xt0[:, 0], xt0[:, 1], s=args.point_size, c=args.color, alpha=args.alpha, linewidths=0, marker=".")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Slider setup
    ax_slider = plt.axes([0.15, 0.04, 0.7, 0.04])
    slider = Slider(ax_slider, "t", 0, args.timesteps - 1, valinit=0, valfmt="%0.0f")

    title = ax.set_title(f"Forward diffusion: t=0 / {args.timesteps-1}")

    def update(val):
        t_idx = int(round(slider.val))
        xt = compute_xt(t_idx)
        sc.set_offsets(xt)
        title.set_text(f"Forward diffusion: t={t_idx} / {args.timesteps-1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Keyboard shortcuts for scrubbing
    def on_key(event):
        if event.key in ("right", "]"):
            new_val = min(args.timesteps - 1, int(round(slider.val)) + 1)
            slider.set_val(new_val)
        elif event.key in ("left", "["):
            new_val = max(0, int(round(slider.val)) - 1)
            slider.set_val(new_val)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
