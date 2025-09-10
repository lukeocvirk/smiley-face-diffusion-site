import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from diffusion_backend import SmileyDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot samples from the SmileyDataset")
    parser.add_argument("-n", "--num", type=int, default=10000, help="Number of points to sample")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--out", type=str, default=None, help="Optional output image path to save instead of showing")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving")
    args = parser.parse_args()

    ds = SmileyDataset(seed=args.seed)

    # Collect samples
    pts = torch.stack([ds.sample() for _ in range(args.num)])  # (N, 2)
    x = pts[:, 0].cpu().numpy()
    y = pts[:, 1].cpu().numpy()

    # Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=2, c="#1f77b4", alpha=0.8, linewidths=0, marker=".")
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
