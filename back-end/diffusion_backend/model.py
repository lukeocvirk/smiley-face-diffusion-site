from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Learnable per-timestep embeddings used to condition AdaLN.

    Each timestep in [0, num_timesteps-1] maps to a cond_dim vector.
    """

    def __init__(self, num_timesteps: int, cond_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_timesteps, cond_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.long:
            t = t.to(torch.long)
        return self.embed(t)


class AdaptiveLayerNorm(nn.Module):
    """LayerNorm with per-sample, time-conditioned scale and shift.

    Uses a base LayerNorm without affine parameters. The conditioning vector
    is mapped to [scale, shift] and applied as y = LN(x) * (1 + scale) + shift.
    """

    def __init__(self, normalized_shape: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * normalized_shape)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return h * (1 + scale) + shift


class ResidualBlock(nn.Module):
    """Hidden-dim residual: H->H (GELU + AdaLN conditioned on t)."""

    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act = nn.GELU()
        self.adaln = AdaptiveLayerNorm(hidden_dim * 4, cond_dim)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.adaln(y, cond)
        y = self.fc2(y)
        return x + y


class NoisePredictor(nn.Module):
    """Noise prediction network for 2D diffusion on smiley points.

    Architecture:
      - Input embedding: Linear(2->2)
      - 3x ResidualBlock: each does 2->6->AdaLN(t)->2 + residual
      - Output projection: Linear(2->2) to predict noise epsilon
    """

    def __init__(
        self,
        num_timesteps: int = 300,
        cond_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(2, hidden_dim)
        self.time = TimeEmbedding(num_timesteps, cond_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, cond_dim) for _ in range(3)])
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon for noisy points x at timesteps t.

        Args:
            x: [B, 2] noisy points
            t: [B] integer timesteps

        Returns:
            eps_pred: [B, 2] predicted noise
        """
        h = self.in_proj(x)
        cond = self.time(t)
        for blk in self.blocks:
            h = blk(h, cond)
        eps = self.out_proj(h)
        return eps


__all__ = ["NoisePredictor"]
