import math
from typing import Optional, Tuple

import torch


class LinearNoiseSchedule:
    """DDPM-style linear beta schedule for 2D smiley points (or any R^D).

    Implements forward diffusion x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) * eps
    and one reverse step using predicted noise eps_hat:
      x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_hat)
                 + sigma_t * z,  with z ~ N(0, I) and sigma_t = sqrt(beta_tilde_t).

    Indexing uses t in [0, T-1]. For t == 0, sigma_0 = 0 and the reverse step becomes deterministic.
    """

    def __init__(
        self,
        timesteps: int = 300,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> None:
        self.T = int(timesteps)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # RNG for reproducibility if desired
        self._g = torch.Generator(device=self.device)
        if seed is not None:
            self._g.manual_seed(seed)
        else:
            self._g.seed()

        # Linear betas in [beta_start, beta_end]
        betas = torch.linspace(beta_start, beta_end, self.T, device=self.device, dtype=self.dtype)
        betas = betas.clamp(1e-8, 0.999)  # numerical safety

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, device=self.device, dtype=self.dtype), alpha_bars[:-1]], dim=0)

        # Precompute commonly used terms
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bar_prev = alpha_bar_prev
        self.sqrt_alpha_bar = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
        # Beta tilde (variance of q(x_{t-1} | x_t, x_0))
        self.sigma2 = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bars)) * betas
        self.sigma = torch.sqrt(self.sigma2.clamp_min(0.0))

    @staticmethod
    def _expand_like(a_b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Expand a 1D batch vector [B] to broadcast across x's remaining dims
        return a_b.view(x.shape[0], *([1] * (x.dim() - 1)))

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: q(x_t | x_0).

        Args:
            x0: clean points, shape [B, ...]
            t: timesteps, LongTensor shape [B], values in [0, T-1]
            noise: optional noise epsilon ~ N(0, I); if None, sampled.

        Returns:
            (x_t, eps) where both have shape like x0.
        """
        if t.dtype != torch.long:
            t = t.to(torch.long)
        if x0.device != self.device or x0.dtype != self.dtype:
            # Align schedule buffers to input dynamically for convenience
            self.to(device=x0.device, dtype=x0.dtype)

        if noise is None:
            noise = torch.randn(x0.shape, generator=self._g, device=x0.device, dtype=x0.dtype)

        s_ab = self.sqrt_alpha_bar[t]  # [B]
        s_1mab = self.sqrt_one_minus_alpha_bar[t]  # [B]
        x_t = self._expand_like(s_ab, x0) * x0 + self._expand_like(s_1mab, x0) * noise
        return x_t, noise

    def backward(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse diffusion step: p_theta(x_{t-1} | x_t).

        Args:
            x_t: noisy points at time t, shape [B, ...]
            eps_pred: predicted noise for x_t, shape like x_t
            t: timesteps (for x_t), LongTensor shape [B], values in [0, T-1]

        Returns:
            x_{t-1} sample, shape like x_t.
        """
        if t.dtype != torch.long:
            t = t.to(torch.long)
        if x_t.device != self.device or x_t.dtype != self.dtype:
            self.to(device=x_t.device, dtype=x_t.dtype)

        a_t = self.alphas[t]  # [B]
        sqrt_recip_a_t = self.sqrt_recip_alpha[t]  # [B]
        s1mab_t = self.sqrt_one_minus_alpha_bar[t]  # [B]
        beta_tilde_sqrt = self.sigma[t]  # [B]

        coeff = (1.0 - a_t) / s1mab_t  # [B]
        mean = self._expand_like(sqrt_recip_a_t, x_t) * (
            x_t - self._expand_like(coeff, x_t) * eps_pred
        )

        # For t == 0, sigma==0 (deterministic); for others add noise
        z = torch.randn(x_t.shape, generator=self._g, device=x_t.device, dtype=x_t.dtype)
        x_prev = mean + self._expand_like(beta_tilde_sqrt, x_t) * z
        return x_prev

    # Utility to move precomputed buffers
    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "LinearNoiseSchedule":
        device = device or self.device
        dtype = dtype or self.dtype
        for name in [
            "betas",
            "alphas",
            "alpha_bars",
            "alpha_bar_prev",
            "sqrt_alpha_bar",
            "sqrt_one_minus_alpha_bar",
            "sqrt_recip_alpha",
            "sigma2",
            "sigma",
        ]:
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device=device, dtype=dtype))
        self.device = device
        self.dtype = dtype
        if self._g.device != device:
            # Recreate generator on the new device; reseed to preserve stream continuity isn't necessary here.
            seed = self._g.seed()  # grab a fresh seed
            self._g = torch.Generator(device=device)
            self._g.manual_seed(seed)
        return self


# Convenience default schedule and module-level functions
_DEFAULT_SCHEDULE = LinearNoiseSchedule()


def forward(x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
    return _DEFAULT_SCHEDULE.forward(x0, t, noise)


def backward(x_t: torch.Tensor, eps_pred: torch.Tensor, t: torch.Tensor):
    return _DEFAULT_SCHEDULE.backward(x_t, eps_pred, t)


__all__ = [
    "LinearNoiseSchedule",
    "forward",
    "backward",
]
