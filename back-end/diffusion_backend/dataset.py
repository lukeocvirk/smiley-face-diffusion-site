import math
from typing import Optional, Tuple

import torch


class SmileyDataset:
    """Sampler for a 2D "smiley face" distribution.

    Repeated calls to `sample()` draw a single 2D point from a mixture of
    simple geometric primitives shaped to resemble a smiley face:
      - outer face circle (outline)
      - left and right eye ellipses (filled)
      - triangular nose (filled)
      - smiling mouth arc (outline)

    Plotting many samples should reveal the smiley outline with reasonable
    density across features (more points on face outline and mouth, fewer on
    nose/eyes).
    """

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        """Create the sampler.

        Args:
            seed: Optional RNG seed for reproducibility.
            device: Optional torch device string (e.g., "cpu", "cuda").
        """
        self.device = device or "cpu"
        self._g = torch.Generator(device=self.device)
        if seed is not None:
            self._g.manual_seed(seed)
        else:
            # Reseed from entropy so multiple instances vary by default
            self._g.seed()

        # Mixture weights across features (sum to 1.0)
        # Heavier weight on outer face and mouth for clearer structure.
        self._mix = {
            "face": 0.38,
            "mouth": 0.33,
            "eye_l": 0.12,
            "eye_r": 0.12,
            "nose": 0.05,
        }

    # ----------------------------- utils ---------------------------------
    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self._g, device=self.device)

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self._g, device=self.device)

    @staticmethod
    def _unit_perp(dx: torch.Tensor, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Perpendicular unit vector to (dx, dy)
        length = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-8)
        return -dy / length, dx / length

    # --------------------------- primitives -------------------------------
    def _sample_circle_ring(
        self,
        center: Tuple[float, float],
        radius: float,
        thickness: float,
    ) -> torch.Tensor:
        # Uniform angle with small radial jitter to form a ring.
        theta = self._rand(1) * (2 * math.pi)
        r = radius + self._randn(1) * thickness
        x = center[0] + r * torch.cos(theta)
        y = center[1] + r * torch.sin(theta)
        return torch.stack((x.squeeze(), y.squeeze())).to(torch.float32)

    def _sample_ellipse_ring(
        self,
        center: Tuple[float, float],
        a: float,
        b: float,
        thickness: float,
    ) -> torch.Tensor:
        # Elliptical ring via angle + small semi-axis jitter for thickness.
        theta = self._rand(1) * (2 * math.pi)
        da = self._randn(1) * thickness
        db = self._randn(1) * thickness
        x = center[0] + (a + da) * torch.cos(theta)
        y = center[1] + (b + db) * torch.sin(theta)
        return torch.stack((x.squeeze(), y.squeeze())).to(torch.float32)

    def _sample_ellipse_filled(
        self,
        center: Tuple[float, float],
        a: float,
        b: float,
    ) -> torch.Tensor:
        # Uniformly sample inside an ellipse using a unit-disk mapping.
        # For uniform area: radius ~ sqrt(U), theta ~ U[0, 2pi].
        theta = self._rand(1) * (2 * math.pi)
        r = torch.sqrt(self._rand(1))
        x = center[0] + a * r * torch.cos(theta)
        y = center[1] + b * r * torch.sin(theta)
        return torch.stack((x.squeeze(), y.squeeze())).to(torch.float32)

    def _sample_arc_ring(
        self,
        center: Tuple[float, float],
        radius: float,
        theta_min: float,
        theta_max: float,
        thickness: float,
    ) -> torch.Tensor:
        # Arc of a circle with small radial jitter for thickness.
        theta = theta_min + self._rand(1) * (theta_max - theta_min)
        r = radius + self._randn(1) * thickness
        x = center[0] + r * torch.cos(theta)
        y = center[1] + r * torch.sin(theta)
        return torch.stack((x.squeeze(), y.squeeze())).to(torch.float32)

    def _sample_segment_ring(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        thickness: float,
    ) -> torch.Tensor:
        # Point on a segment with perpendicular jitter for a thin outline.
        t = self._rand(1)
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])

        dx = torch.tensor(p1[0] - p0[0], device=self.device)
        dy = torch.tensor(p1[1] - p0[1], device=self.device)
        nx, ny = self._unit_perp(dx, dy)
        j = self._randn(1) * thickness
        x = x + j * nx
        y = y + j * ny
        return torch.stack((x.squeeze(), y.squeeze())).to(torch.float32)

    def _sample_triangle_filled(
        self,
        A: Tuple[float, float],
        B: Tuple[float, float],
        C: Tuple[float, float],
    ) -> torch.Tensor:
        # Uniformly sample inside triangle ABC using barycentric coords.
        u = self._rand(1)
        v = self._rand(1)
        # Fold across the diagonal to ensure uniform area.
        mask = (u + v > 1).to(torch.float32)
        u = u * (1 - mask) + (1 - u) * mask
        v = v * (1 - mask) + (1 - v) * mask

        Ax, Ay = A
        Bx, By = B
        Cx, Cy = C
        x = Ax + u * (Bx - Ax) + v * (Cx - Ax)
        y = Ay + u * (By - Ay) + v * (Cy - Ay)
        return torch.stack((x.squeeze(), y.squeeze())).to(torch.float32)

    # ---------------------------- public API ------------------------------
    def sample(self) -> torch.Tensor:
        """Draw a single 2D point shaped like a smiley face.

        Returns:
            torch.Tensor: shape (2,), dtype float32
        """
        # Select which feature to sample from, according to mixture weights
        u = self._rand(1).item()
        thresholds = {}
        cumsum = 0.0
        for k, w in self._mix.items():
            cumsum += float(w)
            thresholds[k] = cumsum

        # Feature geometry (tuned for a clear smiley at unit scale)
        if u < thresholds["face"]:
            # Outer face circle
            return self._sample_circle_ring(center=(0.0, 0.0), radius=1.0, thickness=0.02)

        if u < thresholds["mouth"]:
            # Mouth: lower circular arc (smile)
            # Arc angles ~ 200° to 340° around center slightly below origin
            deg = math.pi / 180.0
            return self._sample_arc_ring(
                center=(0.0, -0.15),
                radius=0.65,
                theta_min=200.0 * deg,
                theta_max=340.0 * deg,
                thickness=0.02,
            )

        if u < thresholds["mouth"] + self._mix["eye_l"]:
            # Left eye: filled ellipse
            return self._sample_ellipse_filled(center=(-0.35, 0.30), a=0.12, b=0.085)

        if u < thresholds["mouth"] + self._mix["eye_l"] + self._mix["eye_r"]:
            # Right eye: filled ellipse
            return self._sample_ellipse_filled(center=(0.35, 0.30), a=0.12, b=0.085)

        # Nose: small filled triangle
        A = (-0.07, 0.08)
        B = (0.00, -0.02)
        C = (0.07, 0.08)
        return self._sample_triangle_filled(A, B, C)


__all__ = ["SmileyDataset"]
