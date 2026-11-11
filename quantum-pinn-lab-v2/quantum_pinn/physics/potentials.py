"""Reusable one-dimensional model potentials (dimensionless units)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


POTENTIAL_DEFAULTS: dict[str, dict[str, float]] = {
    "free": {},
    "harmonic": {"omega": 1.0},
    "quartic": {"omega": 0.5, "strength": 0.03},
    "double_well": {"barrier": 0.5, "separation": 1.2},
    "finite_well": {"depth": 8.0, "width": 2.0},
    "gaussian_well": {"depth": 8.0, "sigma": 1.0},
    "barrier": {"height": 8.0, "width": 1.0},
    "soft_coulomb": {"charge": 1.0, "softening": 0.35},
}


@dataclass(frozen=True)
class Potential:
    name: str
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in POTENTIAL_DEFAULTS:
            raise ValueError(f"Unknown potential '{self.name}'. Choose from {list(POTENTIAL_DEFAULTS)}")
        merged = dict(POTENTIAL_DEFAULTS[self.name])
        unknown = set(self.params) - set(merged)
        if unknown:
            raise ValueError(f"Unsupported parameters for {self.name}: {sorted(unknown)}")
        merged.update({k: float(v) for k, v in self.params.items()})
        object.__setattr__(self, "params", merged)
        for key, value in merged.items():
            if not np.isfinite(value):
                raise ValueError(f"Potential parameter {key} must be finite")
            if value < 0:
                raise ValueError(f"Potential parameter {key} must be non-negative")
        for key in ("width", "sigma", "softening"):
            if key in merged and merged[key] <= 0:
                raise ValueError(f"Potential parameter {key} must be positive")

    def numpy(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        x = np.asarray(x, dtype=np.float64)
        if self.name == "free":
            return np.zeros_like(x)
        if self.name == "harmonic":
            return 0.5 * (p["omega"] * x) ** 2
        if self.name == "quartic":
            return 0.5 * (p["omega"] * x) ** 2 + p["strength"] * x**4
        if self.name == "double_well":
            return p["barrier"] * (x**2 - p["separation"]**2) ** 2
        if self.name == "finite_well":
            return np.where(np.abs(x) <= p["width"] / 2.0, -p["depth"], 0.0)
        if self.name == "gaussian_well":
            return -p["depth"] * np.exp(-0.5 * (x / p["sigma"]) ** 2)
        if self.name == "barrier":
            return np.where(np.abs(x) <= p["width"] / 2.0, p["height"], 0.0)
        if self.name == "soft_coulomb":
            return -p["charge"] / np.sqrt(x**2 + p["softening"]**2)
        raise RuntimeError("Unreachable potential branch")

    def torch(self, x: torch.Tensor) -> torch.Tensor:
        p = self.params
        if self.name == "free":
            return torch.zeros_like(x)
        if self.name == "harmonic":
            return 0.5 * (p["omega"] * x) ** 2
        if self.name == "quartic":
            return 0.5 * (p["omega"] * x) ** 2 + p["strength"] * x**4
        if self.name == "double_well":
            return p["barrier"] * (x**2 - p["separation"]**2) ** 2
        if self.name == "finite_well":
            return torch.where(torch.abs(x) <= p["width"] / 2.0, -p["depth"], torch.zeros_like(x))
        if self.name == "gaussian_well":
            return -p["depth"] * torch.exp(-0.5 * (x / p["sigma"]) ** 2)
        if self.name == "barrier":
            return torch.where(torch.abs(x) <= p["width"] / 2.0, p["height"], torch.zeros_like(x))
        if self.name == "soft_coulomb":
            return -p["charge"] / torch.sqrt(x**2 + p["softening"]**2)
        raise RuntimeError("Unreachable potential branch")

    def description(self) -> str:
        formulas = {
            "free": "V(x) = 0",
            "harmonic": "V(x) = ½ ω²x²",
            "quartic": "V(x) = ½ ω²x² + λx⁴",
            "double_well": "V(x) = a(x² − b²)²",
            "finite_well": "V(x) = −V₀ inside the well, 0 outside",
            "gaussian_well": "V(x) = −V₀ exp[−x²/(2σ²)]",
            "barrier": "V(x) = V₀ inside the barrier, 0 outside",
            "soft_coulomb": "V(x) = −Z / √(x² + ε²)",
        }
        return formulas[self.name]


def make_potential(name: str, **params: Any) -> Potential:
    return Potential(name=name, params=params)
