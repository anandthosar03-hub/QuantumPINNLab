"""Parameter spaces and differentiable potential definitions for v2 design studies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from scipy.stats import qmc


@dataclass(frozen=True)
class ParameterBound:
    name: str
    low: float
    high: float
    default: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f"Invalid bounds for {self.name}: low must be below high")
        if not self.low <= self.default <= self.high:
            raise ValueError(f"Default for {self.name} must lie inside its bounds")


PARAMETER_SPACES: dict[str, tuple[ParameterBound, ...]] = {
    "harmonic": (ParameterBound("omega", 0.50, 2.00, 1.00),),
    "quartic": (
        ParameterBound("omega", 0.50, 2.00, 1.00),
        ParameterBound("strength", 0.005, 0.080, 0.030),
    ),
    "double_well": (
        ParameterBound("barrier", 0.15, 1.50, 0.65),
        ParameterBound("separation", 0.70, 1.80, 1.20),
    ),
}


def parameter_names(family: str) -> tuple[str, ...]:
    try:
        return tuple(bound.name for bound in PARAMETER_SPACES[family])
    except KeyError as exc:
        raise ValueError(f"Unsupported parameter-conditioned family {family!r}; choose {list(PARAMETER_SPACES)}") from exc


def parameter_bounds(family: str) -> np.ndarray:
    try:
        return np.asarray([(b.low, b.high) for b in PARAMETER_SPACES[family]], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"Unsupported parameter-conditioned family {family!r}; choose {list(PARAMETER_SPACES)}") from exc


def default_parameters(family: str) -> dict[str, float]:
    try:
        return {b.name: b.default for b in PARAMETER_SPACES[family]}
    except KeyError as exc:
        raise ValueError(f"Unsupported parameter-conditioned family {family!r}") from exc


def validate_parameters(family: str, values: Mapping[str, float], *, enforce_bounds: bool = True) -> dict[str, float]:
    bounds = PARAMETER_SPACES.get(family)
    if bounds is None:
        raise ValueError(f"Unsupported family {family!r}; choose {list(PARAMETER_SPACES)}")
    expected = {b.name for b in bounds}
    if set(values) != expected:
        raise ValueError(f"{family} expects parameters {sorted(expected)}, got {sorted(values)}")
    result = {b.name: float(values[b.name]) for b in bounds}
    for bound in bounds:
        value = result[bound.name]
        if not np.isfinite(value):
            raise ValueError(f"{bound.name} must be finite")
        if enforce_bounds and not bound.low <= value <= bound.high:
            raise ValueError(f"{bound.name}={value} is outside [{bound.low}, {bound.high}]")
        if value <= 0:
            raise ValueError(f"{bound.name} must be positive")
    return result


def normalize_parameter_array(family: str, parameters: np.ndarray) -> np.ndarray:
    bounds = parameter_bounds(family)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape[-1] != bounds.shape[0]:
        raise ValueError("Last parameter dimension does not match the selected family")
    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def denormalize_parameter_tensor(family: str, normalized: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(parameter_bounds(family), dtype=normalized.dtype, device=normalized.device)
    return normalized * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def sample_parameters(family: str, n_samples: int, seed: int = 42) -> np.ndarray:
    """Latin-hypercube samples across the declared design box."""
    if n_samples < 4:
        raise ValueError("Use at least four parameter combinations so train/held-out evaluation is meaningful")
    bounds = parameter_bounds(family)
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed, optimization="random-cd")
    unit = sampler.random(n=n_samples)
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])


def potential_torch(family: str, x: torch.Tensor, actual_parameters: torch.Tensor) -> torch.Tensor:
    """Vectorized differentiable V(x; p). x and parameters have matching batch rows."""
    p = actual_parameters
    if family == "harmonic":
        omega = p[:, 0:1]
        return 0.5 * (omega * x) ** 2
    if family == "quartic":
        omega, strength = p[:, 0:1], p[:, 1:2]
        return 0.5 * (omega * x) ** 2 + strength * x**4
    if family == "double_well":
        barrier, separation = p[:, 0:1], p[:, 1:2]
        return barrier * (x**2 - separation**2) ** 2
    raise ValueError(f"No differentiable potential implemented for {family!r}")
