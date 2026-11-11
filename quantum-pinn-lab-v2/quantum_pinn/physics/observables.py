"""Expectation values and uncertainty diagnostics for 1D states."""
from __future__ import annotations

import numpy as np


def normalized_state(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    norm2 = np.trapezoid(np.abs(psi) ** 2, x)
    if not np.isfinite(norm2) or norm2 <= 1e-15:
        raise ValueError("Wavefunction has zero or invalid norm")
    return psi / np.sqrt(norm2)


def calculate_observables(x: np.ndarray, psi: np.ndarray, potential: np.ndarray) -> dict[str, float]:
    """Return dimensionless observables with hbar=m=1 and H=-1/2 d²+V."""
    x = np.asarray(x, dtype=float)
    psi = normalized_state(x, np.asarray(psi))
    density = np.abs(psi) ** 2
    dpsi = np.gradient(psi, x, edge_order=2)
    x_mean = float(np.trapezoid(x * density, x).real)
    x2_mean = float(np.trapezoid(x**2 * density, x).real)
    p_mean = float(np.trapezoid(np.conjugate(psi) * (-1j * dpsi), x).real)
    p2_mean = float(np.trapezoid(np.abs(dpsi) ** 2, x).real)
    potential_energy = float(np.trapezoid(np.asarray(potential) * density, x).real)
    kinetic_energy = 0.5 * p2_mean
    return {
        "norm": float(np.trapezoid(density, x).real),
        "x_mean": x_mean,
        "x2_mean": x2_mean,
        "sigma_x": float(np.sqrt(max(0.0, x2_mean - x_mean**2))),
        "p_mean": p_mean,
        "p2_mean": p2_mean,
        "sigma_p": float(np.sqrt(max(0.0, p2_mean - p_mean**2))),
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": kinetic_energy + potential_energy,
    }
