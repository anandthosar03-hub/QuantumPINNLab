"""Crank-Nicolson time propagation for one-dimensional wavepackets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from quantum_pinn.config import Domain
from quantum_pinn.physics.potentials import Potential


@dataclass
class DynamicsResult:
    x: np.ndarray
    times: np.ndarray
    wavefunctions: np.ndarray  # (n_saved_times, n_grid), complex
    norms: np.ndarray
    mean_positions: np.ndarray


def gaussian_wavepacket(x: np.ndarray, x0: float = -2.5, sigma: float = 0.45, k0: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    psi = np.exp(-((x - x0) ** 2) / (4.0 * sigma**2)) * np.exp(1j * k0 * x)
    psi[[0, -1]] = 0.0
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    return psi / norm


def propagate_crank_nicolson(
    potential: Potential,
    domain: Domain = Domain(),
    n_grid: int = 501,
    dt: float = 0.002,
    n_steps: int = 1500,
    save_every: int = 25,
    x0: float = -2.5,
    sigma: float = 0.45,
    k0: float = 3.0,
) -> DynamicsResult:
    """Propagate i dpsi/dt = H psi, using hbar=m=1 and fixed zero boundaries."""
    if n_grid < 101 or dt <= 0 or n_steps < 1 or save_every < 1:
        raise ValueError("Invalid grid or time-step configuration")
    x = np.linspace(domain.xmin, domain.xmax, n_grid)
    dx = x[1] - x[0]
    xi = x[1:-1]
    v = potential.numpy(xi)
    n = xi.size
    main = np.full(n, 1.0 / dx**2) + v
    off = np.full(n - 1, -0.5 / dx**2)
    hamiltonian = diags([off, main, off], [-1, 0, 1], format="csc", dtype=np.complex128)
    identity = eye(n, dtype=np.complex128, format="csc")
    a = identity + 0.5j * dt * hamiltonian
    b = identity - 0.5j * dt * hamiltonian
    factor = splu(a)

    psi_full = gaussian_wavepacket(x, x0=x0, sigma=sigma, k0=k0)
    psi = psi_full[1:-1].astype(np.complex128)
    saved_t: list[float] = [0.0]
    saved_psi: list[np.ndarray] = [psi_full.copy()]
    norms: list[float] = [float(np.trapezoid(np.abs(psi_full) ** 2, x))]
    means: list[float] = [float(np.trapezoid(x * np.abs(psi_full) ** 2, x).real)]

    for step in range(1, n_steps + 1):
        psi = factor.solve(b @ psi)
        if step % save_every == 0 or step == n_steps:
            current = np.zeros(n_grid, dtype=np.complex128)
            current[1:-1] = psi
            density = np.abs(current) ** 2
            saved_t.append(step * dt)
            saved_psi.append(current)
            norms.append(float(np.trapezoid(density, x).real))
            means.append(float(np.trapezoid(x * density, x).real))

    return DynamicsResult(
        x=x,
        times=np.asarray(saved_t),
        wavefunctions=np.asarray(saved_psi),
        norms=np.asarray(norms),
        mean_positions=np.asarray(means),
    )


def probability_current(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Dimensionless probability current J = Im(psi* dpsi/dx)."""
    return np.imag(np.conjugate(psi) * np.gradient(psi, x, edge_order=2))
