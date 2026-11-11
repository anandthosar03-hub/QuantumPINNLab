"""High-accuracy finite-difference reference eigensolver for 1D Hamiltonians."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from quantum_pinn.config import Domain
from quantum_pinn.physics.potentials import Potential


@dataclass
class ReferenceResult:
    x: np.ndarray
    energies: np.ndarray
    wavefunctions: np.ndarray  # shape: (n_grid, n_states), real and normalized
    potential: np.ndarray


def solve_reference(
    potential: Potential,
    domain: Domain = Domain(),
    n_grid: int = 801,
    n_states: int = 4,
) -> ReferenceResult:
    """Solve [-1/2 d²/dx² + V(x)] psi = E psi with Dirichlet boundaries."""
    if n_grid < 101:
        raise ValueError("n_grid must be at least 101")
    if not 1 <= n_states <= n_grid - 2:
        raise ValueError("n_states must be between 1 and n_grid-2")
    x = np.linspace(domain.xmin, domain.xmax, n_grid, dtype=np.float64)
    dx = x[1] - x[0]
    xi = x[1:-1]
    v = potential.numpy(xi)

    # Three-point central difference for -1/2 d²/dx².
    diagonal = np.full(xi.size, 1.0 / dx**2, dtype=np.float64) + v
    off_diagonal = np.full(xi.size - 1, -0.5 / dx**2, dtype=np.float64)
    energies, vectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, n_states - 1),
        check_finite=True,
    )

    psi = np.zeros((n_grid, n_states), dtype=np.float64)
    psi[1:-1, :] = vectors
    for j in range(n_states):
        norm = np.sqrt(np.trapezoid(psi[:, j] ** 2, x))
        psi[:, j] /= max(norm, 1.0e-15)
        # Fix the arbitrary global sign for repeatable plots and warm starts.
        peak = int(np.argmax(np.abs(psi[:, j])))
        if psi[peak, j] < 0:
            psi[:, j] *= -1.0
    return ReferenceResult(x=x, energies=energies, wavefunctions=psi, potential=potential.numpy(x),)
