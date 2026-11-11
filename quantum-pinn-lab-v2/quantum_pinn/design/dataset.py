"""Finite-difference ground-truth dataset generation for conditional surrogate training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantum_pinn.config import Domain
from quantum_pinn.design.parameter_space import PARAMETER_SPACES, sample_parameters
from quantum_pinn.physics.potentials import make_potential
from quantum_pinn.physics.reference import solve_reference


@dataclass
class SurrogateDataset:
    family: str
    x: np.ndarray  # [grid]
    parameters: np.ndarray  # [configurations, parameter_count]
    energies: np.ndarray  # [configurations, states]
    wavefunctions: np.ndarray  # [configurations, states, grid]
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    domain: Domain
    seed: int

    @property
    def n_states(self) -> int:
        return int(self.energies.shape[1])

    @property
    def n_configurations(self) -> int:
        return int(self.parameters.shape[0])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            family=np.asarray(self.family), x=self.x, parameters=self.parameters,
            energies=self.energies, wavefunctions=self.wavefunctions,
            train_indices=self.train_indices, heldout_indices=self.heldout_indices,
            xmin=np.asarray(self.domain.xmin), xmax=np.asarray(self.domain.xmax), seed=np.asarray(self.seed),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateDataset":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                family=str(data["family"]), x=data["x"], parameters=data["parameters"],
                energies=data["energies"], wavefunctions=data["wavefunctions"],
                train_indices=data["train_indices"], heldout_indices=data["heldout_indices"],
                domain=Domain(float(data["xmin"]), float(data["xmax"])), seed=int(data["seed"]),
            )


def generate_dataset(
    family: str,
    n_configurations: int = 32,
    n_grid: int = 301,
    n_states: int = 2,
    train_fraction: float = 0.80,
    seed: int = 42,
    domain: Domain = Domain(),
) -> SurrogateDataset:
    """Generate FD labels; split by parameter combination before any model fitting."""
    if family not in PARAMETER_SPACES:
        raise ValueError(f"family must be one of {list(PARAMETER_SPACES)}")
    if n_configurations < 8:
        raise ValueError("n_configurations must be >=8 to retain a useful held-out split")
    if n_grid < 101:
        raise ValueError("n_grid must be >=101")
    if n_states < 1 or n_states > 4:
        raise ValueError("n_states must be between 1 and 4")
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")

    params = sample_parameters(family, n_configurations, seed)
    energies: list[np.ndarray] = []
    states: list[np.ndarray] = []
    x_grid: np.ndarray | None = None
    for row in params:
        family_params = dict(zip((b.name for b in PARAMETER_SPACES[family]), row.tolist(), strict=True))
        reference = solve_reference(make_potential(family, **family_params), domain, n_grid, n_states)
        if x_grid is None:
            x_grid = reference.x
        energies.append(reference.energies)
        states.append(reference.wavefunctions.T)

    rng = np.random.default_rng(seed + 1009)
    permutation = rng.permutation(n_configurations)
    n_train = min(n_configurations - 1, max(2, int(round(n_configurations * train_fraction))))
    train_indices = np.sort(permutation[:n_train])
    heldout_indices = np.sort(permutation[n_train:])
    assert x_grid is not None
    return SurrogateDataset(
        family=family, x=x_grid, parameters=params,
        energies=np.asarray(energies), wavefunctions=np.asarray(states),
        train_indices=train_indices, heldout_indices=heldout_indices,
        domain=domain, seed=seed,
    )
