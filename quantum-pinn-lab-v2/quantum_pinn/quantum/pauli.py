"""Map a finite-difference Hamiltonian to qubit Pauli strings.

This is an exact matrix-to-Pauli encoding for small power-of-two Hilbert spaces.
It is a bridge for quantum-computing experiments, not a claim of quantum advantage.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.sparse import diags

from quantum_pinn.physics.potentials import Potential


@dataclass(frozen=True)
class PauliTerm:
    label: str
    coefficient: complex


def build_grid_hamiltonian(
    potential: Potential,
    n_qubits: int = 4,
    xmin: float = -6.0,
    xmax: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense Dirichlet finite-difference H on 2**n_qubits interior points."""
    if not 1 <= n_qubits <= 8:
        raise ValueError("n_qubits must be in [1, 8] for this demonstrator")
    n = 2**n_qubits
    dx = (xmax - xmin) / (n + 1)
    x = xmin + dx * np.arange(1, n + 1)
    main = 1.0 / dx**2 + potential.numpy(x)
    off = np.full(n - 1, -0.5 / dx**2)
    h = diags([off, main, off], [-1, 0, 1]).toarray().astype(np.complex128)
    return x, h


def _pauli_trace_coefficient(h: np.ndarray, label: str) -> complex:
    n = h.shape[0]
    indices = np.arange(n, dtype=np.int64)
    row = indices.copy()
    phase = np.ones(n, dtype=np.complex128)
    n_qubits = len(label)
    for char_index, char in enumerate(label):
        bit_position = n_qubits - 1 - char_index
        bits = (indices >> bit_position) & 1
        if char in ("X", "Y"):
            row ^= 1 << bit_position
        if char == "Y":
            phase *= 1j * np.where(bits == 0, 1.0, -1.0)
        elif char == "Z":
            phase *= np.where(bits == 0, 1.0, -1.0)
    # Tr(P H) = sum_c P[row(c), c] H[c, row(c)].
    return complex(np.sum(phase * h[indices, row]) / n)


def pauli_decompose(h: np.ndarray, atol: float = 1.0e-10) -> list[PauliTerm]:
    """Return coefficients c_P in H = sum_P c_P P for an N=2**n Hermitian matrix."""
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Hamiltonian must be square")
    n = h.shape[0]
    if n < 2 or n & (n - 1):
        raise ValueError("Hamiltonian dimension must be a power of two >=2")
    if not np.allclose(h, h.conj().T, atol=1e-9):
        raise ValueError("Hamiltonian must be Hermitian")
    n_qubits = int(np.log2(n))
    terms: list[PauliTerm] = []
    for chars in product("IXYZ", repeat=n_qubits):
        label = "".join(chars)
        coeff = _pauli_trace_coefficient(h, label)
        if abs(coeff) > atol:
            # Numerical round-off can leave tiny imaginary parts for Hermitian H.
            if abs(coeff.imag) < 1e-10:
                coeff = complex(coeff.real, 0.0)
            terms.append(PauliTerm(label, coeff))
    return terms


def pauli_matrix(label: str) -> np.ndarray:
    matrices = {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    result = np.array([[1.0 + 0.0j]])
    for char in label:
        if char not in matrices:
            raise ValueError(f"Invalid Pauli character {char}")
        result = np.kron(result, matrices[char])
    return result


def reconstruct_from_paulis(terms: Iterable[PauliTerm]) -> np.ndarray:
    terms = list(terms)
    if not terms:
        raise ValueError("At least one Pauli term is required")
    dim = 2 ** len(terms[0].label)
    result = np.zeros((dim, dim), dtype=np.complex128)
    for term in terms:
        result += term.coefficient * pauli_matrix(term.label)
    return result


def to_qiskit_sparse_pauli_op(terms: Iterable[PauliTerm]):
    """Optional adapter. Install the qiskit extra to use this function."""
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as exc:
        raise ImportError("Qiskit is optional. Install with `pip install -e '.[quantum]'`.") from exc
    return SparsePauliOp.from_list([(term.label, term.coefficient) for term in terms]).simplify()
