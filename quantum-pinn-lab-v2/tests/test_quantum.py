import numpy as np

from quantum_pinn.physics.potentials import make_potential
from quantum_pinn.quantum.pauli import (
    build_grid_hamiltonian,
    pauli_decompose,
    reconstruct_from_paulis,
)


def test_pauli_decomposition_reconstructs_one_qubit_hamiltonian():
    h = np.array([[1.0, 0.2], [0.2, 2.0]], dtype=np.complex128)
    terms = pauli_decompose(h)
    rebuilt = reconstruct_from_paulis(terms)
    assert np.linalg.norm(h - rebuilt) < 1e-10


def test_grid_hamiltonian_is_hermitian_and_decomposes():
    _, h = build_grid_hamiltonian(make_potential("harmonic"), n_qubits=3)
    assert np.allclose(h, h.conj().T)
    terms = pauli_decompose(h, atol=1e-10)
    rebuilt = reconstruct_from_paulis(terms)
    assert np.linalg.norm(h - rebuilt) / np.linalg.norm(h) < 1e-10
