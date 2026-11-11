import numpy as np

from quantum_pinn.config import Domain
from quantum_pinn.physics.dynamics import propagate_crank_nicolson
from quantum_pinn.physics.observables import calculate_observables
from quantum_pinn.physics.potentials import make_potential
from quantum_pinn.physics.reference import solve_reference


def test_harmonic_reference_ground_energy():
    result = solve_reference(make_potential("harmonic", omega=1.0), Domain(-8, 8), n_grid=1201, n_states=3)
    exact = np.array([0.5, 1.5, 2.5])
    assert np.max(np.abs(result.energies - exact)) < 2e-3


def test_reference_states_are_normalized_and_orthogonal():
    result = solve_reference(make_potential("harmonic"), Domain(-6, 6), n_grid=601, n_states=3)
    overlap = result.wavefunctions.T @ (result.wavefunctions * (result.x[1] - result.x[0]))
    # Trapezoid correction is tiny at zero-valued endpoints; diagonal is normalized separately.
    for i in range(3):
        assert abs(np.trapezoid(result.wavefunctions[:, i] ** 2, result.x) - 1.0) < 1e-10
    assert np.max(np.abs(overlap - np.diag(np.diag(overlap)))) < 2e-2


def test_observables_ground_state_is_centered():
    ref = solve_reference(make_potential("harmonic"), Domain(-7, 7), n_grid=701, n_states=1)
    obs = calculate_observables(ref.x, ref.wavefunctions[:, 0], ref.potential)
    assert abs(obs["x_mean"]) < 1e-8
    assert abs(obs["norm"] - 1.0) < 1e-10
    assert obs["sigma_x"] > 0
    assert obs["total_energy"] > 0


def test_crank_nicolson_conserves_norm():
    result = propagate_crank_nicolson(
        make_potential("harmonic"), Domain(-6, 6), n_grid=201,
        dt=0.003, n_steps=100, save_every=10,
    )
    assert np.max(np.abs(result.norms - 1.0)) < 1e-8
