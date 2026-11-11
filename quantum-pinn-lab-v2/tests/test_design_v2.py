import numpy as np

from quantum_pinn.config import Domain
from quantum_pinn.design.dataset import generate_dataset
from quantum_pinn.design.inverse import optimize_potential
from quantum_pinn.design.parameter_space import PARAMETER_SPACES, parameter_names
from quantum_pinn.design.training import (
    SurrogateTrainingConfig, evaluate_heldout, load_surrogate, save_surrogate, train_surrogate,
)


def test_parameter_dataset_splits_by_configuration_and_has_fd_labels():
    dataset = generate_dataset("harmonic", n_configurations=8, n_grid=101, n_states=2, seed=8, domain=Domain(-5, 5))
    assert len(set(dataset.train_indices) & set(dataset.heldout_indices)) == 0
    assert dataset.parameters.shape == (8, 1)
    assert dataset.wavefunctions.shape == (8, 2, 101)
    assert np.allclose(np.trapezoid(dataset.wavefunctions[:, 0, :] ** 2, dataset.x, axis=1), 1.0, atol=1e-8)


def test_conditional_surrogate_trains_and_reports_heldout_metrics(tmp_path):
    dataset = generate_dataset("harmonic", n_configurations=8, n_grid=101, n_states=2, seed=13, domain=Domain(-5, 5))
    config = SurrogateTrainingConfig(epochs=2, batch_size=256, width=16, hidden_layers=2, device="cpu")
    result = train_surrogate(dataset, config)
    metrics = evaluate_heldout(result)
    assert metrics["heldout_configurations"] == len(dataset.heldout_indices)
    assert np.isfinite(metrics["all_state_energy_mae"])
    assert 0.0 <= metrics["mean_ground_state_fidelity"] <= 1.0

    checkpoint = save_surrogate(result, tmp_path / "conditional.pt", config)
    model, metadata = load_surrogate(checkpoint, device="cpu")
    assert model.n_states == 2
    assert metadata["family"] == "harmonic"


def test_inverse_design_stays_inside_training_bounds_and_fd_verifies():
    dataset = generate_dataset("harmonic", n_configurations=8, n_grid=101, n_states=2, seed=19, domain=Domain(-5, 5))
    result = train_surrogate(
        dataset,
        SurrogateTrainingConfig(epochs=1, batch_size=512, width=16, hidden_layers=2, device="cpu"),
    )
    designed = optimize_potential(
        result.model, "harmonic", "ground_energy", target=0.75,
        initial_parameters={"omega": 1.0}, steps=3, domain=dataset.domain, n_grid=101,
        fd_refine=True, fd_maxiter=30,
    )
    low, high = PARAMETER_SPACES["harmonic"][0].low, PARAMETER_SPACES["harmonic"][0].high
    assert low <= designed.parameters["omega"] <= high
    assert np.isfinite(designed.finite_difference_value)
    assert abs(designed.finite_difference_value - designed.reference_energies[0]) < 1e-10
    assert designed.fd_refined is True
    # E0=omega/2 for the dimensionless harmonic oscillator; FD polishing
    # should recover the target far more accurately than the tiny PINN smoke fit.
    assert designed.finite_difference_target_error < 2e-2


def test_parameter_names_are_stable_and_tunnel_split_requires_double_well():
    assert parameter_names("double_well") == ("barrier", "separation")
