import numpy as np

from quantum_pinn.config import Domain, PINNConfig
from quantum_pinn.physics.potentials import make_potential
from quantum_pinn.solvers.stationary import solve_stationary


def test_pinn_ground_state_smoke():
    config = PINNConfig(
        n_states=1,
        n_hidden_layers=3,
        width=32,
        pretrain_steps=250,
        epochs=100,
        n_collocation=96,
        n_eval=301,
        device="cpu",
    )
    result = solve_stationary(make_potential("harmonic"), Domain(-6, 6), config)
    diagnostic = result.diagnostics[0]
    assert np.isfinite(diagnostic.pinn_energy)
    assert abs(diagnostic.norm - 1.0) < 1e-5
    assert diagnostic.fidelity > 0.95
    assert diagnostic.node_count == 0
