"""Small API example; run from the repository root."""
from quantum_pinn.config import Domain, PINNConfig
from quantum_pinn.physics.potentials import make_potential
from quantum_pinn.solvers.stationary import solve_stationary

potential = make_potential("harmonic", omega=1.0)
config = PINNConfig(n_states=2, epochs=300, pretrain_steps=150, device="auto")
result = solve_stationary(potential, Domain(-6, 6), config)

for item in result.diagnostics:
    print(
        f"n={item.state_index}: PINN E={item.pinn_energy:.6f}, "
        f"FD E={item.reference_energy:.6f}, "
        f"relative error={item.relative_energy_error_percent:.3f}%"
    )
