"""End-to-end starter: train a double-well surrogate and design a target splitting."""
from pathlib import Path

from quantum_pinn.config import Domain
from quantum_pinn.design.dataset import generate_dataset
from quantum_pinn.design.inverse import optimize_potential
from quantum_pinn.design.training import (
    SurrogateTrainingConfig, evaluate_heldout, save_surrogate, train_surrogate,
)


def main() -> None:
    domain = Domain(-6.0, 6.0)
    dataset = generate_dataset(
        "double_well", n_configurations=256, n_grid=241, n_states=2,
        train_fraction=0.8, seed=2026, domain=domain,
    )
    config = SurrogateTrainingConfig(
        epochs=200, batch_size=1024, steps_per_epoch=4, physics_every=4, width=96, hidden_layers=4,
        physics_weight=0.05, seed=2026, device="auto",
    )
    result = train_surrogate(dataset, config)
    print("Held-out report:", evaluate_heldout(result))
    checkpoint = save_surrogate(result, Path("outputs/double_well_surrogate.pt"), config)
    print("Saved:", checkpoint)

    design = optimize_potential(
        result.model,
        family="double_well",
        objective="tunnel_splitting",
        target=0.05,
        initial_parameters={"barrier": 0.65, "separation": 1.2},
        steps=300,
        domain=domain,
        n_grid=1001,
    )
    print("Optimized parameters:", design.parameters)
    print("Surrogate splitting:", design.surrogate_value)
    print("Independent FD splitting:", design.finite_difference_value)


if __name__ == "__main__":
    main()
