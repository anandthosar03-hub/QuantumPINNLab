"""Parameter-conditioned surrogate learning and inverse quantum design (v2)."""
from quantum_pinn.design.dataset import SurrogateDataset, generate_dataset
from quantum_pinn.design.inverse import InverseDesignResult, optimize_potential
from quantum_pinn.design.training import (
    SurrogateTrainingConfig, SurrogateTrainingResult, evaluate_heldout,
    load_surrogate, save_surrogate, train_surrogate,
)

__all__ = [
    "InverseDesignResult", "SurrogateDataset", "SurrogateTrainingConfig",
    "SurrogateTrainingResult", "evaluate_heldout", "generate_dataset",
    "load_surrogate", "optimize_potential", "save_surrogate", "train_surrogate",
]
