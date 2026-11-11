"""Configuration models for QuantumPINN Lab."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PINNConfig:
    """Training settings for one-dimensional stationary eigenstates."""

    n_states: int = 3
    n_hidden_layers: int = 4
    width: int = 64
    pretrain_steps: int = 300
    epochs: int = 500
    n_collocation: int = 128
    n_eval: int = 601
    learning_rate: float = 1.0e-3
    fine_tune_learning_rate: float = 2.0e-4
    normalization_weight: float = 20.0
    orthogonality_weight: float = 25.0
    energy_anchor_weight: float = 0.02
    warm_start_from_fd: bool = True
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError("n_states must be at least 1")
        if self.n_hidden_layers < 1 or self.width < 8:
            raise ValueError("The network needs >=1 hidden layer and width >=8")
        if self.pretrain_steps < 0 or self.epochs < 1:
            raise ValueError("pretrain_steps must be >=0 and epochs must be >=1")
        if self.n_collocation < 32 or self.n_eval < 101:
            raise ValueError("Use at least 32 collocation and 101 evaluation points")
        if self.learning_rate <= 0 or self.fine_tune_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")


@dataclass(frozen=True)
class Domain:
    xmin: float = -6.0
    xmax: float = 6.0

    def __post_init__(self) -> None:
        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin")
        if abs(self.xmin + self.xmax) > 1e-10:
            raise ValueError("This version uses a symmetric domain: xmin must equal -xmax")
