"""Differentiable inverse design with finite-difference verification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.optimize import minimize

from quantum_pinn.config import Domain
from quantum_pinn.design.conditional import ParameterConditionedSurrogate
from quantum_pinn.design.parameter_space import PARAMETER_SPACES, default_parameters, denormalize_parameter_tensor, parameter_names, validate_parameters
from quantum_pinn.design.training import predict_state
from quantum_pinn.physics.observables import calculate_observables
from quantum_pinn.physics.potentials import make_potential
from quantum_pinn.physics.reference import solve_reference

ObjectiveName = Literal["ground_energy", "localization_width", "tunnel_splitting"]


@dataclass
class InverseDesignResult:
    family: str
    objective: str
    target: float
    parameters: dict[str, float]
    surrogate_parameters: dict[str, float]
    surrogate_value: float
    finite_difference_value: float
    surrogate_target_error: float
    finite_difference_target_error: float
    surrogate_candidate_value: float
    finite_difference_candidate_value: float
    fd_refined: bool
    history: list[float]
    reference_energies: list[float]


def _surrogate_metric(
    model: ParameterConditionedSurrogate,
    family: str,
    normalized_parameters: torch.Tensor,
    objective: ObjectiveName,
    x: torch.Tensor,
) -> torch.Tensor:
    model_device = x.device
    n = model.n_states
    if objective == "tunnel_splitting" and (family != "double_well" or n < 2):
        raise ValueError("tunnel_splitting requires a double_well surrogate trained with at least two states")

    p_batch = normalized_parameters.reshape(1, -1).expand(x.shape[0], -1)
    if objective in ("ground_energy", "tunnel_splitting"):
        energy_values: list[torch.Tensor] = []
        for state in range(2 if objective == "tunnel_splitting" else 1):
            state_batch = torch.full((x.shape[0],), state, device=model_device, dtype=torch.long)
            _, energy_grid = model(x, p_batch, state_batch)
            energy_values.append(torch.mean(energy_grid))
        if objective == "ground_energy":
            return energy_values[0]
        return torch.abs(energy_values[1] - energy_values[0])

    state_batch = torch.zeros((x.shape[0],), device=model_device, dtype=torch.long)
    psi, _ = model(x, p_batch, state_batch)
    density = psi**2
    norm = torch.trapezoid(density, x[:, 0]).clamp_min(1.0e-12)
    probability = density / norm
    mean_x = torch.trapezoid(x[:, 0] * probability, x[:, 0])
    mean_x2 = torch.trapezoid(x[:, 0] ** 2 * probability, x[:, 0])
    return torch.sqrt(torch.clamp(mean_x2 - mean_x**2, min=1.0e-12))


def _fd_metric(family: str, params: dict[str, float], domain: Domain, n_grid: int, objective: ObjectiveName) -> tuple[float, np.ndarray]:
    n_states = 2 if objective == "tunnel_splitting" else 1
    reference = solve_reference(make_potential(family, **params), domain, n_grid=n_grid, n_states=n_states)
    if objective == "ground_energy":
        value = float(reference.energies[0])
    elif objective == "tunnel_splitting":
        value = float(reference.energies[1] - reference.energies[0])
    else:
        obs = calculate_observables(reference.x, reference.wavefunctions[:, 0], reference.potential)
        value = float(obs["sigma_x"])
    return value, reference.energies


def optimize_potential(
    model: ParameterConditionedSurrogate,
    family: str,
    objective: ObjectiveName,
    target: float,
    initial_parameters: dict[str, float] | None = None,
    steps: int = 300,
    learning_rate: float = 0.03,
    domain: Domain | None = None,
    n_grid: int = 801,
    fd_refine: bool = True,
    fd_maxiter: int = 60,
) -> InverseDesignResult:
    """Optimize only inside the surrogate's training bounds, then verify with FD."""
    if model.family is not None and model.family != family:
        raise ValueError(f"Checkpoint family {model.family!r} does not match requested family {family!r}")
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if fd_maxiter < 1:
        raise ValueError("fd_maxiter must be >=1")
    if objective not in ("ground_energy", "localization_width", "tunnel_splitting"):
        raise ValueError(f"Unknown objective: {objective}")
    if objective == "tunnel_splitting" and family != "double_well":
        raise ValueError("tunnel_splitting is currently defined only for the double_well family")
    if objective == "tunnel_splitting" and model.n_states < 2:
        raise ValueError("The surrogate checkpoint must contain at least two states")

    initial = initial_parameters or default_parameters(family)
    initial = validate_parameters(family, initial, enforce_bounds=True)
    bounds = torch.as_tensor(
        np.asarray([(b.low, b.high) for b in PARAMETER_SPACES[family]]),
        dtype=torch.float32, device=next(model.parameters()).device,
    )
    start = torch.tensor([initial[name] for name in parameter_names(family)], dtype=torch.float32, device=bounds.device)
    normalized_start = (start - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    clipped = normalized_start.clamp(1e-5, 1.0 - 1e-5)
    unconstrained = torch.nn.Parameter(torch.logit(clipped))
    optimizer = torch.optim.Adam([unconstrained], lr=learning_rate)
    domain = domain or Domain(model.xmin, model.xmax)
    x = torch.linspace(domain.xmin, domain.xmax, 601, device=bounds.device).reshape(-1, 1)
    objective_scale = max(abs(float(target)), 0.1)
    history: list[float] = []
    model.eval()

    for _ in range(steps):
        normalized = torch.sigmoid(unconstrained)
        predicted = _surrogate_metric(model, family, normalized, objective, x)
        loss = ((predicted - float(target)) / objective_scale) ** 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        optimized_normalized = torch.sigmoid(unconstrained)
        optimized_parameters_tensor = denormalize_parameter_tensor(family, optimized_normalized)
        surrogate_parameters = {
            name: float(value)
            for name, value in zip(parameter_names(family), optimized_parameters_tensor.cpu().tolist(), strict=True)
        }
        surrogate_candidate_value = float(_surrogate_metric(model, family, optimized_normalized, objective, x).cpu())

    # Check the raw neural proposal with FD before any optional direct-FD polishing.
    fd_candidate_value, _ = _fd_metric(family, surrogate_parameters, domain, n_grid, objective)
    selected_parameters = dict(surrogate_parameters)
    fd_refined = False

    if fd_refine:
        names = parameter_names(family)
        bounds_list = [(float(b.low), float(b.high)) for b in PARAMETER_SPACES[family]]
        x0 = np.asarray([surrogate_parameters[name] for name in names], dtype=np.float64)

        def fd_objective(parameter_vector: np.ndarray) -> float:
            trial = {name: float(value) for name, value in zip(names, parameter_vector, strict=True)}
            value, _ = _fd_metric(family, trial, domain, n_grid, objective)
            return ((value - float(target)) / objective_scale) ** 2

        polished = minimize(
            fd_objective, x0, method="Powell", bounds=bounds_list,
            options={"maxiter": int(fd_maxiter), "xtol": 1e-5, "ftol": 1e-8},
        )
        selected_parameters = {
            name: float(value) for name, value in zip(names, polished.x, strict=True)
        }
        fd_refined = True

    # Final FD calculation is the acceptance check, not the neural surrogate.
    fd_value, fd_energies = _fd_metric(family, selected_parameters, domain, n_grid, objective)
    final_parameter_tensor = torch.tensor(
        [selected_parameters[name] for name in parameter_names(family)],
        dtype=torch.float32, device=bounds.device,
    )
    final_normalized = (final_parameter_tensor - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    with torch.no_grad():
        surrogate_value = float(_surrogate_metric(model, family, final_normalized, objective, x).cpu())

    return InverseDesignResult(
        family=family, objective=objective, target=float(target), parameters=selected_parameters,
        surrogate_parameters=surrogate_parameters, surrogate_value=surrogate_value,
        finite_difference_value=fd_value,
        surrogate_target_error=abs(surrogate_value - target),
        finite_difference_target_error=abs(fd_value - target),
        surrogate_candidate_value=surrogate_candidate_value,
        finite_difference_candidate_value=fd_candidate_value, fd_refined=fd_refined,
        history=history, reference_energies=fd_energies.tolist(),
    )
