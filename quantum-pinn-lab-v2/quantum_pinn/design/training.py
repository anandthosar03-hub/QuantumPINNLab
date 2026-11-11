"""Train and evaluate parameter-conditioned surrogates against FD ground truth."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from quantum_pinn.design.conditional import ParameterConditionedSurrogate
from quantum_pinn.design.dataset import SurrogateDataset
from quantum_pinn.design.parameter_space import (
    PARAMETER_SPACES, denormalize_parameter_tensor, normalize_parameter_array, parameter_names, potential_torch,
    validate_parameters,
)
from quantum_pinn.physics.observables import calculate_observables


@dataclass(frozen=True)
class SurrogateTrainingConfig:
    epochs: int = 100
    batch_size: int = 1024
    steps_per_epoch: int = 4
    physics_every: int = 4
    learning_rate: float = 1.0e-3
    width: int = 96
    hidden_layers: int = 4
    physics_weight: float = 0.05
    normalization_weight: float = 5.0
    orthogonality_weight: float = 2.0
    constraint_batch_configs: int = 4
    energy_weight: float = 1.0
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 32 or self.steps_per_epoch < 1 or self.physics_every < 1:
            raise ValueError("epochs and steps_per_epoch must be >=1 and batch_size must be >=32")
        if self.learning_rate <= 0 or self.width < 8 or self.hidden_layers < 1:
            raise ValueError("Invalid learning-rate or network dimensions")
        if min(self.physics_weight, self.energy_weight, self.normalization_weight, self.orthogonality_weight) < 0:
            raise ValueError("Loss weights must be non-negative")
        if self.constraint_batch_configs < 1:
            raise ValueError("constraint_batch_configs must be >=1")


@dataclass
class SurrogateTrainingResult:
    model: ParameterConditionedSurrogate
    history: list[dict[str, float]]
    energy_scale: float
    device: str
    family: str
    dataset: SurrogateDataset


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(requested)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation")
    return result


def _make_flat_training_arrays(dataset: SurrogateDataset) -> tuple[np.ndarray, ...]:
    indices = dataset.train_indices
    c = len(indices)
    n_states = dataset.n_states
    n_grid = dataset.x.size
    x = np.broadcast_to(dataset.x[None, None, :], (c, n_states, n_grid)).reshape(-1, 1)
    params = np.broadcast_to(
        dataset.parameters[indices, None, None, :], (c, n_states, n_grid, dataset.parameters.shape[1])
    ).reshape(-1, dataset.parameters.shape[1])
    state = np.broadcast_to(np.arange(n_states)[None, :, None], (c, n_states, n_grid)).reshape(-1)
    psi = dataset.wavefunctions[indices].reshape(-1)
    energy = np.broadcast_to(dataset.energies[indices, :, None], (c, n_states, n_grid)).reshape(-1)
    return x, params, state, psi, energy


def train_surrogate(
    dataset: SurrogateDataset,
    config: SurrogateTrainingConfig = SurrogateTrainingConfig(),
    progress_callback: Any | None = None,
) -> SurrogateTrainingResult:
    """Fit a parameter-conditioned PINN using FD labels plus the strong-form residual.

    Held-out parameter combinations are never included in the training tensors.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _device(config.device)
    arrays = _make_flat_training_arrays(dataset)
    x_np, params_np, state_np, psi_np, energy_np = arrays
    p_norm_np = normalize_parameter_array(dataset.family, params_np)
    energy_scale = max(float(np.std(dataset.energies[dataset.train_indices])),
                       float(np.mean(np.abs(dataset.energies[dataset.train_indices]))), 0.25)

    x_all = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    p_all = torch.as_tensor(p_norm_np, dtype=torch.float32, device=device)
    state_all = torch.as_tensor(state_np, dtype=torch.long, device=device)
    psi_all = torch.as_tensor(psi_np, dtype=torch.float32, device=device)
    energy_all = torch.as_tensor(energy_np, dtype=torch.float32, device=device)

    model = ParameterConditionedSurrogate(
        dataset.domain.xmin, dataset.domain.xmax, dataset.parameters.shape[1],
        n_states=dataset.n_states, width=config.width, hidden_layers=config.hidden_layers, family=dataset.family,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.05)
    history: list[dict[str, float]] = []
    train_parameter_tensor = torch.as_tensor(
        normalize_parameter_array(dataset.family, dataset.parameters[dataset.train_indices]),
        dtype=torch.float32, device=device,
    )
    x_constraint = torch.as_tensor(dataset.x, dtype=torch.float32, device=device)
    bounds_tensor = torch.as_tensor(
        np.asarray([(b.low, b.high) for b in PARAMETER_SPACES[dataset.family]]),
        dtype=torch.float32, device=device,
    )

    global_step = 0
    for epoch in range(config.epochs):
        model.train()
        totals = {"total": 0.0, "wavefunction": 0.0, "energy": 0.0, "physics": 0.0, "normalization": 0.0, "orthogonality": 0.0}
        count = 0
        active_counts = {"physics": 0, "normalization": 0, "orthogonality": 0}
        # Random mini-batches with replacement: second spatial derivatives are expensive,
        # so the model sees a fresh stochastic subset instead of sweeping the full grid
        # on every epoch. Across epochs, samples are revisited many times.
        for _ in range(config.steps_per_epoch):
            global_step += 1
            idx = torch.randint(0, x_all.shape[0], (config.batch_size,), device=device)
            constraint_step = global_step % config.physics_every == 0
            compute_physics = constraint_step and config.physics_weight > 0
            compute_constraints = constraint_step and (
                config.normalization_weight > 0 or config.orthogonality_weight > 0
            )
            x = x_all[idx].detach().clone().requires_grad_(compute_physics)
            p_norm = p_all[idx]
            state = state_all[idx]
            target_psi = psi_all[idx]
            target_energy = energy_all[idx]

            pred_psi, pred_energy = model(x, p_norm, state)
            wavefunction_loss = torch.mean((pred_psi - target_psi) ** 2)
            energy_loss = torch.mean(((pred_energy - target_energy) / energy_scale) ** 2)
            normalization_loss = torch.zeros((), dtype=pred_psi.dtype, device=device)
            orthogonality_loss = torch.zeros((), dtype=pred_psi.dtype, device=device)
            physics_loss = torch.zeros((), dtype=pred_psi.dtype, device=device)
            if compute_physics:
                dpsi = torch.autograd.grad(pred_psi.sum(), x, create_graph=True)[0]
                d2psi = torch.autograd.grad(dpsi.sum(), x, create_graph=True)[0]
                actual_params = p_norm * (bounds_tensor[:, 1] - bounds_tensor[:, 0]) + bounds_tensor[:, 0]
                potential = potential_torch(dataset.family, x, actual_params)
                residual = -0.5 * d2psi.squeeze(-1) + potential.squeeze(-1) * pred_psi - pred_energy * pred_psi
                physics_loss = torch.mean((residual / energy_scale) ** 2)

            if compute_constraints:
                # Explicitly prevent the trivial psi≈0 solution and retain pairwise
                # orthogonality. These constraints are evaluated on complete x-grids.
                n_constraint = min(config.constraint_batch_configs, train_parameter_tensor.shape[0])
                selected = torch.randint(0, train_parameter_tensor.shape[0], (n_constraint,), device=device)
                selected_params = train_parameter_tensor[selected]
                n_grid = x_constraint.numel()
                n_states = dataset.n_states
                x_group = x_constraint.reshape(1, 1, n_grid, 1).expand(n_constraint, n_states, n_grid, 1).reshape(-1, 1)
                p_group = selected_params[:, None, None, :].expand(
                    n_constraint, n_states, n_grid, selected_params.shape[1]
                ).reshape(-1, selected_params.shape[1])
                s_group = torch.arange(n_states, device=device).reshape(1, n_states, 1).expand(
                    n_constraint, n_states, n_grid
                ).reshape(-1)
                psi_group, _ = model(x_group, p_group, s_group)
                psi_group = psi_group.reshape(n_constraint, n_states, n_grid)
                group_norms = torch.trapezoid(psi_group**2, x_constraint, dim=-1)
                normalization_loss = torch.mean((group_norms - 1.0) ** 2)
                if n_states > 1:
                    products = psi_group.unsqueeze(2) * psi_group.unsqueeze(1)
                    overlaps = torch.trapezoid(products, x_constraint, dim=-1)
                    off_diagonal = overlaps * (1.0 - torch.eye(n_states, device=device).unsqueeze(0))
                    orthogonality_loss = torch.mean(off_diagonal**2)
            loss = (
                wavefunction_loss
                + config.energy_weight * energy_loss
                + config.physics_weight * physics_loss
                + config.normalization_weight * normalization_loss
                + config.orthogonality_weight * orthogonality_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_count = int(idx.numel())
            totals["total"] += float(loss.detach().cpu()) * batch_count
            totals["wavefunction"] += float(wavefunction_loss.detach().cpu()) * batch_count
            totals["energy"] += float(energy_loss.detach().cpu()) * batch_count
            if compute_physics:
                totals["physics"] += float(physics_loss.detach().cpu())
                active_counts["physics"] += 1
            if compute_constraints:
                totals["normalization"] += float(normalization_loss.detach().cpu())
                totals["orthogonality"] += float(orthogonality_loss.detach().cpu())
                active_counts["normalization"] += 1
                active_counts["orthogonality"] += 1
            count += batch_count

        scheduler.step()
        record = {key: value / max(count, 1) for key, value in totals.items() if key in ("total", "wavefunction", "energy")}
        record["physics"] = totals["physics"] / max(active_counts["physics"], 1)
        record["normalization"] = totals["normalization"] / max(active_counts["normalization"], 1)
        record["orthogonality"] = totals["orthogonality"] / max(active_counts["orthogonality"], 1)
        record["epoch"] = float(epoch + 1)
        history.append(record)
        if progress_callback is not None:
            progress_callback(epoch + 1, config.epochs, record["total"])

    model.eval()
    return SurrogateTrainingResult(model, history, energy_scale, str(device), dataset.family, dataset)


def predict_state(
    model: ParameterConditionedSurrogate,
    family: str,
    params: dict[str, float],
    state_index: int,
    x: np.ndarray,
    device: str | None = None,
) -> tuple[np.ndarray, float]:
    device_obj = torch.device(device or next(model.parameters()).device)
    params = validate_parameters(family, params, enforce_bounds=True)
    parameter_array = np.asarray([[params[name] for name in parameter_names(family)]], dtype=np.float32)
    p_norm = normalize_parameter_array(family, parameter_array)
    x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device_obj).reshape(-1, 1)
    p_tensor = torch.as_tensor(p_norm, dtype=torch.float32, device=device_obj)
    state_tensor = torch.full((x_tensor.shape[0],), int(state_index), dtype=torch.long, device=device_obj)
    model.eval()
    with torch.no_grad():
        psi, energies = model(x_tensor, p_tensor, state_tensor)
    psi_np = psi.cpu().numpy()
    norm = float(np.trapezoid(np.abs(psi_np) ** 2, x))
    if norm > 1.0e-14:
        psi_np = psi_np / np.sqrt(norm)
    return psi_np, float(torch.mean(energies).cpu())


def evaluate_heldout(result: SurrogateTrainingResult) -> dict[str, float | int]:
    """Evaluate energy, wavefunction fidelity and width only on unseen parameter sets."""
    dataset = result.dataset
    indices = dataset.heldout_indices
    if indices.size == 0:
        raise ValueError("No held-out parameter combinations are available")
    errors: list[float] = []
    ground_fidelities: list[float] = []
    width_errors: list[float] = []
    for config_index in indices:
        params = dict(zip(parameter_names(dataset.family), dataset.parameters[config_index].tolist(), strict=True))
        for state_idx in range(dataset.n_states):
            psi_pred, energy_pred = predict_state(
                result.model, dataset.family, params, state_idx, dataset.x, result.device
            )
            errors.append(energy_pred - float(dataset.energies[config_index, state_idx]))
            if state_idx == 0:
                psi_ref = dataset.wavefunctions[config_index, state_idx]
                overlap = float(np.trapezoid(psi_pred * psi_ref, dataset.x))
                ground_fidelities.append(overlap**2)
                ref_obs = calculate_observables(dataset.x, psi_ref, np.zeros_like(dataset.x))
                pred_obs = calculate_observables(dataset.x, psi_pred, np.zeros_like(dataset.x))
                width_errors.append(abs(pred_obs["sigma_x"] - ref_obs["sigma_x"]))
    error_array = np.asarray(errors)
    return {
        "heldout_configurations": int(indices.size),
        "all_state_energy_mae": float(np.mean(np.abs(error_array))),
        "all_state_energy_rmse": float(np.sqrt(np.mean(error_array**2))),
        "ground_state_energy_mae": float(np.mean(np.abs(error_array[::dataset.n_states]))),
        "mean_ground_state_fidelity": float(np.mean(ground_fidelities)),
        "ground_state_sigma_x_mae": float(np.mean(width_errors)),
    }


def save_surrogate(result: SurrogateTrainingResult, path: str | Path, config: SurrogateTrainingConfig) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "family": result.family,
        "xmin": result.dataset.domain.xmin,
        "xmax": result.dataset.domain.xmax,
        "n_parameters": result.dataset.parameters.shape[1],
        "n_states": result.dataset.n_states,
        "width": config.width,
        "hidden_layers": config.hidden_layers,
        "energy_scale": result.energy_scale,
        "training_config": asdict(config),
        "parameter_names": list(parameter_names(result.family)),
    }
    torch.save({"metadata": metadata, "state_dict": result.model.state_dict()}, destination)
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def load_surrogate(path: str | Path, device: str = "cpu") -> tuple[ParameterConditionedSurrogate, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    model = ParameterConditionedSurrogate(
        metadata["xmin"], metadata["xmax"], metadata["n_parameters"],
        n_states=metadata["n_states"], width=metadata["width"], hidden_layers=metadata["hidden_layers"],
        family=metadata["family"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, metadata
