"""Sequential, hybrid physics-informed neural eigenstate solver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from quantum_pinn.config import Domain, PINNConfig
from quantum_pinn.models.pinn import WavefunctionPINN
from quantum_pinn.physics.potentials import Potential
from quantum_pinn.physics.reference import ReferenceResult, solve_reference


@dataclass
class StateDiagnostics:
    state_index: int
    pinn_energy: float
    reference_energy: float
    absolute_energy_error: float
    relative_energy_error_percent: float
    norm: float
    orthogonality_error: float
    fidelity: float
    node_count: int
    residual_rms: float
    training_loss: list[float]


@dataclass
class StationaryResult:
    x: np.ndarray
    potential: np.ndarray
    energies: np.ndarray
    reference_energies: np.ndarray
    wavefunctions: np.ndarray
    reference_wavefunctions: np.ndarray
    diagnostics: list[StateDiagnostics]
    device: str
    models: list[WavefunctionPINN]


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _as_tensor(values: np.ndarray | list[float], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)


def _fit_reference_shape(
    model: WavefunctionPINN,
    x_grid: torch.Tensor,
    target: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> None:
    """Supervised initialization only; final refinement is driven by the PDE residual."""
    if steps <= 0:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x_grid) - target) ** 2)
        loss.backward()
        optimizer.step()


def _pde_residual(
    model: WavefunctionPINN,
    x: torch.Tensor,
    energy: torch.Tensor,
    potential: Potential,
) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    psi = model(x)
    dpsi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    d2psi = torch.autograd.grad(dpsi.sum(), x, create_graph=True)[0]
    return -0.5 * d2psi + potential.torch(x) * psi - energy * psi


def solve_stationary(
    potential: Potential,
    domain: Domain = Domain(),
    config: PINNConfig = PINNConfig(),
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> StationaryResult:
    """Compute low-lying states and compare them with a finite-difference baseline.

    The reference solver is used for a stable optional warm start and for validation.
    During refinement, the loss is the differential-equation residual plus normalization,
    orthogonality, and a weak energy anchor. Set warm_start_from_fd=False for a more
    challenging reference-free experiment.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    ref = solve_reference(potential, domain, n_grid=config.n_eval, n_states=config.n_states)
    x_eval = _as_tensor(ref.x, device)
    previous_grid_states: list[torch.Tensor] = []
    models: list[WavefunctionPINN] = []
    found_energies: list[float] = []
    found_wavefunctions: list[np.ndarray] = []
    diagnostics: list[StateDiagnostics] = []

    for state_idx in range(config.n_states):
        model = WavefunctionPINN(
            xmin=domain.xmin,
            xmax=domain.xmax,
            width=config.width,
            hidden_layers=config.n_hidden_layers,
        ).to(device)
        target_np = ref.wavefunctions[:, state_idx]
        target = _as_tensor(target_np, device)
        if config.warm_start_from_fd:
            _fit_reference_shape(model, x_eval, target, config.pretrain_steps, config.learning_rate)
            initial_energy = float(ref.energies[state_idx])
            energy_anchor_weight = config.energy_anchor_weight
        else:
            # Reference-free initialization: estimate a scale from the potential minimum,
            # but do not use FD eigenvalues in the training objective.
            probe_x = np.linspace(domain.xmin, domain.xmax, config.n_eval)
            initial_energy = float(np.min(potential.numpy(probe_x))) + 0.5 + state_idx
            energy_anchor_weight = 0.0

        energy = nn.Parameter(torch.tensor(initial_energy, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam(
            list(model.parameters()) + [energy],
            lr=config.fine_tune_learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1), eta_min=config.fine_tune_learning_rate * 0.1)
        x_uniform = torch.linspace(domain.xmin, domain.xmax, config.n_collocation, device=device).reshape(-1, 1)
        train_losses: list[float] = []
        best_loss = float("inf")
        best_model_state: dict[str, torch.Tensor] | None = None
        best_energy = float(energy.detach().cpu())

        for epoch in range(config.epochs):
            # Mix a uniform grid with fresh random points to reduce collocation bias.
            random_x = domain.xmin + (domain.xmax - domain.xmin) * torch.rand(
                (config.n_collocation, 1), device=device
            )
            x_col = 0.5 * (x_uniform + random_x)
            residual = _pde_residual(model, x_col, energy, potential)
            residual_loss = torch.mean(residual**2)

            psi_eval = model(x_eval)
            norm = torch.trapezoid(psi_eval[:, 0] ** 2, x_eval[:, 0])
            norm_loss = (norm - 1.0) ** 2
            if previous_grid_states:
                overlap_losses = [torch.trapezoid(psi_eval[:, 0] * p[:, 0], x_eval[:, 0]) ** 2 for p in previous_grid_states]
                orth_loss = torch.stack(overlap_losses).sum()
            else:
                orth_loss = torch.zeros((), dtype=torch.float32, device=device)

            anchor = (energy - float(ref.energies[state_idx])) ** 2 if energy_anchor_weight > 0 else torch.zeros((), device=device)
            loss = (
                residual_loss
                + config.normalization_weight * norm_loss
                + config.orthogonality_weight * orth_loss
                + energy_anchor_weight * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [energy], max_norm=2.0)
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            if np.isfinite(loss_value) and loss_value < best_loss:
                best_loss = loss_value
                best_energy = float(energy.detach().cpu())
                best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if progress_callback and (epoch % max(1, config.epochs // 20) == 0 or epoch == config.epochs - 1):
                progress_callback(state_idx + 1, epoch + 1, loss_value)

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            with torch.no_grad():
                energy_value = best_energy
        else:
            energy_value = float(energy.detach().cpu())

        with torch.no_grad():
            psi_np = model(x_eval).detach().cpu().numpy().reshape(-1)
        norm_np = np.sqrt(np.trapezoid(psi_np**2, ref.x))
        if norm_np <= 1.0e-12 or not np.isfinite(norm_np):
            raise RuntimeError(f"State {state_idx} collapsed to a zero/invalid wavefunction")
        psi_np /= norm_np
        # Align sign only for comparison; quantum states are invariant under a global phase.
        if np.trapezoid(psi_np * target_np, ref.x) < 0:
            psi_np *= -1.0
        psi_grid_t = _as_tensor(psi_np, device).detach()
        previous_grid_states.append(psi_grid_t)
        models.append(model)
        found_energies.append(float(energy_value))
        found_wavefunctions.append(psi_np)

        with torch.enable_grad():
            x_diag = torch.linspace(domain.xmin, domain.xmax, max(401, config.n_eval), device=device).reshape(-1, 1)
            residual_diag = _pde_residual(model, x_diag, torch.tensor(energy_value, device=device), potential)
            residual_rms = float(torch.sqrt(torch.mean(residual_diag.detach() ** 2)).cpu())
        orth_error = 0.0
        for prior in found_wavefunctions[:-1]:
            orth_error = max(orth_error, abs(float(np.trapezoid(psi_np * prior, ref.x))))
        abs_error = abs(float(energy_value) - float(ref.energies[state_idx]))
        rel_error = 100.0 * abs_error / max(abs(float(ref.energies[state_idx])), 1e-12)
        overlap_with_reference = float(np.trapezoid(psi_np * target_np, ref.x))
        fidelity = min(1.0, max(0.0, overlap_with_reference**2))
        threshold = 1.0e-2 * float(np.max(np.abs(psi_np)))
        significant = psi_np[np.abs(psi_np) > threshold]
        node_count = int(np.count_nonzero(significant[1:] * significant[:-1] < 0)) if significant.size > 1 else 0
        diagnostics.append(
            StateDiagnostics(
                state_index=state_idx,
                pinn_energy=float(energy_value),
                reference_energy=float(ref.energies[state_idx]),
                absolute_energy_error=abs_error,
                relative_energy_error_percent=rel_error,
                norm=float(np.trapezoid(psi_np**2, ref.x)),
                orthogonality_error=orth_error,
                fidelity=fidelity,
                node_count=node_count,
                residual_rms=residual_rms,
                training_loss=train_losses,
            )
        )

    return StationaryResult(
        x=ref.x,
        potential=ref.potential,
        energies=np.asarray(found_energies),
        reference_energies=ref.energies,
        wavefunctions=np.column_stack(found_wavefunctions),
        reference_wavefunctions=ref.wavefunctions,
        diagnostics=diagnostics,
        device=str(device),
        models=models,
    )
