# QuantumPINN v2 — parameter-conditioned inverse design

Version 2 keeps the v1 single-potential solver intact and adds a family-specific conditional surrogate. The network receives coordinate `x`, normalized potential parameters, and state index. It predicts a wavefunction value and a coordinate-independent eigenenergy.

## Core mathematical task

For a parameterized potential `V(x; p)`, learn

`(x, p, n) -> (psi_n(x; p), E_n(p))`

with finite-difference (FD) eigenpairs as supervised ground truth and the strong-form residual

`r = -0.5 * d²psi/dx² + V(x; p) * psi - E * psi`

as a physics-informed training term. Training and held-out splits are made by *parameter combination*, never by individual coordinate rows. That prevents leakage of the same potential into both sets.

## Supported conditioned families

| Family | Parameters and training interval |
|---|---|
| Harmonic | `omega`: 0.50–2.00 |
| Quartic | `omega`: 0.50–2.00; `strength`: 0.005–0.080 |
| Double well | `barrier`: 0.15–1.50; `separation`: 0.70–1.80 |

The optimizer is constrained to these intervals. The model is not intended to extrapolate beyond its training parameter box.

## Train

```bash
python -m quantum_pinn.design.cli train --family double_well --configurations 32 --grid 241 --states 2 --epochs 160 --steps-per-epoch 3 --physics-every 3 --out outputs/double_well_surrogate.pt
```

This creates:

- `outputs/double_well_surrogate.pt` — checkpoint (state dict + architecture metadata)
- `outputs/double_well_surrogate.json` — checkpoint metadata
- `outputs/double_well_surrogate_fd_dataset.npz` — FD ground-truth data, including the parameter-level split
- `outputs/double_well_surrogate.metrics.json` — held-out accuracy report

## Evaluate held-out configurations

```bash
python -m quantum_pinn.design.cli evaluate --checkpoint outputs/double_well_surrogate.pt --dataset outputs/double_well_surrogate_fd_dataset.npz
```

The report includes all-state energy MAE/RMSE, ground-state energy MAE, ground-state fidelity and localization-width MAE.

## Inverse design

Ground-state energy target:

```bash
python -m quantum_pinn.design.cli optimize --checkpoint outputs/harmonic_surrogate.pt --objective ground_energy --target 0.8
```

Ground-state localization width target:

```bash
python -m quantum_pinn.design.cli optimize --checkpoint outputs/harmonic_surrogate.pt --objective localization_width --target 0.9
```

Double-well tunnel splitting target:

```bash
python -m quantum_pinn.design.cli optimize --checkpoint outputs/double_well_surrogate.pt --objective tunnel_splitting --target 0.05
```

Optimization first runs through the differentiable surrogate. By default, a bounded Powell polishing stage then starts from the neural proposal and minimizes the same target error using the FD eigensolver directly. The report preserves both parameter vectors, the FD value at the neural proposal, and the final surrogate/FD values. Disable polishing with `--no-fd-refine` to study pure surrogate optimization. Use the final FD result—not the surrogate estimate—as the acceptance check.

## Important scientific limitations

- This is a 1D, dimensionless prototype with `hbar = m = 1` and Dirichlet boundaries.
- The FD solver is a numerical reference, not an exact analytic solution; perform grid/domain convergence studies for publishable claims.
- Surrogate uncertainty is not yet calibrated. A low training loss does not prove out-of-distribution reliability. Stochastic mini-batches are used to keep second-derivative training practical on a workstation; increase epochs and sample count for production-quality studies.
- Parameter bounds are deliberately finite. Do not optimize outside them or claim generalization beyond the sampled domain.
- Double-well tunneling splitting is defined as `E1 - E0` for the two lowest states; it becomes difficult when the splitting approaches the FD discretization / domain error.
- The PINN residual is evaluated at supervised sample points. A stronger research version should add independent residual collocation, adaptive sampling, and uncertainty-aware active learning.
