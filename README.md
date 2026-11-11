# QuantumPINN Lab v0.2.0 — Parameter-Conditioned Quantum Design

An educational/research software project that combines physics-informed neural networks (PINNs), numerical quantum mechanics, time-dependent wavepacket propagation and quantum-computing Hamiltonian mapping. **v0.2.0 preserves the v0.1.0 solver and adds parameter-conditioned surrogate learning plus inverse design.**

## Two project levels / release history

| Release | Level | Main work | What it teaches |
|---|---|---|---|
| **v0.1.0 — Solver foundation** | Intermediate | Single-potential PINN eigenstates, 8 potentials, FD reference, observables, Crank–Nicolson dynamics, Pauli mapping | Numerical quantum mechanics, Autograd derivatives, model validation, scientific Python |
| **v0.2.0 — Conditional design** | Intermediate → advanced | All v0.1 features + parameter-conditioned network, FD training dataset, held-out parameter validation, gradient-based inverse design | Scientific ML, parameterized Hamiltonians, surrogate modelling, constrained optimization, reproducible experiments |

The original v0.1.0 ZIP is preserved at `releases/QuantumPINN_Lab_v0.1.0.zip`. The v0.2.0 source tree contains both the v1 features and the new v2 design modules.

## What v2 adds

For a potential family parameterized by `p`, the conditional network learns

`(x, p, n) -> (psi_n(x; p), E_n(p))`.

The network receives the spatial coordinate, normalized potential parameters and state index. It has two output branches: a coordinate-dependent wavefunction head and a coordinate-independent energy head. A hard Dirichlet envelope sets the wavefunction to zero at the domain edges.

Training labels come from the finite-difference (FD) eigensolver. The loss combines FD-supervised wavefunction and energy errors with the strong-form Schrödinger residual, explicit normalization and pairwise orthogonality penalties. Parameter combinations—not coordinate points—are split into training and held-out sets. This is important: coordinates from a single potential must not leak between train and test.

### Parameter families and optimization objectives

| Family | Parameters | Supported training interval |
|---|---|---|
| Harmonic | `omega` | 0.50–2.00 |
| Quartic | `omega`, `strength` | 0.50–2.00; 0.005–0.080 |
| Double well | `barrier`, `separation` | 0.15–1.50; 0.70–1.80 |

Inverse design objectives:

- `ground_energy`: match the lowest eigenenergy `E0`.
- `localization_width`: match the ground-state standard deviation `sigma_x`.
- `tunnel_splitting`: match `E1 - E0` for a double-well system.

Optimization is performed through the differentiable surrogate with sigmoid-constrained parameters that stay inside the data/training box. Every returned candidate is independently evaluated with the FD reference solver. The FD verification—not the surrogate score—is the final acceptance check.

## System architecture

```text
                         QuantumPINN Lab v0.2.0
                                     |
           +-------------------------+-------------------------+
           |                         |                         |
     Streamlit UI                 CLI tools                Python API
           |                         |                         |
           +-------------------------+-------------------------+
                                     |
                         Solver / Experiment layer
                                     |
          +--------------------------+--------------------------+
          |                          |                          |
   v1 Stationary PINN          v1 Time dynamics          v1 Quantum bridge
   MLP + Autograd              Crank–Nicolson            Grid Hamiltonian
          |                          |                     Pauli strings
          |                          |                          |
          +--------------------------+--------------------------+
                                     |
                      v2 Parameter-conditioned design
                                     |
        Latin-hypercube sampler -> FD dataset generator
                                     |
                     Parameter-conditioned neural model
                      /                             \
            Wavefunction head                  Energy head
                      \                             /
                       FD supervision + PDE residual
                       + norm + orthogonality losses
                                     |
                         Parameter-level held-out set
                                     |
                       Held-out metrics / diagnostics
                                     |
                     Differentiable inverse optimizer
             E0 target | sigma_x target | E1-E0 target
                                     |
                       Independent FD verification
                                     |
                   JSON / NPZ / Streamlit visualization
```

## Repository layout

```text
quantum-pinn-lab-v2/
├── quantum_pinn/
│   ├── models/              # v1 single-potential PINN
│   ├── physics/             # potentials, FD solver, observables, dynamics
│   ├── solvers/             # v1 stationary PINN orchestration
│   ├── quantum/             # grid Hamiltonian and Pauli mapping
│   ├── design/              # v2 conditional model, datasets, training, inverse design, CLI
│   └── ui/                  # Streamlit application, including v2 design tab
├── tests/                   # v1 physics tests and v2 design tests
├── examples/                # v1 quick start and v2 inverse-design example
├── outputs/                 # generated checkpoints, datasets and reports
├── releases/                # unchanged v0.1.0 archive
├── ARCHITECTURE.md
├── PROJECT_PLAN.md
├── INTERMEDIATE_PROJECT_GUIDE.md
├── CHANGELOG.md
└── pyproject.toml
```

## Installation

Recommended baseline: **64-bit Python 3.12 or 3.13**. The declared Python range is 3.12–3.15; the current package baselines publish Windows wheels for Python 3.15 as well, so an existing 3.15 installation can be used. The dependency constraints target current stable releases checked on 11 November 2026. The CUDA build of PyTorch must be selected to match the local NVIDIA driver; install the CPU default first or use the official PyTorch selector for a compatible CUDA wheel.

### Windows PowerShell

```powershell
py -3.15 -m venv .venv  # or py -3.12
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### macOS / Linux

```bash
python3.12 -m venv .venv  # Python 3.13–3.15 also supported by the current dependency wheels
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional Qiskit adapter:

```bash
pip install -e ".[quantum]"
```

## Run v1 features

Start the dashboard:

```bash
streamlit run quantum_pinn/ui/app.py
```

Run a single-potential stationary PINN from the command line:

```bash
python -m quantum_pinn.cli --potential harmonic --states 3 --epochs 500 --pretrain-steps 300
```

The same dashboard still includes the v1 time-dynamics and quantum Hamiltonian tabs.

## Run v2: train, evaluate, optimize

### 1. Train a conditional double-well surrogate

```bash
python -m quantum_pinn.design.cli train --family double_well --configurations 256 --grid 241 --states 2 --epochs 200 --batch-size 1024 --steps-per-epoch 4 --physics-every 4 --out outputs/double_well_surrogate.pt
```

Training generates the FD dataset, records a whole-configuration 80/20 split, trains the conditional model, evaluates the held-out parameter configurations and saves:

- `outputs/double_well_surrogate.pt` — model checkpoint and architecture metadata.
- `outputs/double_well_surrogate.json` — readable checkpoint metadata.
- `outputs/double_well_surrogate_fd_dataset.npz` — parameters, FD energies/eigenfunctions and split indices.
- `outputs/double_well_surrogate.metrics.json` — held-out evaluation report.

### 2. Re-evaluate held-out parameter combinations

```bash
python -m quantum_pinn.design.cli evaluate --checkpoint outputs/double_well_surrogate.pt --dataset outputs/double_well_surrogate_fd_dataset.npz
```

The evaluation reports energy MAE/RMSE, ground-state fidelity and localization-width MAE. Use the same dataset file created during training so the held-out set remains untouched.

### 3. Optimize a target tunnel splitting

```bash
python -m quantum_pinn.design.cli optimize --checkpoint outputs/double_well_surrogate.pt --objective tunnel_splitting --target 0.05 --steps 300 --grid 1001
```

Other objectives:

```bash
python -m quantum_pinn.design.cli optimize --checkpoint outputs/harmonic_surrogate.pt --objective ground_energy --target 0.75
python -m quantum_pinn.design.cli optimize --checkpoint outputs/harmonic_surrogate.pt --objective localization_width --target 0.70
```

The CLI first optimizes the differentiable surrogate, then (by default) optionally polishes the candidate with bounded Powell optimization against the FD solver. The report contains the neural proposal, its direct FD value, the polished parameters and the final FD result. Use `--no-fd-refine` to benchmark the pure surrogate optimizer without direct-FD polishing.

### 4. Python API

```python
from quantum_pinn.config import Domain
from quantum_pinn.design.dataset import generate_dataset
from quantum_pinn.design.training import (
    SurrogateTrainingConfig, train_surrogate, evaluate_heldout, save_surrogate,
)
from quantum_pinn.design.inverse import optimize_potential

# Generate labelled eigenpairs. The held-out split is by parameter combination.
dataset = generate_dataset(
    "double_well", n_configurations=256, n_grid=241, n_states=2,
    train_fraction=0.80, seed=42, domain=Domain(-6, 6),
)
config = SurrogateTrainingConfig(
    epochs=200, batch_size=1024, steps_per_epoch=4, physics_every=4,
    width=96, hidden_layers=4, device="auto",
)
trained = train_surrogate(dataset, config)
print(evaluate_heldout(trained))
save_surrogate(trained, "outputs/double_well_surrogate.pt", config)

candidate = optimize_potential(
    trained.model, "double_well", "tunnel_splitting", target=0.05,
    initial_parameters={"barrier": 0.65, "separation": 1.20},
    steps=300, domain=dataset.domain, n_grid=1001,
)
print("PINN proposal:", candidate.surrogate_parameters)
print("FD-polished parameters:", candidate.parameters)
print("Surrogate:", candidate.surrogate_value)
print("FD verification:", candidate.finite_difference_value)
```

A complete script is available at `examples/v2_inverse_design.py`.

## Tests

```bash
pytest
```

The test suite covers reference energies, normalization, orthogonality, time propagation, Pauli reconstruction, FD dataset splitting, conditional training/checkpoint reload and a bounded inverse-design smoke test.

## Benchmark snapshot (one reproducible development run)

The optimization figures in this table describe the **neural surrogate candidate before optional direct-FD polishing**. The final v2 workflow additionally runs a bounded Powell refinement against the FD solver and reports the polished FD value separately.

A local CPU smoke benchmark with the double-well family used 256 parameter combinations, a 241-point FD grid, 2 states, an 80/20 parameter-level split, a width-96 / 4-hidden-layer model, 200 epochs, 4 stochastic batches per epoch, and a physics/constraint update every fourth step. On that seed/split it produced:

| Metric | Observed value |
|---|---:|
| Held-out combinations | 51 |
| All-state energy MAE | 0.0874 |
| All-state energy RMSE | 0.1062 |
| Mean ground-state fidelity | 0.7847 |
| Ground-state `sigma_x` MAE | 0.3178 |
| Tunnel-splitting target | 0.0500 |
| Surrogate optimized splitting | 0.04999 |
| Independent FD splitting | 0.04643 |

This is a development snapshot, not a universal accuracy guarantee. In particular, the moderate held-out wavefunction fidelity demonstrates why the validation report and independent FD re-check are essential. Repeat with several seeds, more parameter combinations and refined grids before using results in a research paper.

## Scientific scope and limitations

- 1D dimensionless stationary Schrödinger equation, `hbar = m = 1`, symmetric finite domain and Dirichlet boundaries.
- FD data are numerical reference solutions; convergence with grid size and domain size remains necessary.
- Current conditional surrogate families are harmonic, quartic and double-well. The v1 single-potential solver still supports all eight potential types.
- The surrogate is not guaranteed to extrapolate outside the declared parameter box. The optimizer intentionally prevents such extrapolation.
- The current uncertainty estimate is not calibrated. Future work should include ensembles or Bayesian uncertainty and active learning.
- A 1D model is not yet a calibrated material/device solver. Semiconductor or molecular applications require validated material parameters, realistic effective masses, units, geometries and experimental comparison.
- Quantum hardware execution is not part of this release. The existing Pauli Hamiltonian bridge is a starting point for VQE simulation and resource analysis.

## Current dependency baselines

| Package | Baseline in metadata |
|---|---:|
| PyTorch | 2.15.0 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.2 |
| Streamlit | 1.67.0 |
| Qiskit (optional) | 2.6.2 |
| pytest (dev) | 9.2.0 |

Version numbers were checked against current PyPI release information and release notes on **11 November 2026**. PyTorch 2.15.0 has CPython 3.12 Windows wheels; NumPy 2.5.4 supports Python 3.12–3.15; SciPy 1.18.1 requires Python 3.12+; Streamlit 1.67.0 and Qiskit 2.6.2 have current release metadata. Use official install guidance for CUDA-specific PyTorch installation. See the links in `ARCHITECTURE.md`.

## License and intended use

MIT. This repository is for education, reproducible numerical experiments and research prototyping. It is not a validated engineering package or a substitute for independent scientific review.
