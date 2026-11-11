# QuantumPINN Lab v0.2.0 — System Architecture

## 1. Purpose

The v1 product is a single-potential 1D quantum eigensolver demonstrator. The v2 product turns that into a reusable design workflow. A conditional surrogate is trained over a bounded family of potentials and then used as a differentiable map from design parameters to quantum observables.

## 2. Layer diagram

```text
Users / researchers
       |
       +---- Streamlit dashboard ----------------------------+
       |                                                     |
       +---- CLI (stationary / train / evaluate / optimize)  |
       |                                                     |
       +---- Python API -------------------------------------+
                                                             |
                                                   Application orchestration
                                                             |
        +---------------------------+------------------------+--------------------+
        |                           |                                             |
  Stationary solver            Time dynamics                             Quantum bridge
        |                           |                                             |
  WavefunctionPINN          Crank–Nicolson                           Finite grid H
  Autograd d²/dx²           Gaussian packets                          Pauli decomposition
  Energy variational head   Probability / current                     Optional Qiskit
        |                           |                                             |
        +---------------------------+------------------------+--------------------+
                                                             |
                                                 Parameter-conditioned design
                                                             |
  Parameter space -> Latin hypercube -> FD reference eigensolver -> labelled dataset
                                                             |
                      Whole-parameter train / held-out split (no coordinate leakage)
                                                             |
                           Conditional multi-state PINN surrogate
                                /                         \
                       ψ head: ψ(x;p,n)              E head: E(p,n)
                                \                         /
                            Data + physics-informed objective
                  L = L_ψ + λ_E L_E + λ_P L_PDE + λ_N L_norm + λ_O L_orth
                                                             |
                                               Held-out evaluation / metrics
                                                             |
                                        Constrained differentiable optimizer
                          ground E0 / localization σx / double-well E1-E0
                                                             |
                                    Independent finite-difference verification
                                                             |
                                               Report + saved checkpoint
```

## 3. Physics model

Dimensionless stationary Hamiltonian:

`H(p) psi_n(x;p) = E_n(p) psi_n(x;p)`

`H(p) = -1/2 d²/dx² + V(x;p)`

Parameter-conditioned network input: `(x, p_norm, n_norm)`.

Network output: `(psi_hat, E_hat)`. `E_hat` is produced by a parameter/state-only head, so it is not allowed to vary with coordinate `x`. The wavefunction head uses a hard Dirichlet envelope `(1 - xi²) * N_theta(xi,p,n)`, where `xi` maps the finite domain to `[-1,1]`.

The strong-form residual is

`r_theta(x;p,n) = -1/2 * d²psi_hat/dx² + V(x;p) * psi_hat - E_hat * psi_hat`.

Loss terms:

- `L_psi`: supervised FD wavefunction MSE.
- `L_E`: supervised FD eigenenergy MSE, scaled by the training energy scale.
- `L_PDE`: mean squared strong-form residual, scaled by the energy scale.
- `L_norm`: mean squared deviation of predicted grid-integrated state norms from one.
- `L_orth`: squared off-diagonal state overlap penalty.

To keep second derivatives tractable on a desktop workstation, stochastic batches are used. PDE and full-grid norm/orthogonality constraints are computed periodically (`physics_every`); the supervised FD loss is used on every update.

## 4. Dataset and leakage prevention

1. Sample parameter vectors using a seeded Latin hypercube within explicit bounds.
2. Compute every state with the same high-resolution finite-difference eigensolver.
3. Normalize each eigenfunction and fix its arbitrary sign for reproducible labels.
4. Split by parameter configuration (default 80% training / 20% held out).
5. Flatten only the *training configurations* into supervised coordinate rows.
6. Evaluate final model metrics only on held-out configurations.

The test partition is a set of unseen potential vectors. It is not a random selection of x-points from potentials that also appear in the training set.

## 5. Parameter spaces

| Family | Parameter vector | Lower / upper limits |
|---|---|---|
| Harmonic | `(omega)` | `(0.50, 2.00)` |
| Quartic | `(omega, strength)` | `(0.50, 2.00)`, `(0.005, 0.080)` |
| Double well | `(barrier, separation)` | `(0.15, 1.50)`, `(0.70, 1.80)` |

These are the supported model-training boxes, not universal physical ranges.

## 6. Inverse-design process

Parameters are represented by unconstrained optimization variables and mapped through a sigmoid into the training bounds. Autograd differentiates the observable with respect to normalized potential parameters through the conditional surrogate.

- Ground energy: `J = ((E0_hat - E_target)/s)^2`.
- Localization: `J = ((sigma_x_hat - sigma_target)/s)^2`, where `sigma_x` is computed from normalized `|psi_0|²`.
- Double-well tunnel splitting: `J = (((E1_hat - E0_hat) - Delta_target)/s)^2`.

After surrogate optimization, the neural candidate is evaluated by the independent SciPy FD eigensolver. By default, bounded Powell refinement then starts from that candidate and minimizes the same target error against the FD solver. The report retains both the original neural candidate and the FD-polished vector, along with surrogate and FD values. Set `fd_refine=False` to evaluate pure surrogate optimization without the direct-FD polishing stage.

## 7. Software modules

| Module | Responsibility |
|---|---|
| `quantum_pinn/physics/potentials.py` | v1 potential formulas |
| `quantum_pinn/physics/reference.py` | FD tridiagonal eigensolver |
| `quantum_pinn/models/pinn.py` | v1 single-potential PINN |
| `quantum_pinn/solvers/stationary.py` | v1 stationary training/refinement |
| `quantum_pinn/physics/dynamics.py` | time evolution / probability current |
| `quantum_pinn/quantum/pauli.py` | Hamiltonian-to-Pauli mapping / optional Qiskit adapter |
| `quantum_pinn/design/parameter_space.py` | v2 bounds, normalization, sampling and torch potentials |
| `quantum_pinn/design/dataset.py` | FD dataset generator and serialization |
| `quantum_pinn/design/conditional.py` | conditional wavefunction + energy network |
| `quantum_pinn/design/training.py` | mixed FD-supervised / physics-informed training and held-out evaluation |
| `quantum_pinn/design/inverse.py` | differentiable optimization and FD re-verification |
| `quantum_pinn/design/cli.py` | `train`, `evaluate`, `optimize` CLI commands |
| `quantum_pinn/ui/app.py` | v1 dashboards plus the v2 parameter-design tab |

## 8. Design choices

- **SciPy tridiagonal eigensolver:** efficient ground truth for the 1D second-order finite-difference Hamiltonian.
- **PyTorch Autograd:** differentiable coordinate derivatives and parameter-conditioned optimization.
- **Latin hypercube:** space-filling coverage for a small number of expensive FD configurations.
- **Separate energy head:** enforces the physical fact that an eigenvalue is independent of x.
- **Hard boundary envelope:** enforces Dirichlet boundary conditions structurally.
- **FD post-check:** catches surrogate error and prevents a predicted target match from being mistaken for a verified quantum design.
- **Parameter-level holdout:** evaluates generalization to previously unseen potentials.

## 9. Extension points

- Active learning: add new FD labels where surrogate ensemble uncertainty or residual is high.
- Residual-adaptive collocation and gradient balancing.
- Symmetry-aware parity branches for symmetric potentials.
- 2D/3D Hamiltonians for quantum dots and realistic nanostructures.
- Complex-valued wavefunctions and vector-potential coupling.
- Differentiable sparse eigensolvers for direct gradient comparisons.
- Bayesian ensembles for uncertainty calibration.
- VQE/Estimator experiments with measurement grouping and hardware-noise analysis.

## 10. Dependency provenance checked 11 November 2026

- PyTorch 2.15.0 (PyPI uploaded 28 October 2026): https://pypi.org/project/torch/ . The wheel listing includes CPython 3.12 and 3.15 Windows builds.
- NumPy 2.5.4 (10 October 2026): https://numpy.org/news/ and https://pypi.org/project/numpy/ .
- SciPy 1.18.1 (21 August 2026; Python >=3.12): https://pypi.org/project/scipy/ and https://scipy.org/news/ .
- Matplotlib 3.11.2 (11 September 2026): https://pypi.org/project/matplotlib/ .
- Streamlit 1.67.0 (5 November 2026): https://pypi.org/project/streamlit/ and https://docs.streamlit.io/develop/quick-reference/release-notes .
- Qiskit 2.6.2 (5 November 2026): https://pypi.org/project/qiskit/ .
- pytest 9.2.0 (27 October 2026): https://pypi.org/project/pytest/ .

Use the official PyTorch installer for CUDA-specific builds: https://pytorch.org/get-started/locally/ .
