# QuantumPINN Lab v0.2.0 — Project and Research Plan

## Overall outcome

Build a reproducible 1D quantum-mechanics learning platform with two usable levels:

- **Level 1 / v0.1.0 (intermediate):** one-potential PINN, FD reference, observables, time propagation, and Pauli Hamiltonian representation.
- **Level 2 / v0.2.0 (intermediate-to-advanced):** parameter-conditioned multi-state surrogate, FD-labelled dataset, held-out generalization evaluation, and inverse design.

## Learning outcomes

By completing v2, the developer should be able to explain and implement:

1. Central finite differences and tridiagonal Hamiltonian eigenproblems.
2. Eigenstates, normalization, orthogonality, parity, localization and tunneling.
3. Automatic differentiation for second spatial derivatives.
4. Multi-input neural surrogates for parameterized differential equations.
5. Sampling of a multidimensional design space and leakage-free holdout testing.
6. Differentiable constrained optimization and independent numerical verification.
7. Pauli decomposition of a discretized Hamiltonian and optional Qiskit use.
8. Scientific software quality: tests, metadata, deterministic seeds and reproducible reports.

## Milestones

| Phase | Tasks | Deliverable / acceptance evidence |
|---|---|---|
| 0 — baseline | Run v0.1, inspect the FD solver, compare harmonic oscillator energy levels | `pytest`; v1 dashboard; benchmark notebook/report |
| 1 — parameter space | Define family bounds, normalization and Latin-hypercube sampler | Reproducible parameter set; bounds-validation tests |
| 2 — FD dataset | Generate eigenpairs for multiple potential vectors; persist split metadata | Dataset `.npz`; no train/held-out parameter overlap |
| 3 — conditional model | Implement coordinate + parameter + state input; two heads for ψ and E | Boundary test; forward-shape tests; state-range validation |
| 4 — hybrid training | FD labels + PDE residual + norm + orthogonality | Loss history; finite gradients; checkpoint round trip |
| 5 — held-out evaluation | Compute energy MAE/RMSE, fidelity and localization error | Evaluation JSON on unseen parameter vectors |
| 6 — inverse design | Optimize E0, sigma_x or double-well E1-E0 inside data bounds | Candidate parameters and independent FD confirmation |
| 7 — UI and CLI | Add dashboard tab, training/evaluation/optimization commands | End-to-end workflow from both interfaces |
| 8 — research validation | Multi-seed study, larger FD grids, domain convergence, active-learning candidates | Reproducible technical report with uncertainty and limitations |

## Recommended experimental protocol

### A. Baseline physics checks

- Harmonic oscillator: compare to `E_n = omega * (n + 1/2)` and verify parity / node counts.
- Increase FD grid density from 161 to 241, 401, 801 and compare eigenvalues.
- Increase domain half-width and confirm bound states stop changing materially.
- Verify norms and pairwise overlaps with trapezoidal integration.

### B. Surrogate data coverage

- Begin with 64 configurations to check the data pipeline.
- Use at least 256 parameter vectors for a first design study; repeat the split with different seeds.
- Keep 20% parameter vectors out of model fitting.
- Do not tune optimizer settings on the held-out set.
- Check that parameter values lie within the declared bounds and that every potential has the same domain/grid.

### C. Model diagnostics

- Save training, PDE, normalization and orthogonality losses separately.
- Evaluate on the whole held-out parameter set.
- Report ground and excited energies separately in any research report.
- Compare wavefunction fidelity with sign-invariant overlap squared.
- Compare the derived localization width and any target observable with FD.
- Use at least three seeds before claiming robust generalization.

### D. Inverse-design verification

1. Set a target observable in the training parameter box.
2. Optimize through the differentiable surrogate.
3. Recompute the final candidate with FD using a grid finer than training. Optionally start a bounded direct-FD polish from the neural candidate.
4. Report surrogate target error and FD target error separately.
5. If the FD target error is too large, reject the candidate or add nearby FD samples and retrain. Do not silently accept a surrogate-only match.

## Suggested default workstation configuration

- Python 3.12.
- CPU for the original v1 tests; optional NVIDIA GPU for v2 conditional training.
- 256 design configurations, 241 FD points, 2 states.
- Network width 96, 4 hidden layers.
- 200 epochs, batch size 1024, 4 stochastic batches per epoch.
- PDE plus full-grid norm/orthogonality constraints every 4th training step.

The settings are a starting benchmark, not a guarantee of a desired accuracy. Increase sample count/epochs or change architecture based on held-out diagnostics.

## Research questions for a paper / dissertation

1. How does the conditional PINN compare with direct interpolation of FD energy tables?
2. How does the sample budget affect error in `E0`, `E1`, fidelity and `sigma_x`?
3. Does PDE residual regularization improve generalization over FD-only supervised learning?
4. How accurately can a surrogate inverse design reproduce a desired double-well tunnel splitting after FD verification?
5. Can active learning identify parameter regions where the surrogate is unreliable?
6. How do grid resolution, domain size and parameter distribution affect the design outcome?

## Explicit scope limits

The current v2 model is one-dimensional and dimensionless (`hbar=m=1`) with hard Dirichlet boundaries. It is a computational-physics learning/research prototype, not a production semiconductor simulator or validated quantum-device design tool. The present benchmark still has noticeable held-out wavefunction error; independent FD verification and multi-seed evaluation are mandatory.
