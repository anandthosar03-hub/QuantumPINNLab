# Intermediate Project Guide: from v1 to v2

The project is organized so v1 is a complete intermediate-level portfolio project, and v2 is the natural advanced extension. Work through them in order rather than starting with the inverse optimizer.

## Level 1 — v0.1.0: Schrödinger PINN solver

Implement, understand and demonstrate:

- One-potential-at-a-time MLP wavefunction solver.
- Automatic second derivatives for the stationary Schrödinger residual.
- Finite-difference reference eigenpairs and error comparison.
- Normalization, orthogonality, node count and state fidelity.
- Eight 1D potential models.
- Gaussian wavepacket evolution with Crank–Nicolson.
- Expectation values / uncertainty and probability current.
- Grid Hamiltonian and Pauli-string mapping.

**Intermediate-level completion criteria:** run the test suite, compare at least three harmonic-oscillator eigenenergies with analytic values, demonstrate double-well splitting, explain the PDE residual/loss terms, and show a grid-convergence plot.

## Level 2 — v0.2.0: conditional quantum design

Build on the v1 numerical reference solver:

- Generate FD eigenpairs over a parameterized potential family.
- Feed both `x` and normalized potential parameters to the network; include state index for multi-state prediction.
- Predict both `psi_n(x;p)` and `E_n(p)`.
- Train only on whole training parameter configurations.
- Reserve unseen parameter vectors for held-out validation.
- Optimize an energy, localization-width or tunnel-splitting target.
- Verify the proposed parameter vector with a denser FD solve.

**Intermediate-to-advanced completion criteria:** show training/held-out splits, energy MAE/RMSE, wavefunction fidelity, width error, optimizer trace, target match and FD verification. Include failures and rejected designs in the report.

## Suggested presentation title

**QuantumPINN Lab: Physics-Informed Neural Surrogates for Quantum Eigenstates and Inverse Potential Design**

## Portfolio deliverables

1. Git repository with source, tests and tagged v0.1.0 / v0.2.0 releases.
2. Streamlit dashboard screenshots and one short demo video.
3. Reproducible training dataset and model checkpoint for one family.
4. Held-out validation report, including limitations.
5. One inverse-design case study: target double-well tunnel splitting and independent FD confirmation.
6. Technical report explaining Hamiltonian discretization, PINN residual, loss functions, optimization and error analysis.
