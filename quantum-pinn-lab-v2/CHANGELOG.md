# Changelog

## 0.2.0 — Parameter-conditioned inverse design

- Preserved the v0.1.0 single-potential PINN, wavepacket dynamics, observables and Pauli Hamiltonian mapping.
- Added a family-specific parameter-conditioned model: `(x, potential parameters, state index) -> (wavefunction, energy)`.
- Added Latin-hypercube parameter sampling and finite-difference reference datasets.
- Added parameter-level train/held-out splitting to prevent coordinate-level leakage.
- Added supervised FD labels plus a differentiable Schrödinger residual in the conditional model's loss.
- Added held-out energy, fidelity and localization-width metrics.
- Added differentiable inverse objectives for ground-state energy, localization width and double-well tunnel splitting.
- Added independent finite-difference verification after each inverse-design run.
- Added CLI commands and a Streamlit parameter-design tab.
- Preserved the v0.1.0 ZIP under `releases/`.
