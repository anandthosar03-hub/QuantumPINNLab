"""Streamlit front end for QuantumPINN Lab."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from quantum_pinn.config import Domain, PINNConfig
from quantum_pinn.physics.dynamics import propagate_crank_nicolson, probability_current
from quantum_pinn.physics.observables import calculate_observables
from quantum_pinn.physics.potentials import POTENTIAL_DEFAULTS, make_potential
from quantum_pinn.quantum.pauli import build_grid_hamiltonian, pauli_decompose, reconstruct_from_paulis, to_qiskit_sparse_pauli_op
from quantum_pinn.solvers.stationary import solve_stationary
from quantum_pinn.design.dataset import generate_dataset
from quantum_pinn.design.inverse import optimize_potential
from quantum_pinn.design.parameter_space import PARAMETER_SPACES, default_parameters
from quantum_pinn.design.training import (
    SurrogateTrainingConfig, evaluate_heldout, save_surrogate, train_surrogate,
)


st.set_page_config(page_title="QuantumPINN Lab", page_icon="⚛️", layout="wide")
st.title("⚛️ QuantumPINN Lab")
st.caption("Physics-informed eigenstates • wavepacket dynamics • qubit Hamiltonian mapping")

with st.sidebar:
    st.header("Model configuration")
    potential_name = st.selectbox("Potential", list(POTENTIAL_DEFAULTS), index=1)
    p = POTENTIAL_DEFAULTS[potential_name]
    params: dict[str, float] = {}
    if potential_name == "harmonic":
        params["omega"] = st.slider("ω", 0.2, 2.0, float(p["omega"]), 0.05)
    elif potential_name == "quartic":
        params["omega"] = st.slider("ω", 0.1, 2.0, float(p["omega"]), 0.05)
        params["strength"] = st.slider("Quartic strength λ", 0.0, 0.2, float(p["strength"]), 0.005)
    elif potential_name == "double_well":
        params["barrier"] = st.slider("Well scale a", 0.05, 2.0, float(p["barrier"]), 0.05)
        params["separation"] = st.slider("Well separation b", 0.3, 2.5, float(p["separation"]), 0.05)
    elif potential_name == "finite_well":
        params["depth"] = st.slider("Well depth V₀", 1.0, 20.0, float(p["depth"]), 0.5)
        params["width"] = st.slider("Well width", 0.5, 5.0, float(p["width"]), 0.1)
    elif potential_name == "gaussian_well":
        params["depth"] = st.slider("Well depth V₀", 1.0, 20.0, float(p["depth"]), 0.5)
        params["sigma"] = st.slider("Well width σ", 0.2, 3.0, float(p["sigma"]), 0.1)
    elif potential_name == "barrier":
        params["height"] = st.slider("Barrier height", 1.0, 20.0, float(p["height"]), 0.5)
        params["width"] = st.slider("Barrier width", 0.2, 3.0, float(p["width"]), 0.1)
    elif potential_name == "soft_coulomb":
        params["charge"] = st.slider("Effective charge", 0.1, 3.0, float(p["charge"]), 0.1)
        params["softening"] = st.slider("Softening ε", 0.1, 1.0, float(p["softening"]), 0.05)
    potential = make_potential(potential_name, **params)
    st.caption(potential.description())
    xmin = st.number_input("Domain half-width", min_value=3.0, max_value=12.0, value=6.0, step=0.5)
    domain = Domain(-float(xmin), float(xmin))

stationary_tab, dynamics_tab, quantum_tab, design_tab, guide_tab = st.tabs(
    ["Stationary PINN", "Time dynamics", "Quantum bridge", "Parameter design (v2)", "Project guide"]
)

with stationary_tab:
    st.subheader("Stationary eigenstates")
    st.latex(r"\left[-\frac{1}{2}\frac{d^2}{dx^2}+V(x)\right]\psi_n(x)=E_n\psi_n(x)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_states = st.selectbox("Number of states", [1, 2, 3, 4], index=2)
    with c2:
        epochs = st.select_slider("PINN refinement epochs", options=[100, 250, 500, 800, 1200], value=500)
    with c3:
        width = st.selectbox("Network width", [32, 64, 96], index=1)
    with c4:
        device = st.selectbox("Compute device", ["auto", "cpu", "cuda"], index=0)
    warm_start = st.checkbox("Warm-start from finite-difference eigenstates", value=True, help="Recommended for reliable excited-state convergence. Final refinement still minimizes the Schrödinger residual.")
    with st.expander("Advanced PINN settings"):
        collocation = st.slider("Collocation points", 64, 1024, 256, 64)
        pretrain_steps = st.slider("Warm-start fitting steps", 0, 1000, 250, 50, disabled=not warm_start)
        hidden_layers = st.slider("Hidden layers", 2, 6, 4)
    if st.button("Solve stationary states", type="primary", use_container_width=True):
        config = PINNConfig(
            n_states=int(n_states), width=int(width), n_hidden_layers=int(hidden_layers),
            pretrain_steps=int(pretrain_steps), epochs=int(epochs), n_collocation=int(collocation),
            n_eval=601, warm_start_from_fd=warm_start, device=device,
        )
        progress = st.progress(0.0, text="Preparing solver…")
        status = st.empty()
        def on_progress(state: int, epoch: int, loss: float) -> None:
            total = n_states * epochs
            done = (state - 1) * epochs + epoch
            progress.progress(min(done / total, 1.0), text=f"State {state}/{n_states} · epoch {epoch}/{epochs}")
            status.caption(f"Current combined loss: {loss:.3e}")
        try:
            with st.spinner("Training physics-informed wavefunctions…"):
                result = solve_stationary(potential, domain, config, progress_callback=on_progress)
            st.session_state["stationary_result"] = result
            st.session_state["stationary_potential"] = potential
            progress.progress(1.0, text="Completed")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("stationary_result")
    if result is not None:
        st.markdown("#### Energy comparison")
        table_rows = []
        for d in result.diagnostics:
            table_rows.append({
                "State": d.state_index,
                "PINN energy": d.pinn_energy,
                "FD reference": d.reference_energy,
                "Relative error (%)": d.relative_energy_error_percent,
                "Residual RMS": d.residual_rms,
                "Norm": d.norm,
                "Orthogonality error": d.orthogonality_error,
                "Reference fidelity": d.fidelity,
                "Node count": d.node_count,
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result.x, result.potential, color="black", linestyle="--", alpha=0.6, label="V(x)")
        for i in range(result.wavefunctions.shape[1]):
            offset = result.energies[i]
            ax.plot(result.x, result.wavefunctions[:, i] + offset, label=f"PINN ψ{i} + E{i}")
            ax.plot(result.x, result.reference_wavefunctions[:, i] + result.reference_energies[i], linestyle=":", alpha=0.7)
            ax.axhline(result.reference_energies[i], linewidth=0.5, alpha=0.3)
        ax.set_xlabel("x (dimensionless)")
        ax.set_ylabel("Potential / offset wavefunction")
        ax.set_title("Neural eigenstates (solid) vs finite-difference reference (dotted)")
        ax.legend(ncol=2)
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        selected = st.selectbox("Inspect state", list(range(result.wavefunctions.shape[1])))
        obs = calculate_observables(result.x, result.wavefunctions[:, selected], result.potential)
        cols = st.columns(4)
        cols[0].metric("⟨x⟩", f"{obs['x_mean']:.4f}")
        cols[1].metric("Δx", f"{obs['sigma_x']:.4f}")
        cols[2].metric("Δp", f"{obs['sigma_p']:.4f}")
        cols[3].metric("Energy ⟨H⟩", f"{obs['total_energy']:.5f}")
        st.download_button(
            "Download result arrays (NPZ)",
            data=_npz_bytes(result),
            file_name="quantumpinn_stationary_results.npz",
            mime="application/octet-stream",
        )

with dynamics_tab:
    st.subheader("Time-dependent Schrödinger equation")
    st.latex(r"i\frac{\partial\psi}{\partial t}=\left[-\frac{1}{2}\frac{\partial^2}{\partial x^2}+V(x)\right]\psi")
    d1, d2, d3 = st.columns(3)
    with d1:
        x0 = st.slider("Initial packet center", -4.0, 0.0, -2.5, 0.1)
        packet_sigma = st.slider("Packet width", 0.2, 1.0, 0.45, 0.05)
    with d2:
        k0 = st.slider("Initial wave number k₀", 0.5, 8.0, 3.0, 0.1)
        dt = st.select_slider("Time step Δt", [0.001, 0.002, 0.004, 0.006], value=0.002)
    with d3:
        n_steps = st.select_slider("Time steps", [250, 500, 1000, 1500, 2500], value=1000)
        save_every = st.select_slider("Save every", [10, 20, 25, 50], value=25)
    if st.button("Run wavepacket dynamics", use_container_width=True):
        try:
            dyn = propagate_crank_nicolson(
                potential, domain, n_grid=501, dt=float(dt), n_steps=int(n_steps),
                save_every=int(save_every), x0=float(x0), sigma=float(packet_sigma), k0=float(k0),
            )
            st.session_state["dynamics_result"] = dyn
        except Exception as exc:
            st.exception(exc)
    dyn = st.session_state.get("dynamics_result")
    if dyn is not None:
        st.caption(f"Crank–Nicolson norm drift: {np.max(np.abs(dyn.norms - dyn.norms[0])):.3e}")
        frame = st.slider("Saved time index", 0, len(dyn.times) - 1, len(dyn.times) - 1)
        density = np.abs(dyn.wavefunctions[frame]) ** 2
        current = probability_current(dyn.x, dyn.wavefunctions[frame])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dyn.x, density, label="|ψ(x,t)|²")
        ax.plot(dyn.x, potential.numpy(dyn.x) / max(np.max(np.abs(potential.numpy(dyn.x))), 1.0) * np.max(density), linestyle="--", label="scaled V(x)")
        ax.set_title(f"Wavepacket at t = {dyn.times[frame]:.3f}")
        ax.set_xlabel("x")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)
        plt.close(fig)
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.plot(dyn.x, current)
        ax2.set_title("Probability current J(x,t)")
        ax2.set_xlabel("x")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)
        plt.close(fig2)
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(dyn.times, dyn.mean_positions)
        ax3.set_title("Wavepacket center ⟨x⟩ versus time")
        ax3.set_xlabel("time")
        ax3.set_ylabel("⟨x⟩")
        ax3.grid(alpha=0.2)
        st.pyplot(fig3)
        plt.close(fig3)

with quantum_tab:
    st.subheader("Discretized Hamiltonian → qubit Pauli operator")
    st.write("A grid with 2ⁿ interior points is represented by n qubits. Its matrix Hamiltonian is expanded in the Pauli basis.")
    q1, q2 = st.columns(2)
    with q1:
        n_qubits = st.select_slider("Number of qubits", [2, 3, 4, 5, 6], value=4)
    with q2:
        pauli_tol = st.select_slider("Coefficient cutoff", [1e-12, 1e-10, 1e-8, 1e-6], value=1e-10, format_func=lambda x: f"{x:.0e}")
    if st.button("Decompose Hamiltonian", use_container_width=True):
        try:
            xq, hq = build_grid_hamiltonian(potential, n_qubits=int(n_qubits), xmin=domain.xmin, xmax=domain.xmax)
            terms = pauli_decompose(hq, atol=float(pauli_tol))
            reconstructed = reconstruct_from_paulis(terms)
            error = float(np.linalg.norm(hq - reconstructed) / max(np.linalg.norm(hq), 1e-15))
            st.session_state["pauli_result"] = (terms, error, n_qubits, hq)
        except Exception as exc:
            st.exception(exc)
    pauli_result = st.session_state.get("pauli_result")
    if pauli_result is not None:
        terms, error, qcount, _ = pauli_result
        st.metric("Non-negligible Pauli terms", len(terms))
        st.metric("Relative reconstruction error", f"{error:.3e}")
        st.code("\n".join(f"{term.coefficient.real:+.8e}  {term.label}" for term in terms[:250]), language="text")
        if len(terms) > 250:
            st.caption("Showing the first 250 terms. The complete expansion is available in JSON export.")
        export = json.dumps([{"label": t.label, "real": t.coefficient.real, "imag": t.coefficient.imag} for t in terms], indent=2)
        st.download_button("Download Pauli expansion (JSON)", export, "hamiltonian_pauli_terms.json", "application/json")
        if st.button("Create Qiskit SparsePauliOp"):
            try:
                op = to_qiskit_sparse_pauli_op(terms)
                st.success(f"Created SparsePauliOp with {len(op)} terms and {op.num_qubits} qubits.")
                st.code(str(op)[:5000])
            except ImportError as exc:
                st.info(str(exc))


with design_tab:
    st.subheader("Parameter-conditioned surrogate and inverse design")
    st.markdown(
        "Train a reusable model for **ψₙ(x; p)** and **Eₙ(p)** across a parameter family. "
        "Finite-difference (FD) eigenpairs provide the supervised ground truth; held-out parameter "
        "combinations are reserved for evaluation. The optimizer is constrained to the training box."
    )
    family = st.selectbox("Potential family", sorted(PARAMETER_SPACES), index=2, key="design_family")
    d1, d2, d3 = st.columns(3)
    with d1:
        n_configurations = st.number_input("FD parameter combinations", min_value=8, max_value=512, value=256, step=16)
    with d2:
        design_grid = st.selectbox("FD grid points", [161, 241, 301, 401], index=1)
    with d3:
        design_epochs = st.selectbox("Training epochs", [50, 100, 200, 300, 400], index=2)
    c1, c2, c3 = st.columns(3)
    with c1:
        design_width = st.selectbox("Network width", [32, 64, 96, 128], index=2)
    with c2:
        design_batch = st.selectbox("Batch size", [512, 1024, 2048, 4096], index=1)
    with c3:
        design_device = st.selectbox("Surrogate device", ["auto", "cpu", "cuda"], key="design_device")

    if st.button("Generate FD data and train conditional surrogate", type="primary", use_container_width=True):
        try:
            training_progress = st.progress(0.0, text="Generating finite-difference data…")
            design_dataset = generate_dataset(
                family, n_configurations=int(n_configurations), n_grid=int(design_grid),
                n_states=2, seed=42, domain=domain,
            )
            def design_callback(epoch: int, total: int, loss: float) -> None:
                training_progress.progress(epoch / total, text=f"Training epoch {epoch}/{total} · loss {loss:.3e}")
            design_config = SurrogateTrainingConfig(
                epochs=int(design_epochs), batch_size=int(design_batch), steps_per_epoch=4, physics_every=4, width=int(design_width),
                hidden_layers=4, device=design_device,
            )
            design_result = train_surrogate(design_dataset, design_config, progress_callback=design_callback)
            checkpoint_path = Path("outputs") / f"{family}_conditional_surrogate.pt"
            save_surrogate(design_result, checkpoint_path, design_config)
            heldout_metrics = evaluate_heldout(design_result)
            st.session_state["conditional_design_result"] = design_result
            st.session_state["conditional_design_family"] = family
            st.session_state["conditional_design_metrics"] = heldout_metrics
            training_progress.progress(1.0, text="Training and held-out evaluation completed")
            st.success(f"Surrogate saved to {checkpoint_path}")
        except Exception as exc:
            st.exception(exc)

    trained = st.session_state.get("conditional_design_result")
    trained_family = st.session_state.get("conditional_design_family")
    if trained is not None and trained_family == family:
        st.markdown("#### Held-out generalization metrics")
        st.dataframe([st.session_state["conditional_design_metrics"]], use_container_width=True, hide_index=True)
        st.caption(
            f"Training parameter sets: {len(trained.dataset.train_indices)} · "
            f"Held-out parameter sets: {len(trained.dataset.heldout_indices)} · Device: {trained.device}"
        )
        st.markdown("#### Inverse design target")
        objectives = ["ground_energy", "localization_width"]
        if family == "double_well":
            objectives.append("tunnel_splitting")
        objective = st.selectbox("Observable to target", objectives, key="design_objective")
        default_target = {"ground_energy": 0.8, "localization_width": 0.9, "tunnel_splitting": 0.05}[objective]
        target = st.number_input("Target value (dimensionless)", value=float(default_target), format="%.6f")
        initial = default_parameters(family)
        param_cols = st.columns(len(initial))
        initial_values = {}
        for col, (name, default_value) in zip(param_cols, initial.items(), strict=True):
            bound = next(b for b in PARAMETER_SPACES[family] if b.name == name)
            with col:
                initial_values[name] = st.number_input(
                    f"Initial {name}", min_value=float(bound.low), max_value=float(bound.high),
                    value=float(default_value), key=f"init_{family}_{name}", format="%.5f",
                )
        optimization_steps = st.slider("Optimization steps", 25, 600, 200, 25)
        if st.button("Optimize potential and verify with FD", use_container_width=True):
            try:
                optimized = optimize_potential(
                    trained.model, family, objective, float(target),
                    initial_parameters=initial_values, steps=int(optimization_steps),
                    domain=trained.dataset.domain, n_grid=max(801, int(design_grid) * 2 + 1),
                    fd_refine=True, fd_maxiter=60,
                )
                st.session_state["last_inverse_design"] = optimized
            except Exception as exc:
                st.exception(exc)
        optimized = st.session_state.get("last_inverse_design")
        if optimized is not None and optimized.family == family and optimized.objective == objective:
            st.markdown("#### Optimized design and independent verification")
            st.dataframe([{
                "Parameter": name,
                "PINN candidate": optimized.surrogate_parameters[name],
                "FD-polished value": value,
            } for name, value in optimized.parameters.items()], use_container_width=True, hide_index=True)
            a, b, c = st.columns(3)
            a.metric("Target", f"{optimized.target:.6g}")
            b.metric("Surrogate", f"{optimized.surrogate_value:.6g}", f"error {optimized.surrogate_target_error:.2e}")
            c.metric("FD verification", f"{optimized.finite_difference_value:.6g}", f"error {optimized.finite_difference_target_error:.2e}")
            st.caption(
                "The neural network proposes the first parameter vector; Powell polishing then minimizes "
                "the target error against the finite-difference solver. Accept the result only after checking "
                "the final FD target error and grid convergence."
            )
    else:
        st.info("Train a surrogate for the selected family to enable inverse design.")

with guide_tab:
    st.subheader("What this prototype can and cannot claim")
    st.markdown(
        """
        **Included:** 1D stationary eigenstates, eight model potentials, exact Dirichlet boundary enforcement, """
        """normalization and orthogonality penalties, finite-difference validation, expectation values, """
        """Crank–Nicolson wavepacket dynamics, and a Pauli-basis qubit encoding.

        **Research limitations:** this is a dimensionless 1D model, not yet a calibrated solver for a real """
        """semiconductor or molecular device. Results depend on domain size, grid resolution, potential """
        """regularity, network capacity, collocation sampling, and training convergence. Discontinuous square """
        """wells/barriers are useful benchmarks but can be harder for a smooth tanh network. Always inspect """
        """the reference error, residual, norm, orthogonality, and grid-convergence study before interpreting results.

        **Next research milestones:** (1) adaptive collocation and residual-based sampling; (2) domain decomposition """
        """for discontinuous interfaces; (3) parameterized potentials and inverse design; (4) 2D quantum dots; """
        """(5) TDSE PINNs; (6) variational quantum eigensolver (VQE) experiments with Qiskit or hardware backends.
        """
    )


def _npz_bytes(result) -> bytes:
    import io
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        x=result.x,
        potential=result.potential,
        energies=result.energies,
        reference_energies=result.reference_energies,
        wavefunctions=result.wavefunctions,
        reference_wavefunctions=result.reference_wavefunctions,
    )
    return stream.getvalue()
