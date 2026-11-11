"""Command-line entry point for QuantumPINN Lab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantum_pinn.config import Domain, PINNConfig
from quantum_pinn.physics.potentials import POTENTIAL_DEFAULTS, make_potential
from quantum_pinn.solvers.stationary import solve_stationary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Physics-informed 1D Schrödinger eigenstate solver")
    parser.add_argument("--potential", choices=sorted(POTENTIAL_DEFAULTS), default="harmonic")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--grid", type=int, default=601)
    parser.add_argument("--collocation", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--no-warm-start", action="store_true", help="Start the PINN randomly instead of fitting FD states")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.03)
    parser.add_argument("--barrier", type=float, default=0.5)
    parser.add_argument("--separation", type=float, default=1.2)
    parser.add_argument("--depth", type=float, default=8.0)
    parser.add_argument("--width", type=float, default=2.0)
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--charge", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    param_map = {
        "harmonic": {"omega": args.omega},
        "quartic": {"omega": args.omega, "strength": args.strength},
        "double_well": {"barrier": args.barrier, "separation": args.separation},
        "finite_well": {"depth": args.depth, "width": args.width},
        "gaussian_well": {"depth": args.depth, "sigma": args.sigma},
        "barrier": {"height": args.height, "width": args.width},
        "soft_coulomb": {"charge": args.charge, "softening": args.softening},
    }
    potential = make_potential(args.potential, **param_map.get(args.potential, {}))
    config = PINNConfig(
        n_states=args.states,
        epochs=args.epochs,
        pretrain_steps=args.pretrain_steps,
        n_eval=args.grid,
        n_collocation=args.collocation,
        device=args.device,
        warm_start_from_fd=not args.no_warm_start,
    )
    result = solve_stationary(potential, Domain(), config)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "stationary_result.npz",
        x=result.x,
        potential=result.potential,
        energies=result.energies,
        reference_energies=result.reference_energies,
        wavefunctions=result.wavefunctions,
        reference_wavefunctions=result.reference_wavefunctions,
    )
    summary = {
        "potential": potential.name,
        "parameters": potential.params,
        "device": result.device,
        "states": [
            {
                "state": d.state_index,
                "pinn_energy": d.pinn_energy,
                "reference_energy": d.reference_energy,
                "relative_energy_error_percent": d.relative_energy_error_percent,
                "residual_rms": d.residual_rms,
                "norm": d.norm,
                "orthogonality_error": d.orthogonality_error,
                "reference_fidelity": d.fidelity,
                "node_count": d.node_count,
            }
            for d in result.diagnostics
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved arrays and summary to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
