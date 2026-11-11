"""CLI for parameter-conditioned surrogate training, evaluation and inverse design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_pinn.config import Domain
from quantum_pinn.design.dataset import SurrogateDataset, generate_dataset
from quantum_pinn.design.inverse import optimize_potential
from quantum_pinn.design.training import (
    SurrogateTrainingConfig, SurrogateTrainingResult, evaluate_heldout,
    load_surrogate, save_surrogate, train_surrogate,
)
from quantum_pinn.design.parameter_space import PARAMETER_SPACES, default_parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuantumPINN v2 parameter-conditioned design tools")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Generate FD training data and train a conditional PINN surrogate")
    train.add_argument("--family", choices=sorted(PARAMETER_SPACES), default="double_well")
    train.add_argument("--configurations", type=int, default=256)
    train.add_argument("--grid", type=int, default=241)
    train.add_argument("--states", type=int, default=2)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--steps-per-epoch", type=int, default=4)
    train.add_argument("--physics-every", type=int, default=4)
    train.add_argument("--width", type=int, default=96)
    train.add_argument("--hidden-layers", type=int, default=4)
    train.add_argument("--physics-weight", type=float, default=0.10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--out", type=Path, default=Path("outputs/double_well_surrogate.pt"))

    evaluate = commands.add_parser("evaluate", help="Evaluate a saved surrogate on held-out FD parameter combinations")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, default=Path("outputs/heldout_metrics.json"))
    evaluate.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    optimize = commands.add_parser("optimize", help="Optimize potential parameters for a target quantum observable")
    optimize.add_argument("--checkpoint", type=Path, required=True)
    optimize.add_argument("--objective", choices=["ground_energy", "localization_width", "tunnel_splitting"], required=True)
    optimize.add_argument("--target", type=float, required=True)
    optimize.add_argument("--steps", type=int, default=300)
    optimize.add_argument("--learning-rate", type=float, default=0.03)
    optimize.add_argument("--initial", type=str, default=None, help="Optional JSON parameter object, e.g. '{\"barrier\":0.7,\"separation\":1.2}'")
    optimize.add_argument("--grid", type=int, default=801, help="FD verification grid size")
    optimize.add_argument("--fd-maxiter", type=int, default=60, help="Maximum Powell iterations for FD polishing")
    optimize.add_argument("--no-fd-refine", action="store_true", help="Only verify the surrogate candidate; skip direct-FD polishing")
    optimize.add_argument("--out", type=Path, default=Path("outputs/inverse_design.json"))
    optimize.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        n_states = max(args.states, 2 if args.family == "double_well" else args.states)
        dataset = generate_dataset(
            args.family, n_configurations=args.configurations, n_grid=args.grid,
            n_states=n_states, seed=args.seed,
        )
        dataset_path = args.out.with_name(f"{args.out.stem}_fd_dataset.npz")
        dataset.save(dataset_path)
        config = SurrogateTrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch,
            physics_every=args.physics_every, width=args.width,
            hidden_layers=args.hidden_layers, physics_weight=args.physics_weight,
            seed=args.seed, device=args.device,
        )
        print(f"Generated {dataset.n_configurations} FD parameter combinations; "
              f"training={len(dataset.train_indices)}, held-out={len(dataset.heldout_indices)}")
        result = train_surrogate(dataset, config, progress_callback=lambda e, n, loss: print(
            f"epoch {e:>4}/{n}  loss={loss:.5e}") if e == 1 or e % 10 == 0 or e == n else None)
        checkpoint = save_surrogate(result, args.out, config)
        metrics = evaluate_heldout(result)
        report = {"family": args.family, "checkpoint": str(checkpoint), "dataset": str(dataset_path),
                  "heldout_metrics": metrics, "final_training_loss": result.history[-1]}
        report_path = args.out.with_suffix(".metrics.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    if args.command == "evaluate":
        dataset = SurrogateDataset.load(args.dataset)
        model, metadata = load_surrogate(args.checkpoint, args.device)
        if dataset.family != metadata["family"]:
            raise ValueError("Dataset family does not match checkpoint family")
        result = SurrogateTrainingResult(
            model=model, history=[], energy_scale=float(metadata["energy_scale"]),
            device=args.device, family=dataset.family, dataset=dataset,
        )
        metrics = evaluate_heldout(result)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "optimize":
        model, metadata = load_surrogate(args.checkpoint, args.device)
        family = metadata["family"]
        initial = json.loads(args.initial) if args.initial else default_parameters(family)
        result = optimize_potential(
            model, family, args.objective, args.target, initial_parameters=initial,
            steps=args.steps, learning_rate=args.learning_rate,
            domain=Domain(metadata["xmin"], metadata["xmax"]), n_grid=args.grid,
            fd_refine=not args.no_fd_refine, fd_maxiter=args.fd_maxiter,
        )
        report = {
            "family": result.family, "objective": result.objective, "target": result.target,
            "surrogate_candidate_parameters": result.surrogate_parameters,
            "optimized_parameters_after_fd_polish": result.parameters,
            "surrogate_candidate_prediction": result.surrogate_candidate_value,
            "fd_value_at_surrogate_candidate": result.finite_difference_candidate_value,
            "surrogate_prediction_at_final_parameters": result.surrogate_value,
            "finite_difference_verification": result.finite_difference_value,
            "fd_refined": result.fd_refined,
            "surrogate_absolute_target_error": result.surrogate_target_error,
            "finite_difference_absolute_target_error": result.finite_difference_target_error,
            "finite_difference_eigenvalues": result.reference_energies,
            "optimizer_final_loss": result.history[-1],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
