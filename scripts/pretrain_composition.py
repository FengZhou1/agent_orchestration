"""Supervised warm start for the composition policy, distilled from the solver.

Near-uniform Dirichlet exploration has to rediscover "put mass on the strong
model when it is available" from a noisy scalar objective.  The inner solver
already knows that answer for a set of deployments, so we clone its composition
into the policy before PPO starts and let PPO refine it.

The target is the Dirichlet *mean* the actor produces, which is what ``act``
returns when called deterministically, so the distilled policy at
``--learning-rate``-optimality reproduces the teacher exactly.

Usage::

    python scripts/pretrain_composition.py \\
        --scenario configs/benchmarks/main_abilene.yaml \\
        --reference data/processed/composition_reference_agent-abilene-20.json \\
        --output results/stage_a/pretrain
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_orch.agents import PPOConfig, StructuredActorCritic
from agent_orch.agents.distributions import _concentrations
from agent_orch.agents.rollout import _observation_to_tensors
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs import CompositionLibraryEnv
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.telemetry import build_run_logger
from agent_orch.workload import ArrivalTrace


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _group_targets(
    layout,
    share: dict[str, float],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Expand a per-(app, ingress, model) share into the grouped action layout."""

    groups = len(layout.model_groups)
    width = len(layout.models)
    target = torch.zeros(groups, width)
    for group_index, (app_id, ingress) in enumerate(layout.model_groups):
        for model_index, model in enumerate(layout.models):
            target[group_index, model_index] = float(
                share.get(f"{app_id}|{ingress}|{model}", 0.0)
            )
    target = target * mask.reshape(groups, width).float()
    total = target.sum(dim=1, keepdim=True)
    return torch.where(total > 0.0, target / total.clamp_min(1.0e-12), target)


def _predicted_mean(
    policy: StructuredActorCritic, observation: dict, device: str
) -> torch.Tensor:
    obs = _observation_to_tensors(observation, device, batched=False)
    _, composition_hidden, _ = policy._encoded_phases(obs)  # noqa: SLF001 - teacher fitting
    logits = policy._composition_logits(composition_hidden).squeeze(0)  # noqa: SLF001
    groups = len(policy.layout.model_groups)
    width = len(policy.layout.models)
    raw = logits.reshape(groups, width)
    mask = torch.as_tensor(observation["model_mask"], device=device).reshape(groups, width).bool()
    concentration = _concentrations(raw, policy.config.composition_concentration_min)
    masked = concentration * mask.float()
    normalizer = masked.sum(dim=1, keepdim=True)
    return torch.where(normalizer > 0.0, masked / normalizer.clamp_min(1.0e-12), masked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--deployment-library", default=None)
    parser.add_argument("--objective-profile", default="slo_constrained")
    parser.add_argument("--attainment-target", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapping-samples", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--telemetry", default="both")
    parser.add_argument("--swanlab-project", default="agent-orch")
    parser.add_argument("--swanlab-online", action="store_true")
    parser.add_argument("--output", default="results/stage_a/pretrain")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scenario = ScenarioLoader.load(args.scenario)
    spec = (
        ObjectiveSpec.slo_constrained(args.attainment_target)
        if args.objective_profile == "slo_constrained"
        else ObjectiveSpec.legacy()
    )
    library = DeploymentLibrary.load(
        args.deployment_library or DeploymentLibrary.default_path(scenario.id)
    )
    payload = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    entries = {int(key): value for key, value in payload["entries"].items()}
    arrival_scale = float(payload["arrival_scale"])

    position_of_source = {entry.index: position for position, entry in enumerate(library.entries)}
    samples: list[tuple[int, dict[str, float]]] = []
    for source_index, record in sorted(entries.items()):
        if source_index not in position_of_source:
            continue
        if not record.get("model_share"):
            continue
        samples.append((position_of_source[source_index], record["model_share"]))
    if not samples:
        raise SystemExit(
            "The reference file contains no composition for this deployment library; "
            "run scripts/solve_composition_reference.py against the same library first."
        )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(samples))
    n_validation = max(1, int(round(len(samples) * args.validation_fraction)))
    validation_positions = {order[index] for index in range(n_validation)}
    train_samples = [sample for index, sample in enumerate(samples) if index not in validation_positions]
    validation_samples = [sample for index, sample in enumerate(samples) if index in validation_positions]

    env = CompositionLibraryEnv(
        scenario,
        max_slots=1,
        seed=args.seed,
        arrival_trace=ArrivalTrace.stationary_poisson_intensity(scenario, 1, rate_scale=arrival_scale),
        mapping_samples=args.mapping_samples,
        objective=spec,
        library=library,
    )
    policy = StructuredActorCritic(env, PPOConfig()).to(args.device)
    policy.set_training_phase("composition")
    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in policy.composition_encoder.parameters()
        ]
        + list(policy.model_head.parameters()),
        lr=args.learning_rate,
    )

    def observe(position: int) -> dict:
        observation, _ = env.reset(seed=args.seed, options={"fixed_deployment_index": position})
        return observation

    train_inputs = [(observe(position), record) for position, record in train_samples]
    validation_inputs = [(observe(position), record) for position, record in validation_samples]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logger = build_run_logger(
        args.telemetry,
        output_dir=output_dir,
        run_name=f"composition-distill-{scenario.id}",
        config={
            "scenario": scenario.id,
            "reference": str(args.reference),
            "objective": spec.to_dict(),
            "arrival_scale": arrival_scale,
            "n_train": len(train_inputs),
            "n_validation": len(validation_inputs),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        project=args.swanlab_project,
        offline=not args.swanlab_online,
    )

    history: list[dict[str, float]] = []
    policy.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for observation, record in train_inputs:
            target = _group_targets(
                policy.layout,
                record,
                torch.as_tensor(observation["model_mask"], device=args.device),
            ).to(args.device)
            predicted = _predicted_mean(policy, observation, args.device)
            mask = torch.as_tensor(observation["model_mask"], device=args.device).reshape(
                target.shape
            ).bool()
            difference = (predicted - target) * mask.float()
            loss = (difference * difference).sum() / mask.float().sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                1.0,
            )
            optimizer.step()
            total += float(loss.detach())
        mean_loss = total / max(1, len(train_inputs))

        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1 or epoch == args.epochs:
            validation_mae = _mean_absolute_error(policy, validation_inputs, args.device)
            train_mae = _mean_absolute_error(policy, train_inputs, args.device)
            record_row = {
                "loss": mean_loss,
                "train_mean_absolute_error": train_mae,
                "validation_mean_absolute_error": validation_mae,
                "epoch": float(epoch),
            }
            history.append(record_row)
            run_logger.log_update(epoch, record_row)
            print(
                "epoch %4d  loss=%.6f  train_mae=%.5f  validation_mae=%.5f"
                % (epoch, mean_loss, train_mae, validation_mae),
                flush=True,
            )

    final_train_mae = _mean_absolute_error(policy, train_inputs, args.device)
    final_validation_mae = _mean_absolute_error(policy, validation_inputs, args.device)
    checkpoint_path = output_dir / "policy_pretrained.pt"
    torch.save(
        {
            "policy_state_dict": {
                key: value.detach().cpu() for key, value in policy.state_dict().items()
            },
            "ppo_config": PPOConfig().to_dict() if hasattr(PPOConfig(), "to_dict") else {},
            "layout_signature": {
                "models": env.layout.models,
                "candidates": env.layout.candidates,
                "servers": env.layout.servers,
                "deployment_targets": env.layout.deployment_targets,
                "model_groups": env.layout.model_groups,
            },
            "selection": "distilled",
            "training_phase": "composition",
            "distillation": {
                "reference": str(args.reference),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "train_mean_absolute_error": final_train_mae,
                "validation_mean_absolute_error": final_validation_mae,
                "n_train": len(train_inputs),
                "n_validation": len(validation_inputs),
                "seed": args.seed,
                "finished_at_utc": _timestamp(),
            },
        },
        checkpoint_path,
    )
    run_logger.finish()

    print(
        "\ndistilled %d contexts (train %d / validation %d): "
        "train MAE %.5f, validation MAE %.5f\nwrote %s"
        % (
            len(samples),
            len(train_inputs),
            len(validation_inputs),
            final_train_mae,
            final_validation_mae,
            checkpoint_path,
        )
    )
    return 0


def _mean_absolute_error(policy, samples, device: str) -> float:
    if not samples:
        return float("nan")
    policy.eval()
    errors: list[float] = []
    with torch.no_grad():
        for observation, record in samples:
            target = _group_targets(
                policy.layout,
                record,
                torch.as_tensor(observation["model_mask"], device=device),
            ).to(device)
            predicted = _predicted_mean(policy, observation, device)
            mask = torch.as_tensor(observation["model_mask"], device=device).reshape(
                target.shape
            ).bool()
            errors.append(float(((predicted - target).abs() * mask.float()).sum() / mask.float().sum().clamp_min(1.0)))
    policy.train()
    return float(np.mean(errors))


if __name__ == "__main__":
    raise SystemExit(main())
