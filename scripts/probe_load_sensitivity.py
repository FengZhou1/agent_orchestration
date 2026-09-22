"""Does a trained composition policy respond to the load at all?

The single most informative pre-flight check for Stage A: a policy trained on one
load level is load-blind -- it emits the same shares at 0.4x and 2.0x, because its
input never carried a difference it had to explain (measured: L1 distance
0.000000).  If training on a mix-varying family does not make the output move with
the load either, then the family is not creating a learnable target and the PPO
run would be measuring nothing.

Reports the L1 distance between the policy's deterministic composition under
several load vectors, and -- the part that matters -- how much of that movement is
in the direction the teacher wants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent_orch.agents import PPOConfig, StructuredActorCritic
from agent_orch.deployment import DeploymentLibrary
from agent_orch.envs.scoring import build_composition_env, dense_composition_action
from agent_orch.objective import ObjectiveSpec
from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SCALE = 7.165234375


def _policy_share(scenario, objective, library, position, trace, periods, state, mapping_samples):
    env = build_composition_env(
        scenario,
        objective,
        trace,
        library,
        position=position,
        periods=periods,
        mapping_samples=mapping_samples,
        seed=0,
    )
    policy = StructuredActorCritic(env, PPOConfig())
    policy.load_state_dict(state)
    policy.eval()
    observation, _ = env.reset(seed=0)
    action, _, _ = policy.act(observation, deterministic=True, device="cpu")
    return (
        np.asarray(action["model"], dtype=float).ravel(),
        dense_composition_action(env, {}),
        env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/benchmarks/main_abilene.yaml")
    parser.add_argument("--policy", required=True, help="checkpoint to probe")
    parser.add_argument("--label", default="policy")
    parser.add_argument("--positions", default="0,5,20")
    parser.add_argument("--periods", type=int, default=8)
    parser.add_argument("--mapping-samples", type=int, default=128)
    parser.add_argument("--seeds", default="0,5,11", help="mix seeds to compare")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scenario = ScenarioLoader.load(REPO_ROOT / args.scenario)
    objective = ObjectiveSpec.slo_constrained(0.9)
    library = DeploymentLibrary.load(DeploymentLibrary.default_path(scenario.id))
    state = torch.load(REPO_ROOT / args.policy, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "policy_state_dict" in state:
        state = state["policy_state_dict"]

    positions = [int(value) for value in args.positions.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    loads = {}
    for seed in seeds:
        loads[f"mix{seed}"] = ArrivalTrace.randomized_mix_intensity(
            scenario, args.periods, base_scale=BASE_SCALE, seed=seed, block=args.periods
        )
    loads["stationary_high"] = ArrivalTrace.stationary_poisson_intensity(
        scenario, args.periods, rate_scale=BASE_SCALE
    )
    loads["stationary_low"] = ArrivalTrace.stationary_poisson_intensity(
        scenario, args.periods, rate_scale=BASE_SCALE * 0.4
    )

    print(f"policy: {args.label}  ({args.policy})")
    results: dict[str, float] = {}
    for position in positions:
        entry = library.entries[position]
        shares = {}
        for name, trace in loads.items():
            share, _, _ = _policy_share(
                scenario, objective, library, position, trace, args.periods, state,
                args.mapping_samples,
            )
            shares[name] = share
        names = list(loads)
        print(f"\n  position {position} ({entry.stratum}), n_models={entry.n_models}")
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                distance = float(np.abs(shares[left] - shares[right]).sum())
                results[f"{position}|{left}-{right}"] = distance
        for name in names:
            top = shares[name].reshape(-1, len(scenario.models))
            biggest = np.sort(top, axis=1)[:, -1]
            print(
                f"    {name:<16} mean top share {biggest.mean():.3f}  "
                f"max |delta vs mix0| {float(np.abs(shares[name] - shares['mix%d' % seeds[0]]).max()):.4f}"
            )

    moves = [value for key, value in results.items() if "mix" in key and "-mix" in key]
    stationary = [
        value for key, value in results.items()
        if "stationary" in key
    ]
    print(
        f"\nacross-mix movement: mean {np.mean(moves):.4f} (max {max(moves):.4f})"
        if moves else "\nno mix pairs compared"
    )
    if stationary:
        print(f"across-stationary movement: mean {np.mean(stationary):.4f} (max {max(stationary):.4f})")
    verdict = bool(moves) and bool(np.mean(moves) > 0.02)
    print(
        "VERDICT: the policy responds to the load."
        if verdict
        else "VERDICT: the policy is still load-blind; training on the family did not create "
        "a load-conditioned response."
    )
    if args.output:
        target = REPO_ROOT / args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"policy": args.policy, "distances": results, "responds": bool(verdict)}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
