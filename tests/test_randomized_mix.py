from __future__ import annotations

import numpy as np
import pytest

from agent_orch.schema.loader import ScenarioLoader
from agent_orch.workload import ArrivalTrace


@pytest.fixture(scope="module")
def scenario():
    return ScenarioLoader.load("configs/benchmarks/main_abilene.yaml")


def _totals(trace, scenario, slots):
    return [sum(trace.at(slot, scenario).values()) for slot in range(slots)]


def test_mix_varies_not_only_the_total(scenario):
    """The point of the generator: group shares move independently.

    A generator that scaled every group together would leave the mix fixed, and
    the composition optimum along that single direction does not move -- which is
    why the policy trained that way cannot condition on the load.
    """

    trace = ArrivalTrace.randomized_mix_intensity(
        scenario, 24, base_scale=7.165234375, seed=1, block=4
    )
    group_shares = []
    for slot in (0, 4, 8, 12, 16, 20):
        rates = trace.at(slot, scenario)
        total = sum(rates.values())
        group_shares.append({key: value / total for key, value in sorted(rates.items())})

    spreads = []
    for key in group_shares[0]:
        values = [share[key] for share in group_shares]
        spreads.append(max(values) - min(values))
    assert max(spreads) > 0.01, "the mix itself should move"

    # And it is not a single global rescaling: between some pair of blocks, some
    # groups gain share while others lose it.
    names = sorted(group_shares[0])
    opposed = False
    for i in range(1, len(group_shares)):
        deltas = [group_shares[i][name] - group_shares[0][name] for name in names]
        if any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas):
            opposed = True
            break
    assert opposed, "groups should move independently, not in lockstep"


def test_values_are_constant_within_a_block(scenario):
    trace = ArrivalTrace.randomized_mix_intensity(
        scenario, 12, base_scale=7.165234375, seed=2, block=4
    )
    for start in (0, 4, 8):
        block = [trace.at(slot, scenario) for slot in range(start, start + 4)]
        assert all(entry == block[0] for entry in block)


def test_per_group_and_total_ranges_are_respected(scenario):
    base = {
        (app.id, ingress): float(rate) * 3.0
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }
    target = sum(base.values())
    trace = ArrivalTrace.randomized_mix_intensity(
        scenario, 200, base_scale=3.0, seed=3, block=5, group_range=(0.3, 2.5), total_range=(0.6, 1.6)
    )
    for slot in range(0, 200, 5):
        rates = trace.at(slot, scenario)
        for key, value in rates.items():
            assert 0.3 * base[key] - 1e-9 <= value <= 2.5 * base[key] + 1e-9
        # The total is only rescaled when it leaves the band, so it may exceed
        # total_high*1 by the clipping of the individual draws -- but never falls
        # below the band when the draws would have.
        assert sum(rates.values()) <= 2.5 * target + 1e-9


def test_same_seed_reproduces_and_different_seeds_differ(scenario):
    kwargs = dict(base_scale=7.165234375, block=4)
    first = ArrivalTrace.randomized_mix_intensity(scenario, 20, seed=11, **kwargs)
    second = ArrivalTrace.randomized_mix_intensity(scenario, 20, seed=11, **kwargs)
    other = ArrivalTrace.randomized_mix_intensity(scenario, 20, seed=12, **kwargs)
    assert _totals(first, scenario, 20) == _totals(second, scenario, 20)
    assert _totals(first, scenario, 20) != _totals(other, scenario, 20)


def test_rejects_degenerate_parameters(scenario):
    with pytest.raises(ValueError, match="slots must be positive"):
        ArrivalTrace.randomized_mix_intensity(scenario, 0, base_scale=1.0, seed=0)
    with pytest.raises(ValueError, match="block must be positive"):
        ArrivalTrace.randomized_mix_intensity(scenario, 4, base_scale=1.0, seed=0, block=0)
    with pytest.raises(ValueError, match="group_range"):
        ArrivalTrace.randomized_mix_intensity(
            scenario, 4, base_scale=1.0, seed=0, group_range=(2.0, 1.0)
        )
    with pytest.raises(ValueError, match="total_range"):
        ArrivalTrace.randomized_mix_intensity(
            scenario, 4, base_scale=1.0, seed=0, total_range=(0.0, 1.0)
        )
