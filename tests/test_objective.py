from __future__ import annotations

import math

import numpy as np
import pytest

from agent_orch.deployment import DeploymentLibraryBuilder
from agent_orch.objective import (
    ObjectiveAudit,
    ObjectiveEvaluator,
    ObjectiveSpec,
    ReferenceScales,
    app_latency_reference,
)
from agent_orch.schema.loader import ScenarioLoader


@pytest.fixture(scope="module")
def scenario():
    return ScenarioLoader.load("configs/toy.yaml")


@pytest.fixture(scope="module")
def library(scenario):
    return DeploymentLibraryBuilder(scenario).build(min_entries=8, max_entries=32)


def _rates(scenario):
    return {
        (app.id, ingress): rate
        for app in scenario.applications.values()
        for ingress, rate in app.ingress_rates.items()
    }


def test_legacy_profile_reproduces_the_four_term_sum(scenario):
    spec = ObjectiveSpec.legacy()
    assert spec.constraint_names == ("llm", "service")
    assert set(spec.objective_terms) == {
        "goodput_normalized",
        "quality_normalized",
        "cost_normalized",
        "latency_normalized",
    }
    references = ReferenceScales.from_scenario(scenario, spec)
    evaluator = ObjectiveEvaluator(scenario, spec, references)
    app_latency = {
        app.id: 0.5 * app_latency_reference(scenario, app.id)
        for app in scenario.applications.values()
    }
    value = evaluator.evaluate_arrays(
        cost=0.5 * (references.cost_min + references.cost_max),
        mean_latency=0.5 * references.latency_reference,
        attainment=0.8,
        quality=0.6,
        app_latency=app_latency,
        arrival_rates=_rates(scenario),
    )
    assert value.components["cost_normalized"] == pytest.approx(0.5)
    assert value.components["latency_normalized"] == pytest.approx(0.5)
    assert value.components["goodput_normalized"] == pytest.approx(0.8)
    assert value.components["quality_normalized"] == pytest.approx(0.6)
    assert value.utility == pytest.approx(0.25 * (0.8 + 0.6 - 0.5 - 0.5))
    assert set(value.components) == {
        "utility",
        "cost_normalized",
        "latency_normalized",
        "goodput_normalized",
        "quality_normalized",
    }


def test_slo_constrained_moves_attainment_into_the_constraint_vector(scenario):
    spec = ObjectiveSpec.slo_constrained(0.9)
    assert spec.constraint_names == ("llm", "service", "attainment")
    assert spec.goodput_weight == 0.0
    assert "goodput_normalized" not in spec.objective_terms
    references = ReferenceScales.from_scenario(scenario, spec)
    evaluator = ObjectiveEvaluator(scenario, spec, references)
    value = evaluator.evaluate_arrays(
        cost=references.cost_min,
        mean_latency=0.1 * references.latency_reference,
        attainment=0.6,
        quality=0.5,
        app_latency={},
        arrival_rates=_rates(scenario),
    )
    assert len(value.constraints) == 3
    assert value.constraints[2] == pytest.approx(0.3)
    # The objective no longer contains an attainment term, so quality dominates.
    assert value.utility == pytest.approx(
        0.5 * 0.5 - 0.25 * 0.0 - 0.25 * value.components["latency_normalized"]
    )


def test_attainment_constraint_is_inactive_when_the_target_is_met(scenario):
    spec = ObjectiveSpec.slo_constrained(0.5)
    evaluator = ObjectiveEvaluator(scenario, spec, ReferenceScales.from_scenario(scenario, spec))
    value = evaluator.evaluate_arrays(
        cost=0.0,
        mean_latency=0.0,
        attainment=0.9,
        quality=0.5,
        app_latency={},
        arrival_rates=_rates(scenario),
    )
    assert value.constraints[2] == 0.0


def test_violation_labels_charge_the_matching_constraint(scenario):
    spec = ObjectiveSpec.slo_constrained(0.9)
    evaluator = ObjectiveEvaluator(scenario, spec, ReferenceScales.from_scenario(scenario, spec))
    common = dict(
        cost=0.0,
        mean_latency=0.0,
        attainment=1.0,
        quality=0.0,
        app_latency={},
        arrival_rates=_rates(scenario),
    )
    clean = evaluator.evaluate_arrays(**common)
    llm_bad = evaluator.evaluate_arrays(violation_labels=("llm_queue_overload:x",), **common)
    tool_bad = evaluator.evaluate_arrays(violation_labels=("service_unserved:a:b",), **common)
    assert clean.constraints[:2] == (0.0, 0.0)
    assert llm_bad.constraints[0] == pytest.approx(1.0)
    assert llm_bad.constraints[1] == pytest.approx(0.0)
    assert tool_bad.constraints[1] == pytest.approx(1.0)


def test_library_cost_bounds_are_tighter_than_theoretical(scenario, library):
    spec = ObjectiveSpec.legacy()
    theoretical = ReferenceScales.from_scenario(scenario, spec, None)
    bounded = ReferenceScales.from_scenario(scenario, spec, library)
    assert bounded.cost_source == "library+switch-on"
    assert theoretical.cost_source == "theoretical"
    assert bounded.cost_max < theoretical.cost_max
    steady = [entry.cost_per_period for entry in library.entries]
    assert bounded.cost_min == pytest.approx(min(steady))


def test_cost_bounds_fall_back_when_cost_bounds_is_theoretical(scenario, library):
    spec = ObjectiveSpec(profile="legacy", cost_bounds="theoretical")
    references = ReferenceScales.from_scenario(scenario, spec, library)
    assert references.cost_source == "theoretical"


def test_smooth_latency_map_never_saturates(scenario):
    spec = ObjectiveSpec(profile="legacy", latency_map="smooth")
    evaluator = ObjectiveEvaluator(scenario, spec, ReferenceScales.from_scenario(scenario, spec))
    app_latency = {app.id: 100.0 * app_latency_reference(scenario, app.id) for app in scenario.applications.values()}
    value = evaluator.evaluate_arrays(
        cost=0.0,
        mean_latency=1.0e6,
        attainment=1.0,
        quality=1.0,
        app_latency=app_latency,
        arrival_rates=_rates(scenario),
    )
    assert value.components["latency_normalized"] < 1.0
    assert value.components["latency_normalized"] == pytest.approx(100.0 / 101.0, abs=1e-6)
    assert value.diagnostics["latency_saturation_fraction"] == pytest.approx(1.0)


def test_audit_flags_a_zero_weight_term_as_inactive_and_detects_dead_terms(scenario):
    spec = ObjectiveSpec.slo_constrained(0.9)
    evaluator = ObjectiveEvaluator(scenario, spec, ReferenceScales.from_scenario(scenario, spec))
    audit = ObjectiveAudit(spec)
    for label, quality, attainment in (("a", 0.2, 0.9), ("b", 0.8, 0.5)):
        for _ in range(3):
            audit.observe(
                label,
                evaluator.evaluate_arrays(
                    cost=0.0,
                    mean_latency=0.0,
                    attainment=attainment,
                    quality=quality,
                    app_latency={},
                    arrival_rates=_rates(scenario),
                ),
            )
    summaries = {summary.term: summary for summary in audit.term_summaries()}
    assert summaries["goodput_normalized"].is_inactive
    assert not summaries["goodput_normalized"].is_dead()
    assert not summaries["quality_normalized"].is_dead()
    # Cost and latency never move between the two labels here.
    assert "cost_normalized" in audit.dead_terms()
    assert "latency_normalized" in audit.dead_terms()
    assert "goodput_normalized" not in audit.dead_terms()
    assert audit.utility_summary().between_label_spread == pytest.approx(0.3)


def test_audit_markdown_and_rows_are_well_formed(scenario):
    spec = ObjectiveSpec.slo_constrained(0.9)
    evaluator = ObjectiveEvaluator(scenario, spec, ReferenceScales.from_scenario(scenario, spec))
    audit = ObjectiveAudit(spec)
    audit.observe(
        "only",
        evaluator.evaluate_arrays(
            cost=0.0,
            mean_latency=0.0,
            attainment=0.5,
            quality=0.5,
            app_latency={},
            arrival_rates=_rates(scenario),
        ),
    )
    markdown = audit.to_markdown()
    assert "| term |" in markdown
    assert "| constraint |" in markdown
    assert "dead terms:" in markdown
    rows = audit.rows()
    assert rows and all("term" in row or "constraint" in row for row in rows)
    payload = audit.to_dict()
    assert set(payload) == {"spec", "utility", "terms", "constraints", "dead_terms", "label_means"}


def test_spec_from_mapping_switches_profile_and_overrides_weights():
    spec = ObjectiveSpec.from_mapping({"profile": "slo_constrained", "quality_weight": 0.7})
    assert spec.profile == "slo_constrained"
    assert spec.quality_weight == 0.7
    assert spec.attainment_target == 0.9
    assert spec.to_dict()["constraint_names"] == ["llm", "service", "attainment"]

    legacy = ObjectiveSpec.from_mapping(None)
    assert legacy.profile == "legacy"
    assert legacy.constraint_names == ("llm", "service")

    promoted = ObjectiveSpec.from_mapping({"profile": "legacy", "attainment_target": 0.8})
    assert promoted.goodput_weight == 0.25
    assert promoted.constraint_names == ("llm", "service", "attainment")


def test_spec_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        ObjectiveSpec(profile="nope")
    with pytest.raises(ValueError):
        ObjectiveSpec(cost_bounds="nope")
    with pytest.raises(ValueError):
        ObjectiveSpec(latency_map="nope")
    with pytest.raises(ValueError):
        ObjectiveSpec(quality_weight=-1.0)
    with pytest.raises(ValueError):
        ObjectiveSpec(attainment_target=1.5)


def test_reference_scales_round_trip_through_mapping(scenario):
    references = ReferenceScales.from_scenario(scenario, ObjectiveSpec.legacy())
    restored = ReferenceScales.from_mapping(references.to_dict())
    assert restored.cost_min == pytest.approx(references.cost_min)
    assert restored.cost_max == pytest.approx(references.cost_max)
    assert restored.app_latency == pytest.approx(references.app_latency)
    assert math.isclose(restored.latency_reference, references.latency_reference)


def test_evaluator_zero_constraints_matches_the_spec(scenario):
    for spec in (ObjectiveSpec.legacy(), ObjectiveSpec.slo_constrained(0.9)):
        evaluator = ObjectiveEvaluator(scenario, spec, ReferenceScales.from_scenario(scenario, spec))
        zeros = evaluator.zero_constraints()
        assert zeros.shape == (len(spec.constraint_names),)
        assert np.all(zeros == 0.0)
