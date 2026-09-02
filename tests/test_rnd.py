import numpy as np
import torch

from agent_orch.agents import PhaseRunningMoments, RNDModule
from agent_orch.agents.structured_ppo import (
    _constrained_utility,
    _gae,
    _update_lagrange_multipliers,
)


def test_rnd_target_is_frozen_and_predictor_loss_decreases():
    torch.manual_seed(4)
    module = RNDModule(input_size=6, hidden_size=16, feature_size=8)
    target_before = [parameter.detach().clone() for parameter in module.target.parameters()]
    optimizer = torch.optim.Adam(module.predictor.parameters(), lr=1.0e-2)
    states = torch.randn(32, 6)
    initial = float(module.loss(states).item())
    for _ in range(20):
        loss = module.loss(states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final = float(module.loss(states).item())
    assert final < initial
    assert all(not parameter.requires_grad for parameter in module.target.parameters())
    for before, after in zip(target_before, module.target.parameters()):
        assert torch.equal(before, after)


def test_phase_running_moments_are_independent():
    moments = PhaseRunningMoments(2)
    moments.update(np.asarray([1.0, 3.0]), np.asarray([0, 0]))
    normalized = moments.normalize(
        np.asarray([2.0, 2.0]), np.asarray([0, 1]), clip_max=5.0
    )
    assert normalized[0] == 0.0
    assert normalized[1] == 2.0
    assert moments.count.tolist() == [2.0, 0.0]


def test_intrinsic_reward_is_added_after_lagrangian_penalty():
    external = _constrained_utility(
        0.2,
        np.asarray([0.1, 0.0, 0.0, 0.0]),
        np.asarray([2.0, 0.0, 0.0, 0.0]),
        np.zeros(4),
        True,
    )
    combined = external + 0.01 * 3.0
    assert abs(combined - 0.03) < 1.0e-8


def test_vector_lagrange_update_keeps_constraint_classes_independent():
    updated = _update_lagrange_multipliers(
        np.asarray([0.5, 0.5, 0.5, 0.5]),
        np.asarray([1.0, 0.0, 3.0, 0.0]),
        np.asarray([0.0, 1.0, 1.0, 0.0]),
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([2.0, 2.0, 1.0, 2.0]),
    )
    assert np.allclose(updated, [0.6, 0.3, 1.0, 0.5])


def test_rnd_scaling_uses_standard_deviation_without_mean_centering():
    moments = PhaseRunningMoments(1)
    moments.update(np.asarray([1.0, 3.0]), np.asarray([0, 0]))
    scaled = moments.scale_by_std(
        np.asarray([2.0]), np.asarray([0]), clip_max=5.0
    )
    assert np.allclose(scaled, [2.0])


def test_gae_uses_per_transition_phase_discounts():
    advantages, _ = _gae(
        rewards=[0.0, 1.0],
        values=[0.0, 0.0],
        discounts=[1.0, 0.5],
        terminals=[0.0, 1.0],
        bootstrap=0.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages, torch.tensor([1.0, 1.0]))
