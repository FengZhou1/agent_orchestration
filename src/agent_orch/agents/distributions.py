"""Action distributions and masked/normalised loss helpers for structured PPO."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical, Dirichlet


def _normalize_advantages(values: torch.Tensor) -> torch.Tensor:
    if len(values) <= 1:
        return values
    return (values - values.mean()) / (values.std(unbiased=False) + 1.0e-8)


def _normalize_phase_advantages(
    values: torch.Tensor, phase_groups: np.ndarray
) -> torch.Tensor:
    normalized = values.clone()
    for phase in np.unique(phase_groups):
        mask = torch.as_tensor(phase_groups == phase, dtype=torch.bool)
        normalized[mask] = _normalize_advantages(values[mask])
    return normalized


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values[mask].mean()
    return values.new_tensor(0.0)


def _masked_mse(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if bool(mask.any()):
        return torch.nn.functional.mse_loss(predicted[mask], target[mask])
    return predicted.new_tensor(0.0)


def _phase_mean(values: np.ndarray, phases: np.ndarray, phase: int) -> float:
    selected = values[phases == phase]
    return float(np.mean(selected)) if len(selected) else 0.0


def _concentrations(raw: torch.Tensor, minimum: float = 0.1) -> torch.Tensor:
    """Dirichlet concentrations from the raw head output.

    The clamp's upper bound scales with the floor.  Pinning it at a constant
    would mean that raising the floor to cut sampling noise also shrinks the
    range of mean compositions the head can express, so the two goals would
    fight; scaling keeps the reachable simplex roughly fixed while the action
    noise falls as ``1 / sqrt(minimum)``.  At the default floor of 1.0 the bound
    is 100, exactly as before.
    """

    minimum = max(float(minimum), 1.0e-3)
    return torch.clamp(
        torch.nn.functional.softplus(raw) + minimum,
        minimum,
        max(100.0, 50.0 * minimum),
    )


def _fixed_concentrations(raw: torch.Tensor, total: float) -> torch.Tensor:
    """Concentrations with a pinned total, so the mean is ``softmax(raw)``.

    The free parameterisation lets the head scale every concentration, which
    changes the action's spread without changing its mean.  The composition is
    scored at its mean, so that extra degree of freedom is a nuisance: the policy
    can raise the sampled reward by sharpening or spreading the action instead of
    improving the composition.  Pinning the total removes it and makes the
    gradient act on the mean alone.
    """

    return float(total) * torch.softmax(raw, dim=-1)


def _sample_categorical(
    raw: torch.Tensor,
    mask: torch.Tensor,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError("The sequential deployment step has no feasible target")
    logits = raw.masked_fill(~valid, -1.0e9)
    distribution = Categorical(logits=logits)
    choice = torch.argmax(logits) if deterministic else distribution.sample()
    return choice, distribution.log_prob(choice), distribution.entropy()


def _evaluate_categorical(
    raw: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    distribution = Categorical(logits=raw.masked_fill(~valid, -1.0e9))
    choice = action.long()
    return distribution.log_prob(choice), distribution.entropy()


def _sample_grouped_dirichlet(
    raw: torch.Tensor,
    mask: torch.Tensor,
    groups: int,
    width: int,
    deterministic: bool,
    concentration_min: float = 0.1,
    concentration_total: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = raw.reshape(groups, width)
    mask = mask.reshape(groups, width).bool()
    action = torch.zeros_like(raw)
    log_prob = raw.new_tensor(0.0)
    entropy = raw.new_tensor(0.0)
    for group in range(groups):
        indices = torch.where(mask[group])[0]
        if len(indices) == 0:
            continue
        if len(indices) == 1:
            action[group, indices[0]] = 1.0
            continue
        distribution = Dirichlet(
            _fixed_concentrations(raw[group, indices], concentration_total)
            if concentration_total is not None
            else _concentrations(raw[group, indices], concentration_min)
        )
        sample = distribution.mean if deterministic else distribution.sample()
        action[group, indices] = sample
        log_prob = log_prob + distribution.log_prob(sample)
        entropy = entropy + distribution.entropy()
    return action.reshape(-1), log_prob, entropy


def _evaluate_grouped_dirichlet(
    raw: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
    groups: int,
    width: int,
    concentration_min: float = 0.1,
    concentration_total: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = raw.reshape(groups, width)
    mask = mask.reshape(groups, width).bool()
    action = action.reshape(groups, width)
    log_prob = raw.new_tensor(0.0)
    entropy = raw.new_tensor(0.0)
    for group in range(groups):
        indices = torch.where(mask[group])[0]
        if len(indices) <= 1:
            continue
        distribution = Dirichlet(
            _fixed_concentrations(raw[group, indices], concentration_total)
            if concentration_total is not None
            else _concentrations(raw[group, indices], concentration_min)
        )
        sample = torch.clamp(action[group, indices], min=1e-8)
        sample = sample / sample.sum()
        log_prob = log_prob + distribution.log_prob(sample)
        entropy = entropy + distribution.entropy()
    return log_prob, entropy


def _grouped_dirichlet_log_probs(
    raw: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor,
    groups: int,
    width: int,
    concentration_min: float = 0.1,
    concentration_total: float | None = None,
) -> torch.Tensor:
    """Per-group Dirichlet log-density of an action, shape ``(groups,)``.

    The summed log-probability is all a single scalar advantage needs, but
    crediting each group with its own application's reward requires the group's
    own log-probability.  A group with fewer than two active models has nothing to
    decide and contributes zero.
    """

    raw = raw.reshape(groups, width)
    mask = mask.reshape(groups, width).bool()
    action = action.reshape(groups, width)
    out = raw.new_zeros(groups)
    for group in range(groups):
        indices = torch.where(mask[group])[0]
        if len(indices) <= 1:
            continue
        distribution = Dirichlet(
            _fixed_concentrations(raw[group, indices], concentration_total)
            if concentration_total is not None
            else _concentrations(raw[group, indices], concentration_min)
        )
        sample = torch.clamp(action[group, indices], min=1e-8)
        sample = sample / sample.sum()
        out[group] = distribution.log_prob(sample)
    return out


def _normalize_grouped_advantages(
    values: torch.Tensor,
    groups: np.ndarray,
    min_group_size: int = 2,
    mode: str = "center",
) -> torch.Tensor:
    """Normalise advantages inside each group rather than across the whole batch.

    A rollout can span several contexts the policy cannot influence.  Batch-wide
    normalisation then encodes mostly "which context am I in", and the resulting
    policy gradient rewards re-sampling whatever action happened to be drawn in
    the luckiest context.  Working inside the group removes that nuisance mean
    and leaves the effect of the decision the policy actually controls.

    ``mode`` decides how much of the group is removed:

    * ``"center"`` subtracts the group mean only.  Use this when the groups differ
      in how much the decision matters: dividing by a group whose reward barely
      moves would amplify pure sampling noise by the inverse of a near-zero
      spread.  Centring keeps each context's influence proportional to the effect
      the decision actually had there.
    * ``"standardize"`` also divides by the group standard deviation, which is
      the GRPO convention and is appropriate only when every group has a
      comparable spread.

    Groups smaller than ``min_group_size`` carry no within-group signal, so their
    advantages are zeroed rather than normalised into noise.
    """

    if mode not in ("center", "standardize"):
        raise ValueError(f"Unknown mode {mode!r}; expected center or standardize")
    normalized = torch.zeros_like(values)
    for group in np.unique(groups):
        mask = torch.as_tensor(groups == group, dtype=torch.bool)
        if int(mask.sum()) < max(2, int(min_group_size)):
            continue
        member = values[mask]
        centered = member - member.mean()
        if mode == "standardize":
            centered = centered / (member.std(unbiased=False) + 1.0e-8)
        normalized[mask] = centered
    return normalized


__all__ = [
    "_concentrations",
    "_fixed_concentrations",
    "_grouped_dirichlet_log_probs",
    "_evaluate_categorical",
    "_evaluate_grouped_dirichlet",
    "_masked_mean",
    "_masked_mse",
    "_normalize_advantages",
    "_normalize_grouped_advantages",
    "_normalize_phase_advantages",
    "_phase_mean",
    "_sample_categorical",
    "_sample_grouped_dirichlet",
]
