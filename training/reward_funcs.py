"""TRL-compatible reward callables.

Each function takes whatever extra columns the rollout function emitted and
returns a list of floats — one per group sample. TRL sums them (weighted by
the order in which they're registered) and shows each as its own column in
Trackio.

Phase 4 deliverable.
"""

from typing import List


def reward_outcome_fn(completions, **kwargs) -> List[float]:  # noqa: ARG001
    return list(kwargs["outcome_reward"])


def reward_calibration_fn(completions, **kwargs) -> List[float]:  # noqa: ARG001
    return list(kwargs["calibration_reward"])


def reward_survival_fn(completions, **kwargs) -> List[float]:  # noqa: ARG001
    return list(kwargs["survival_reward"])


def reward_format_fn(completions, **kwargs) -> List[float]:  # noqa: ARG001
    return list(kwargs["format_reward"])


# Weights are applied implicitly by passing the four functions in order with
# matching `reward_weights=[0.50, 0.25, 0.15, 0.10]` to GRPOTrainer.
DEFAULT_REWARD_WEIGHTS = [0.50, 0.25, 0.15, 0.10]
