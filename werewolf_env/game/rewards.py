"""Pure reward functions, each returning a float in [0, 1].

Composed by the trainer as:
    0.50 * outcome  +  0.25 * calibration  +  0.15 * survival  +  0.10 * format

But registered to GRPOTrainer as four SEPARATE callables so each shows up
as its own column in Trackio (judges value multiple independent rubrics).
"""

from __future__ import annotations

from werewolf_env.game.constants import MAX_DAYS, ROLE_TO_FACTION
from werewolf_env.game.state import GameState


def reward_outcome(state: GameState, player_id: int) -> float:
    """1.0 if `player_id`'s faction won, 0.0 otherwise."""
    if state.winner is None:
        return 0.0
    player_faction = ROLE_TO_FACTION[state.players[player_id].role].value
    return 1.0 if state.winner == player_faction else 0.0


def reward_calibration(state: GameState, player_id: int) -> float:
    """Mean accuracy of the player's stated suspicions vs ground truth.

    Each suspicion is correct if the accused player is in the OPPOSING
    faction. Returns 0.5 if the player never accused anyone (neutral —
    don't reward or punish silence here; format reward handles that).
    """
    suspicions = state.players[player_id].suspicions_stated
    if not suspicions:
        return 0.5
    player_faction = ROLE_TO_FACTION[state.players[player_id].role]
    correct = sum(
        1
        for sid in suspicions
        if sid in state.players
        and ROLE_TO_FACTION[state.players[sid].role] != player_faction
    )
    return correct / len(suspicions)


def reward_survival(state: GameState, player_id: int) -> float:
    """Fraction of game days the player survived. 1.0 if alive at end."""
    p = state.players[player_id]
    if p.alive:
        return 1.0
    if p.day_died is None:
        return 0.0
    # Survived through `day_died` inclusive (died at end of that day).
    return min(1.0, p.day_died / MAX_DAYS)


def reward_format(state: GameState, player_id: int) -> float:
    """1.0 - 0.25 * format_violations, floored at 0.0."""
    violations = state.players[player_id].format_violations
    return max(0.0, 1.0 - 0.25 * violations)
