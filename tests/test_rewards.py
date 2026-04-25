"""Phase 1 contract tests for reward functions."""

from werewolf_env.game.constants import MAX_DAYS, Role
from werewolf_env.game.engine import GameEngine
from werewolf_env.game.rewards import (
    reward_calibration,
    reward_format,
    reward_outcome,
    reward_survival,
)


def _build_state(seeded_rng, *, winner=None, game_id="g1"):
    engine = GameEngine()
    state = engine.new_game(game_id, seeded_rng)
    if winner is not None:
        state.winner = winner
    return state


def _werewolf_id(state):
    return next(pid for pid, p in state.players.items() if p.role == Role.WEREWOLF)


def _villager_id(state):
    return next(pid for pid, p in state.players.items() if p.role == Role.VILLAGER)


def test_outcome_returns_1_for_winning_faction(seeded_rng):
    state = _build_state(seeded_rng, winner="villager")
    villager = _villager_id(state)
    assert reward_outcome(state, villager) == 1.0


def test_outcome_returns_0_for_losing_faction(seeded_rng):
    state = _build_state(seeded_rng, winner="villager")
    werewolf = _werewolf_id(state)
    assert reward_outcome(state, werewolf) == 0.0


def test_calibration_zero_when_all_accusations_wrong(seeded_rng):
    """A werewolf that only accuses other villagers (its own faction-mates'
    *opposing* faction is villagers, but villagers are the werewolf's
    OPPOSING faction... wait, a werewolf accusing villagers IS calibrated.
    Let me reframe: a VILLAGER who accuses other villagers is mis-calibrated."""
    state = _build_state(seeded_rng, winner="villager")
    villager = _villager_id(state)
    # Find another villager and the seer (also villager-faction)
    same_faction = [
        pid for pid, p in state.players.items()
        if pid != villager and p.role in (Role.VILLAGER, Role.SEER)
    ]
    state.players[villager].suspicions_stated = same_faction
    assert reward_calibration(state, villager) == 0.0


def test_calibration_one_when_all_accusations_correct(seeded_rng):
    state = _build_state(seeded_rng, winner="villager")
    villager = _villager_id(state)
    werewolf = _werewolf_id(state)
    state.players[villager].suspicions_stated = [werewolf]
    assert reward_calibration(state, villager) == 1.0


def test_calibration_neutral_when_no_accusations(seeded_rng):
    state = _build_state(seeded_rng)
    villager = _villager_id(state)
    state.players[villager].suspicions_stated = []
    assert reward_calibration(state, villager) == 0.5


def test_survival_full_for_alive_at_end(seeded_rng):
    state = _build_state(seeded_rng, winner="villager")
    villager = _villager_id(state)
    assert state.players[villager].alive
    assert reward_survival(state, villager) == 1.0


def test_survival_partial_for_dead_at_day_2(seeded_rng):
    state = _build_state(seeded_rng, winner="werewolf")
    villager = _villager_id(state)
    state.players[villager].alive = False
    state.players[villager].day_died = 2
    expected = min(1.0, 2 / MAX_DAYS)
    assert reward_survival(state, villager) == expected


def test_format_floors_at_zero(seeded_rng):
    state = _build_state(seeded_rng, winner="villager")
    villager = _villager_id(state)
    state.players[villager].format_violations = 0
    assert reward_format(state, villager) == 1.0
    state.players[villager].format_violations = 1
    assert reward_format(state, villager) == 0.75
    state.players[villager].format_violations = 4
    assert reward_format(state, villager) == 0.0
    state.players[villager].format_violations = 100
    assert reward_format(state, villager) == 0.0  # floored
