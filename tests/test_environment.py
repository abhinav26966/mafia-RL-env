"""Phase 2 contract tests for the OpenEnv server (WerewolfEnvironment).

These tests run against the env class directly — no HTTP boot. The
WebSocket round-trip is exercised separately in test_client.py.
"""

import random

import pytest

from werewolf_env.models import WerewolfAction
from werewolf_env.server.werewolf_environment import WerewolfEnvironment


# ── Phase 0 smoke (already passing) ──────────────────────────────────────────


def test_environment_imports_cleanly():
    env = WerewolfEnvironment()
    obs = env.reset()
    assert obs is not None
    assert hasattr(obs, "player_id")
    assert hasattr(obs, "role")


def test_environment_supports_concurrent_sessions():
    assert WerewolfEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True


# ── Phase 2 contract tests ───────────────────────────────────────────────────


def test_reset_returns_observation_with_legal_actions():
    """A reset where the spotlight survives night 1 must give the trainer
    an observation with a non-empty legal_action_types list and the
    spotlight as current_actor."""
    env = WerewolfEnvironment()
    obs = env.reset(seed=2)  # seat 0 = villager, day_discussion (alive)
    assert obs.done is False
    assert obs.current_actor_id == 0
    assert len(obs.legal_action_types) > 0
    assert obs.role in ("werewolf", "seer", "villager")
    # Public/private logs must be Pydantic models, not raw dicts
    from werewolf_env.models import PrivateNote, PublicEvent
    if obs.public_log:
        assert isinstance(obs.public_log[0], PublicEvent)
    if obs.private_log:
        assert isinstance(obs.private_log[0], PrivateNote)


def test_step_advances_state():
    """A valid step must return a fresh observation; state.step_count grows."""
    env = WerewolfEnvironment()
    obs = env.reset(seed=2)
    initial_history = env.state.history_length
    # Spotlight is in day_discussion → speak action
    assert "speak" in obs.legal_action_types
    action = WerewolfAction(
        player_id=0,
        action_type="speak",
        content="I think Player 4 has been acting suspicious. We should look at them.",
    )
    obs2 = env.step(action)
    assert env.state.history_length > initial_history
    # We may now be back to the spotlight (vote phase) or the game continued
    # with NPC turns; either way, observation is well-formed
    assert obs2.role == obs.role  # role doesn't change mid-episode


def test_step_with_wrong_actor_id_logs_format_violation():
    """Sending action.player_id != spotlight increments format_violations
    on the spotlight. The episode does NOT raise."""
    env = WerewolfEnvironment()
    env.reset(seed=2)  # spotlight controls seat 0
    pre_violations = env._state.players[0].format_violations
    # Wrong actor id — not the spotlight
    bad_action = WerewolfAction(
        player_id=4,
        action_type="speak",
        content="this is the wrong actor",
    )
    obs = env.step(bad_action)  # should not raise
    assert obs is not None
    assert env._state.players[0].format_violations > pre_violations


def test_state_endpoint_exposes_ground_truth_roles():
    env = WerewolfEnvironment()
    env.reset(seed=2)
    state = env.state
    # All 5 seats present in roles map
    assert set(state.roles.keys()) == {0, 1, 2, 3, 4}
    # Exactly one werewolf and one seer
    role_values = list(state.roles.values())
    assert role_values.count("werewolf") == 1
    assert role_values.count("seer") == 1
    assert role_values.count("villager") == 3


def test_full_episode_terminates():
    """Drive an episode to completion using a simple in-test policy.
    Must terminate with done=True within MAX_DAYS days."""
    from werewolf_env.game.npc import HeuristicNPC

    env = WerewolfEnvironment()
    obs = env.reset(seed=2)
    spot_npc = HeuristicNPC(player_id=0, rng=random.Random(13))

    for _ in range(50):  # safety cap; real episodes are <16 spotlight turns
        if obs.done:
            break
        action = spot_npc.act(env._state)
        obs = env.step(action)
    assert obs.done is True
    assert env._state.day <= 4  # MAX_DAYS


def test_terminal_observation_has_winning_faction():
    env = WerewolfEnvironment()
    obs = env.reset(seed=7)  # spotlight dies night 1 → terminal at reset
    assert obs.done is True
    assert obs.winning_faction in ("werewolf", "villager")
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


# ── Bonus: seed determinism across the three roles ───────────────────────────


@pytest.mark.parametrize(
    "seed,expected_role",
    [(0, "werewolf"), (2, "villager"), (3, "seer")],
)
def test_role_rotation_via_seed(seed, expected_role):
    """Same seed must produce same role assignment (engine determinism)."""
    env = WerewolfEnvironment()
    obs = env.reset(seed=seed)
    assert obs.role == expected_role
