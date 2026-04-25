"""Phase 2 contract tests for the WerewolfEnv client.

The two integration tests boot a uvicorn subprocess via the `live_server_url`
fixture in conftest.py. Run with `pytest -m integration` to scope to just
these, or skip them with `pytest -m "not integration"`.
"""

import pytest

from werewolf_env.client import WerewolfEnv
from werewolf_env.models import WerewolfAction


# ── Phase 0 smoke (already passing) ──────────────────────────────────────────


def test_client_imports_cleanly():
    assert WerewolfEnv is not None


def test_client_can_be_instantiated():
    client = WerewolfEnv(base_url="http://localhost:8000")
    assert client is not None


# ── Phase 2 integration tests ────────────────────────────────────────────────


@pytest.mark.integration
def test_client_round_trips_against_local_server(live_server_url):
    """Full reset → step → state round-trip via WebSocket."""
    with WerewolfEnv(base_url=live_server_url).sync() as env:
        result = env.reset(seed=2)
        assert result.observation.player_id == 0
        assert result.observation.role in ("werewolf", "seer", "villager")
        assert result.done is False
        assert "speak" in result.observation.legal_action_types

        # Send a single speech action
        action = WerewolfAction(
            player_id=0,
            action_type="speak",
            content="I have been watching Player 4 closely. They look suspicious to me.",
        )
        result2 = env.step(action)
        assert result2.observation is not None
        # Either we have another spotlight turn, or the game advanced via NPCs
        assert result2.observation.role == result.observation.role


@pytest.mark.integration
def test_state_returns_full_ground_truth(live_server_url):
    """The /state endpoint exposes ground-truth roles for trainer reward
    computation, never via observations."""
    with WerewolfEnv(base_url=live_server_url).sync() as env:
        env.reset(seed=2)
        state = env.state()
        assert state.episode_id is not None
        assert isinstance(state.roles, dict)
        assert set(state.roles.keys()) == {0, 1, 2, 3, 4}
        # Ground-truth role distribution is intact
        roles = list(state.roles.values())
        assert roles.count("werewolf") == 1
        assert roles.count("seer") == 1
        assert roles.count("villager") == 3
