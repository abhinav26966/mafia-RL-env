"""Phase 4 plumbing tests for the GRPO rollout function.

Validates rollout_func with a deterministic mock `generate_fn` (no GPU,
no Unsloth, no vLLM). Catches integration bugs locally before burning
Colab compute.

Uses the `live_server_url` fixture (subprocess uvicorn) — same pattern
as test_client.py.
"""

import pytest

from training.rollout import (
    DEFAULT_TRAJECTORY_MODE,
    build_action_from_text,
    compute_rubric_rewards,
    rollout_func,
)
from werewolf_env.models import WerewolfState


# ── Pure unit tests (no env / server needed) ─────────────────────────────────


def test_compute_rubric_rewards_outcome_villager_wins():
    state = WerewolfState(
        episode_id="t1",
        step_count=0,
        roles={0: "villager", 1: "werewolf", 2: "villager", 3: "seer", 4: "villager"},
        winner="villager",
    )
    r = compute_rubric_rewards(
        state,
        spotlight_role="villager",
        suspicions=[1],
        format_violations=0,
        spotlight_alive_at_end=True,
        spotlight_day_died=None,
    )
    assert r.outcome == 1.0
    assert r.calibration == 1.0  # accused werewolf, correct
    assert r.survival == 1.0
    assert r.format == 1.0


def test_compute_rubric_rewards_werewolf_loses():
    state = WerewolfState(
        episode_id="t1",
        step_count=0,
        roles={0: "werewolf", 1: "seer", 2: "villager", 3: "villager", 4: "villager"},
        winner="villager",
    )
    r = compute_rubric_rewards(
        state,
        spotlight_role="werewolf",
        suspicions=[1, 2],
        format_violations=2,
        spotlight_alive_at_end=False,
        spotlight_day_died=2,
    )
    assert r.outcome == 0.0
    assert r.calibration == 1.0  # both accused are villager faction (different from werewolf)
    assert r.survival == 0.5  # day_died=2, MAX_DAYS=4
    assert r.format == 0.5  # 1.0 - 0.25 * 2


def test_compute_rubric_rewards_no_suspicions_neutral_calibration():
    state = WerewolfState(
        episode_id="t1",
        step_count=0,
        roles={0: "villager", 1: "werewolf", 2: "villager", 3: "seer", 4: "villager"},
        winner="villager",
    )
    r = compute_rubric_rewards(
        state,
        spotlight_role="villager",
        suspicions=[],
        format_violations=0,
        spotlight_alive_at_end=True,
        spotlight_day_died=None,
    )
    assert r.calibration == 0.5


def test_build_action_from_text_speak():
    from werewolf_env.models import WerewolfObservation

    obs = WerewolfObservation(
        player_id=0,
        role="villager",
        day=1,
        phase="day_discussion",
        alive_player_ids=[0, 1, 2, 3, 4],
        legal_action_types=["speak"],
        current_actor_id=0,
    )
    action, valid, accuses = build_action_from_text(
        "I think Player 4 has been quiet today. They are suspicious to me.", obs
    )
    assert action.action_type == "speak"
    assert action.player_id == 0
    assert valid is True
    assert 4 in accuses


def test_build_action_from_text_vote_with_brackets():
    from werewolf_env.models import WerewolfObservation

    obs = WerewolfObservation(
        player_id=0,
        role="villager",
        day=1,
        phase="day_vote",
        alive_player_ids=[0, 1, 2, 3, 4],
        legal_action_types=["vote"],
        current_actor_id=0,
    )
    action, valid, _ = build_action_from_text("I vote them out [VOTE: 3]", obs)
    assert action.action_type == "vote"
    assert action.target_id == 3
    assert valid is True


def test_build_action_from_text_malformed_vote_falls_back():
    from werewolf_env.models import WerewolfObservation

    obs = WerewolfObservation(
        player_id=0,
        role="villager",
        day=1,
        phase="day_vote",
        alive_player_ids=[0, 1, 2, 3, 4],
        legal_action_types=["vote"],
        current_actor_id=0,
    )
    action, valid, _ = build_action_from_text("no brackets here", obs)
    assert action.action_type == "vote"
    assert valid is False
    assert action.target_id != 0  # not self


# ── Integration: rollout against live local server with mock generation ──────


def _make_phase_aware_mock_generate_fn():
    """Returns a generate_fn that emits well-formatted text for each phase."""

    def mock_generate_fn(trainer, messages_batch):
        results = []
        for messages in messages_batch:
            user_content = ""
            for m in messages:
                if m.get("role") == "user":
                    user_content = m.get("content", "")
                    break
            # Detect phase from instruction tag examples in the prompt
            if "[VOTE:" in user_content and "vote" in user_content.lower():
                text = "Reasoning: Player 3 has been evasive. [VOTE: 3]"
            elif "[TARGET:" in user_content:
                text = "Player 1 looks like a threat. [TARGET: 1]"
            else:  # day_discussion
                text = (
                    "I have been watching Player 4 closely. They have been "
                    "acting suspicious to me and I think we should consider voting them."
                )
            # Fake token ids — we don't need real tokens since the rollout
            # falls back to gen[0]['text'] when there's no tokenizer
            c_ids = list(range(len(text)))
            results.append(
                {
                    "prompt_ids": [101, 102, 103],
                    "completion_ids": c_ids,
                    "logprobs": [-0.1] * len(c_ids),
                    "text": text,
                }
            )
        return results

    return mock_generate_fn


@pytest.mark.integration
def test_rollout_func_full_game_against_local_server(live_server_url):
    """End-to-end rollout: real env (subprocess uvicorn) + mock generation."""
    mock_gen = _make_phase_aware_mock_generate_fn()
    prompts = [{"prompt": "", "seed": s} for s in [2, 3, 0]]  # villager / seer / werewolf

    out = rollout_func(
        prompts,
        trainer=None,  # mock generate_fn handles everything
        env_url=live_server_url,
        trajectory_mode=DEFAULT_TRAJECTORY_MODE,
        generate_fn=mock_gen,
    )

    expected_keys = {
        "prompt_ids",
        "completion_ids",
        "logprobs",
        "outcome_reward",
        "calibration_reward",
        "survival_reward",
        "format_reward",
    }
    assert set(out.keys()) == expected_keys

    # Lengths consistent across all output lists
    n = len(prompts)
    for key in expected_keys:
        assert len(out[key]) == n, f"{key} has wrong length"

    # Rewards bounded in [0, 1]
    for key in ("outcome_reward", "calibration_reward", "survival_reward", "format_reward"):
        for val in out[key]:
            assert 0.0 <= val <= 1.0, f"{key} out of bounds: {val}"

    # At least one game produced model turns (i.e., didn't insta-die at reset)
    assert any(len(c) > 0 for c in out["completion_ids"]), "no model turns recorded"


@pytest.mark.integration
def test_rollout_func_handles_spotlight_death_at_reset(live_server_url):
    """row_seed=4 (with the rollout's seed-mixing) makes the spotlight die
    during reset's NPC fast-forward — rollout must still emit a
    (placeholder) trajectory and rewards."""
    mock_gen = _make_phase_aware_mock_generate_fn()
    prompts = [{"prompt": "", "seed": 4}]

    out = rollout_func(
        prompts,
        trainer=None,
        env_url=live_server_url,
        generate_fn=mock_gen,
    )

    assert len(out["prompt_ids"]) == 1
    # Spotlight died at reset → 0 model turns
    assert out["completion_ids"][0] == [], (
        "spotlight should have insta-died, but got "
        f"{len(out['completion_ids'][0])} tokens of completion"
    )
    # All rewards still computed (game ran to completion via NPC drive-to-end)
    for key in ("outcome_reward", "calibration_reward", "survival_reward", "format_reward"):
        assert 0.0 <= out[key][0] <= 1.0
