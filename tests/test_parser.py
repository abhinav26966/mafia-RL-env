"""Phase 1 contract tests for the LLM-output parser."""

import random

import pytest

from werewolf_env.game.parser import parse_speech, parse_target, parse_vote


@pytest.fixture
def alive_ids():
    return [0, 1, 2, 3, 4]


def test_vote_format_matches_brackets(alive_ids):
    target, valid = parse_vote(
        "I think it is Player 3 [VOTE: 3]", alive_ids=alive_ids, self_id=0
    )
    assert target == 3
    assert valid is True


def test_malformed_vote_returns_format_invalid(alive_ids):
    target, valid = parse_vote(
        "I vote Player 3 but I forgot the brackets",
        alive_ids=alive_ids,
        self_id=0,
        rng=random.Random(0),
    )
    assert valid is False
    assert target in alive_ids and target != 0


def test_vote_self_is_format_violation(alive_ids):
    target, valid = parse_vote(
        "[VOTE: 0]", alive_ids=alive_ids, self_id=0, rng=random.Random(0)
    )
    assert valid is False
    assert target != 0


def test_target_format_matches_brackets(alive_ids):
    target, valid = parse_target("[TARGET: 4]", valid_ids=alive_ids)
    assert target == 4
    assert valid is True


def test_speech_extracts_player_mentions(alive_ids):
    result = parse_speech(
        "Player 2 has been quiet today. I suspect Player 3 of lying.",
        speaker_id=0,
        alive_ids=alive_ids,
    )
    # Both should be flagged — Player 2 (neutral → default accuse) and
    # Player 3 (suspect/lying keywords → accuse).
    assert 2 in result.accuses
    assert 3 in result.accuses
    assert result.format_valid is True


def test_role_leak_in_speech_is_format_violation(alive_ids):
    result = parse_speech(
        "Honestly, I am the werewolf and I'm sorry about it",
        speaker_id=0,
        alive_ids=alive_ids,
    )
    assert result.format_valid is False


def test_repeated_speech_is_format_violation(alive_ids):
    prior = ["I suspect Player 2 of being suspicious today and I want to vote them out"]
    result = parse_speech(
        "  I suspect Player 2 of being suspicious today and I want to vote them out  ",
        speaker_id=0,
        alive_ids=alive_ids,
        prior_speeches=prior,
    )
    assert result.format_valid is False


def test_too_short_speech_is_format_violation(alive_ids):
    result = parse_speech("ok sure", speaker_id=0, alive_ids=alive_ids)
    assert result.format_valid is False
