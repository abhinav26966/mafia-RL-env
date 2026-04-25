"""Phase 1 contract tests for the game engine."""

import random

import pytest

from werewolf_env.game.constants import MAX_DAYS, Phase, Role
from werewolf_env.game.engine import GameEngine
from werewolf_env.models import WerewolfAction


def _werewolf_id(state) -> int:
    return next(pid for pid, p in state.players.items() if p.role == Role.WEREWOLF)


def _seer_id(state) -> int:
    return next(pid for pid, p in state.players.items() if p.role == Role.SEER)


def _any_villager_id(state) -> int:
    return next(pid for pid, p in state.players.items() if p.role == Role.VILLAGER)


def test_new_game_has_one_werewolf_and_one_seer(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    roles = [p.role for p in state.players.values()]
    assert roles.count(Role.WEREWOLF) == 1
    assert roles.count(Role.SEER) == 1
    assert roles.count(Role.VILLAGER) == 3
    assert all(p.alive for p in state.players.values())
    assert state.day == 1
    assert state.phase == Phase.NIGHT_WEREWOLF
    # Each player gets a private role-reveal note
    for pid in state.players:
        assert any("role is" in n["note"].lower() for n in state.private_logs[pid])


def test_illegal_actor_id_raises(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    werewolf = _werewolf_id(state)
    wrong_seat = next(pid for pid in state.players if pid != werewolf)
    with pytest.raises(ValueError, match="player_id"):
        engine.apply_action(
            state,
            WerewolfAction(player_id=wrong_seat, action_type="night_kill", target_id=0),
        )


def test_werewolf_kills_advance_to_seer_phase(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    werewolf = _werewolf_id(state)
    target = next(
        pid for pid, p in state.players.items()
        if p.alive and p.role != Role.WEREWOLF
    )
    engine.apply_action(
        state,
        WerewolfAction(player_id=werewolf, action_type="night_kill", target_id=target),
    )
    assert state.phase == Phase.NIGHT_SEER
    assert state.players[target].alive is True  # kill resolves at end of night


def test_vote_tie_breaks_randomly_with_seed():
    """Identical state + game_id + day should produce identical tie-break."""
    engine = GameEngine()

    def setup_tie_state(game_id: str):
        state = engine.new_game(game_id, random.Random(0))
        # Force phase to RESOLUTION-like setup: 2-2 tie between seats 1 and 2
        state.pending_votes = {0: 1, 3: 1, 4: 2, 2: 2}
        return state

    # Two states with same game_id+day should resolve to same eliminated seat
    s1 = setup_tie_state("tie")
    s2 = setup_tie_state("tie")
    engine._resolve_votes(s1)
    engine._resolve_votes(s2)
    e1 = next(pid for pid, p in s1.players.items() if not p.alive)
    e2 = next(pid for pid, p in s2.players.items() if not p.alive)
    assert e1 == e2  # determinism
    assert e1 in (1, 2)  # tie-break stays within tied candidates

    # Different game_id → potentially different eliminated seat (over many runs
    # the choices would diverge; test just confirms same id == same outcome)
    s3 = setup_tie_state("other")
    engine._resolve_votes(s3)
    e3 = next(pid for pid, p in s3.players.items() if not p.alive)
    assert e3 in (1, 2)


def test_werewolf_alone_with_villager_wins(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    werewolf = _werewolf_id(state)
    villager = _any_villager_id(state)
    # Kill everyone except werewolf and one villager → parity → werewolf wins
    for pid, p in state.players.items():
        if pid not in (werewolf, villager):
            p.alive = False
            p.day_died = 1
    assert engine._check_win(state) is True
    assert state.winner == "werewolf"


def test_villagers_vote_out_werewolf_wins(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    werewolf = _werewolf_id(state)
    state.players[werewolf].alive = False
    state.players[werewolf].day_died = 1
    assert engine._check_win(state) is True
    assert state.winner == "villager"


def test_max_days_stalemate_favors_werewolf(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g_stale", seeded_rng)
    state.day = MAX_DAYS
    state.phase = Phase.DAY_VOTE
    # Set up votes that don't eliminate the werewolf — vote out a villager
    werewolf = _werewolf_id(state)
    seer = _seer_id(state)
    villager_a, villager_b, villager_c = (
        pid for pid, p in state.players.items() if p.role == Role.VILLAGER
    )
    # All alive players vote villager_a (not the werewolf)
    state.pending_votes = {
        werewolf: villager_a,
        seer: villager_a,
        villager_b: villager_a,
        villager_c: villager_a,
        villager_a: villager_b,  # everyone alive votes
    }
    engine._maybe_advance(state)
    assert state.phase == Phase.DONE
    # On stalemate at MAX_DAYS the impostor wins
    assert state.winner == "werewolf"


def test_dead_player_cannot_act(seeded_rng):
    """current_actor must skip dead seats in DAY_VOTE."""
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    state.phase = Phase.DAY_VOTE
    state.pending_votes = {}
    # Mark seat 0 dead
    state.players[0].alive = False
    state.players[0].day_died = 1
    # current_actor should never return 0
    actor = engine.current_actor(state)
    assert actor is not None
    assert actor != 0
    assert state.players[actor].alive is True
