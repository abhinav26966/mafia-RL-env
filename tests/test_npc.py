"""Phase 1 contract tests for the NPC pool."""

import random

from werewolf_env.game.constants import Phase, Role
from werewolf_env.game.engine import GameEngine
from werewolf_env.game.npc import (
    ConfusedWerewolf,
    HeuristicNPC,
    LoyalVillager,
    NPCPool,
    ParanoidVillager,
)
from werewolf_env.game.parser import parse_speech, parse_target, parse_vote


def _push_speech(state, *, day, actor_id, text):
    state.public_log.append(
        {
            "day": day,
            "phase": Phase.DAY_DISCUSSION.value,
            "actor_id": actor_id,
            "kind": "speech",
            "text": text,
        }
    )


def test_paranoid_villager_votes_most_mentioned(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    state.phase = Phase.DAY_DISCUSSION
    state.day = 1
    # Heavy mentions of player 4
    _push_speech(state, day=1, actor_id=1, text="Player 4 has been weird and lying")
    _push_speech(state, day=1, actor_id=2, text="I think Player 4 is suspicious")
    _push_speech(state, day=1, actor_id=3, text="Player 4 is acting strange")

    npc = ParanoidVillager(player_id=0, rng=random.Random(0))
    target = npc.suspect_target(state)
    assert target == 4


def test_loyal_villager_follows_trusted_player(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    state.phase = Phase.DAY_DISCUSSION
    state.day = 1
    npc = LoyalVillager(player_id=0, rng=random.Random(7))
    npc._lazy_init(state)
    # Force trusted_player to a known seat for a deterministic test
    npc._trusted_player = 2
    npc._suspicion_bias = 1
    # Trusted player accuses Player 4
    _push_speech(state, day=1, actor_id=2, text="Player 4 looks really guilty to me")
    target = npc.suspect_target(state)
    assert target == 4


def test_confused_werewolf_redirects_accuser(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    state.phase = Phase.DAY_DISCUSSION
    state.day = 1
    # Player 3 accuses our NPC (seat 0)
    _push_speech(state, day=1, actor_id=3, text="Player 0 is the wolf, I am sure")
    npc = ConfusedWerewolf(player_id=0, rng=random.Random(0))
    target = npc.suspect_target(state)
    assert target == 3


def test_npc_output_passes_strict_parser(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g_strict", seeded_rng)
    pool = NPCPool(rng=random.Random(0))
    pool.init_for_game(state, spotlight_id=-1)

    # Drive a full game and parse every NPC output; assert no format violations.
    for _step in range(200):
        if state.done:
            break
        actor = engine.current_actor(state)
        if actor is None:
            break
        action = pool.act(state, actor)

        if action.action_type == "speak":
            prior = [
                e["text"]
                for e in state.public_log
                if e.get("actor_id") == actor and e.get("kind") == "speech"
            ]
            speech = parse_speech(
                action.content or "",
                speaker_id=actor,
                alive_ids=[pid for pid, p in state.players.items() if p.alive],
                prior_speeches=prior,
            )
            assert speech.format_valid, f"NPC speech invalid: {action.content!r}"
            # The speech text also has a [VOTE: N] tail — verify parses
            target, valid = parse_vote(
                action.content or "",
                alive_ids=[pid for pid, p in state.players.items() if p.alive],
                self_id=actor,
            )
            assert valid, f"NPC speech missing valid [VOTE: N]: {action.content!r}"
        elif action.action_type == "vote":
            assert action.target_id is not None
            assert action.target_id != actor
        elif action.action_type in ("night_kill", "seer_check"):
            assert action.target_id is not None
            assert action.target_id != actor
            # parse_target on a synthetic message confirms our brackets format
            tgt, valid = parse_target(
                f"[TARGET: {action.target_id}]",
                valid_ids=[pid for pid, p in state.players.items() if p.alive],
            )
            assert valid

        engine.apply_action(state, action)


def test_npc_pool_assigns_one_persona_per_seat(seeded_rng):
    engine = GameEngine()
    state = engine.new_game("g1", seeded_rng)
    pool = NPCPool(rng=random.Random(0))
    pool.init_for_game(state, spotlight_id=0)

    # Spotlight excluded; the other 4 seats each get one persona instance
    assert set(pool._npcs.keys()) == {1, 2, 3, 4}
    for pid, npc in pool._npcs.items():
        assert isinstance(npc, HeuristicNPC)
        # Werewolf seats get ConfusedWerewolf; non-werewolf seats get a villager persona
        if state.players[pid].role == Role.WEREWOLF:
            assert isinstance(npc, ConfusedWerewolf)
        else:
            assert isinstance(npc, (ParanoidVillager, LoyalVillager))

    # Same seed → same assignments (stability)
    pool2 = NPCPool(rng=random.Random(0))
    pool2.init_for_game(state, spotlight_id=0)
    for pid in (1, 2, 3, 4):
        assert type(pool._npcs[pid]) is type(pool2._npcs[pid])
