#!/usr/bin/env python3
"""Human-vs-NPCs CLI for the Werewolf env.

Plays one full game where you control the spotlight (seat 0). The other 4
seats are deterministic NPCs. Two modes:

    # Direct (no server) — fastest
    python scripts/play_manual.py --seed 2

    # Live server (requires uvicorn running) — exercises the full HTTP path
    python scripts/play_manual.py --seed 2 --server http://localhost:8000

Phase 2 DoD gate.
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap

from werewolf_env.game.parser import parse_target, parse_vote
from werewolf_env.models import WerewolfAction, WerewolfObservation


def _print_observation(obs: WerewolfObservation) -> None:
    print()
    print("─" * 70)
    print(f"Day {obs.day}  │  Phase: {obs.phase}  │  Your role: {obs.role.upper()}")
    print(f"Alive: {obs.alive_player_ids}   Dead: {obs.dead_player_ids}")
    if obs.private_log:
        print()
        print("Private notes (yours only):")
        for note in obs.private_log[-3:]:
            print(f"  • day {note.day}: {note.note}")
    if obs.public_log:
        print()
        print("Recent public log:")
        for evt in obs.public_log[-6:]:
            kind = evt.kind
            if kind == "speech":
                snippet = (evt.text or "").strip()
                if len(snippet) > 80:
                    snippet = snippet[:77] + "…"
                print(f"  • day {evt.day}  Player {evt.actor_id} says: {snippet}")
            elif kind == "vote":
                print(f"  • day {evt.day}  Player {evt.actor_id} → vote Player {evt.target_id}")
            elif kind == "death_announcement":
                print(f"  • day {evt.day}  Player {evt.target_id} eliminated")
    print()


def _prompt_action(obs: WerewolfObservation) -> WerewolfAction:
    legal = obs.legal_action_types
    if "speak" in legal:
        print("Make a speech (free text). End with [VOTE: N] to nominate someone.")
        text = input("> ").strip()
        return WerewolfAction(player_id=0, action_type="speak", content=text)

    rng = random.Random()
    alive = obs.alive_player_ids

    if "vote" in legal:
        print(f"Vote to eliminate. Format: [VOTE: N] where N ∈ {alive} (not {obs.player_id}).")
        text = input("> ").strip()
        target, valid = parse_vote(text, alive_ids=alive, self_id=0, rng=rng)
        if not valid:
            print(f"  (parser fell back: target={target})")
        return WerewolfAction(
            player_id=0,
            action_type="vote",
            target_id=target,
            metadata={"format_violation": True} if not valid else {},
        )

    if "night_kill" in legal:
        valid_targets = [p for p in alive if p != 0]
        print(f"Pick a kill target. Format: [TARGET: N] where N ∈ {valid_targets}.")
        text = input("> ").strip()
        target, valid = parse_target(text, valid_ids=valid_targets, rng=rng)
        if not valid:
            print(f"  (parser fell back: target={target})")
        return WerewolfAction(
            player_id=0,
            action_type="night_kill",
            target_id=target,
            metadata={"format_violation": True} if not valid else {},
        )

    if "seer_check" in legal:
        valid_targets = [p for p in alive if p != 0]
        print(f"Pick a check target. Format: [TARGET: N] where N ∈ {valid_targets}.")
        text = input("> ").strip()
        target, valid = parse_target(text, valid_ids=valid_targets, rng=rng)
        if not valid:
            print(f"  (parser fell back: target={target})")
        return WerewolfAction(
            player_id=0,
            action_type="seer_check",
            target_id=target,
            metadata={"format_violation": True} if not valid else {},
        )

    raise RuntimeError(f"Unexpected legal action types: {legal}")


def _print_terminal(obs: WerewolfObservation, ground_truth_roles: dict | None) -> None:
    print()
    print("═" * 70)
    print(f"Game over. Winner: {obs.winning_faction}")
    print(f"Your reward: {obs.reward:.3f}")
    if ground_truth_roles is not None:
        print(f"Ground-truth roles: {ground_truth_roles}")
    print(f"Final survivors: {obs.alive_player_ids}")
    print("═" * 70)


def _play_direct(seed: int | None) -> None:
    """Run the game in-process — no HTTP."""
    from werewolf_env.server.werewolf_environment import WerewolfEnvironment

    env = WerewolfEnvironment()
    obs = env.reset(seed=seed)
    while not obs.done:
        _print_observation(obs)
        action = _prompt_action(obs)
        obs = env.step(action)
    _print_observation(obs)
    _print_terminal(obs, env.state.roles)


def _play_server(seed: int | None, base_url: str) -> None:
    """Run the game against a live uvicorn server."""
    from werewolf_env.client import WerewolfEnv

    with WerewolfEnv(base_url=base_url).sync() as env:
        result = env.reset(seed=seed)
        obs = result.observation
        while not result.done:
            _print_observation(obs)
            action = _prompt_action(obs)
            result = env.step(action)
            obs = result.observation
        _print_observation(obs)
        state = env.state()
        _print_terminal(obs, dict(state.roles))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--server", default=None, help="Live server URL (else use the env directly)"
    )
    args = parser.parse_args()

    try:
        if args.server:
            _play_server(args.seed, args.server)
        else:
            _play_direct(args.seed)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
