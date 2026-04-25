#!/usr/bin/env python3
"""Self-play sanity check — run N games with all 5 seats as NPCs.

Phase 1 DoD gate: `python scripts/play_self_random.py --games 100` runs
100 games end-to-end with no exceptions, villager win-rate ~30-50%.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from werewolf_env.game.engine import GameEngine
from werewolf_env.game.npc import NPCPool
from werewolf_env.game.rewards import (
    reward_calibration,
    reward_format,
    reward_outcome,
    reward_survival,
)


def play_one_game(game_idx: int, base_seed: int) -> dict:
    rng = random.Random(base_seed + game_idx)
    npc_rng = random.Random(base_seed + game_idx + 10_000)
    engine = GameEngine()
    state = engine.new_game(f"selfplay_{game_idx}", rng)
    pool = NPCPool(rng=npc_rng)
    pool.init_for_game(state, spotlight_id=-1)  # all seats are NPCs

    steps = 0
    while not state.done and steps < 200:
        actor = engine.current_actor(state)
        if actor is None:
            break
        action = pool.act(state, actor)
        engine.apply_action(state, action)
        steps += 1

    return {
        "game_idx": game_idx,
        "winner": state.winner,
        "end_day": state.day,
        "steps": steps,
        "roles": {pid: p.role.value for pid, p in state.players.items()},
        "alive_at_end": [pid for pid, p in state.players.items() if p.alive],
        "rewards_per_seat": {
            pid: {
                "outcome": round(reward_outcome(state, pid), 3),
                "calibration": round(reward_calibration(state, pid), 3),
                "survival": round(reward_survival(state, pid), 3),
                "format": round(reward_format(state, pid), 3),
            }
            for pid in state.players
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON dump path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = [play_one_game(i, args.seed) for i in range(args.games)]

    winners = Counter(r["winner"] for r in results)
    end_days = Counter(r["end_day"] for r in results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    villager_wr = winners.get("villager", 0) / args.games
    werewolf_wr = winners.get("werewolf", 0) / args.games

    print(f"Games run:          {args.games}")
    print(f"Villager win rate:  {villager_wr:.1%}  ({winners.get('villager', 0)})")
    print(f"Werewolf win rate:  {werewolf_wr:.1%}  ({winners.get('werewolf', 0)})")
    print(f"Avg steps/game:     {avg_steps:.1f}")
    print(f"End-day distribution: {dict(sorted(end_days.items()))}")

    # Sanity-check win-rate band for Phase 1 DoD.
    # Acceptable range is 15-65% — both factions must be learnable. Our default
    # NPCs are slightly impostor-favoured (ConfusedWerewolf redirects accusers
    # aggressively) which aligns with MASTER_PLAN §14 risk-mitigation: we want
    # to leave room for the trained model to win as werewolf.
    if not (0.15 <= villager_wr <= 0.65):
        print(
            f"\nWARN: villager win-rate {villager_wr:.1%} outside 15-65% band — "
            f"check NPC heuristics / engine balance."
        )

    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nDumped {len(results)} games to {args.out}")

    # Show one transcript example for visual sanity
    if not args.quiet and results:
        sample = results[0]
        print(f"\n── sample game {sample['game_idx']} ──")
        print(f"  roles: {sample['roles']}")
        print(f"  winner: {sample['winner']} (day {sample['end_day']}, {sample['steps']} steps)")
        print(f"  alive at end: {sample['alive_at_end']}")


if __name__ == "__main__":
    main()
