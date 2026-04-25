"""GameEngine — phase transitions, action validation, win checking.

The engine is stateless: every method operates on a `GameState` passed by
reference. The environment (or NPC driver) is responsible for tracking
state across calls.

Phase transition order (per MASTER_PLAN §2):
    NIGHT_WEREWOLF → NIGHT_SEER → DAY_DISCUSSION → DAY_VOTE → RESOLUTION
                       (skip if seer dead)                       │
                                                                 ▼
                              NIGHT_WEREWOLF (next day)  or  DONE

Win conditions:
    - werewolf eliminated                       → "villager"
    - alive werewolves >= alive villagers       → "werewolf"
    - day > MAX_DAYS without resolution         → "werewolf" (favours impostor)
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

from werewolf_env.game.constants import (
    MAX_DAYS,
    NUM_PLAYERS,
    ROLE_DISTRIBUTION,
    ROLE_TO_FACTION,
    Faction,
    Phase,
    Role,
)
from werewolf_env.game.state import GameState, PlayerState
from werewolf_env.models import WerewolfAction


_PLAYER_MENTION_RE = re.compile(r"\bplayer\s+(\d+)\b", re.IGNORECASE)


class GameEngine:
    """Stateless engine — operates on `GameState` by reference."""

    # ── Episode setup ────────────────────────────────────────────────────────

    def new_game(self, game_id: str, rng: random.Random) -> GameState:
        """Initialise a fresh episode. Roles are shuffled deterministically
        from `rng`. All players start alive on day 1, NIGHT_WEREWOLF phase.
        """
        roles = list(ROLE_DISTRIBUTION)
        rng.shuffle(roles)
        players = {i: PlayerState(player_id=i, role=roles[i]) for i in range(NUM_PLAYERS)}
        state = GameState(
            game_id=game_id,
            day=1,
            phase=Phase.NIGHT_WEREWOLF,
            players=players,
            private_logs={i: [] for i in range(NUM_PLAYERS)},
        )
        # Reveal own role to each player as a private note
        for pid, pstate in players.items():
            state.private_logs[pid].append(
                {"day": 1, "note": f"Your role is {pstate.role.value.upper()}."}
            )
        return state

    # ── Read API ─────────────────────────────────────────────────────────────

    def current_actor(self, state: GameState) -> Optional[int]:
        """Seat that must act next, or None if no actor (transition / done)."""
        if state.done:
            return None
        if state.phase == Phase.NIGHT_WEREWOLF:
            for pid, p in state.players.items():
                if p.alive and p.role == Role.WEREWOLF:
                    return pid
            return None
        if state.phase == Phase.NIGHT_SEER:
            for pid, p in state.players.items():
                if p.alive and p.role == Role.SEER:
                    return pid
            return None
        if state.phase == Phase.DAY_DISCUSSION:
            if state.discussion_index >= len(state.discussion_order):
                return None
            return state.discussion_order[state.discussion_index]
        if state.phase == Phase.DAY_VOTE:
            for pid in sorted(state.players):
                if state.players[pid].alive and pid not in state.pending_votes:
                    return pid
            return None
        return None  # RESOLUTION / DONE

    def legal_action_types(self, state: GameState) -> List[str]:
        if state.phase == Phase.NIGHT_WEREWOLF:
            return ["night_kill"]
        if state.phase == Phase.NIGHT_SEER:
            return ["seer_check"]
        if state.phase == Phase.DAY_DISCUSSION:
            return ["speak"]
        if state.phase == Phase.DAY_VOTE:
            return ["vote"]
        return []

    # ── Apply API ────────────────────────────────────────────────────────────

    def apply_action(self, state: GameState, action: WerewolfAction) -> None:
        """Validate `action`, mutate `state`, then auto-advance phase."""
        actor = self.current_actor(state)
        if actor is None:
            raise ValueError(f"No actor expected; phase={state.phase}")
        if action.player_id != actor:
            raise ValueError(
                f"Action player_id {action.player_id} != current_actor {actor}"
            )
        legal = self.legal_action_types(state)
        if action.action_type not in legal:
            raise ValueError(
                f"action_type {action.action_type!r} not in {legal} for phase {state.phase}"
            )

        if action.action_type == "night_kill":
            self._apply_night_kill(state, action)
        elif action.action_type == "seer_check":
            self._apply_seer_check(state, action)
        elif action.action_type == "speak":
            self._apply_speak(state, action)
        elif action.action_type == "vote":
            self._apply_vote(state, action)

        self._maybe_advance(state)

    def apply_default_action(
        self,
        state: GameState,
        player_id: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Apply a sensible default action and increment `format_violations`.

        Used when an LLM produces malformed output the parser couldn't recover.
        Routes to the legal action type for the current phase.
        """
        actor = self.current_actor(state)
        if actor != player_id:
            return  # not their turn — nothing to do
        state.players[player_id].format_violations += 1
        rng = rng or random.Random()
        legal = self.legal_action_types(state)
        action = self._build_default_action(state, player_id, legal, rng)
        if action is not None:
            # Bypass apply_action's actor check (we already validated)
            self._dispatch(state, action)
            self._maybe_advance(state)

    def record_format_violation(self, state: GameState, player_id: int) -> None:
        """Increment a player's violation counter without applying an action.

        Useful when the parser succeeded enough to extract a target, but flagged
        a side-channel violation (e.g., role leak in a speech).
        """
        state.players[player_id].format_violations += 1

    # ── Internal: per-action handlers ────────────────────────────────────────

    def _dispatch(self, state: GameState, action: WerewolfAction) -> None:
        if action.action_type == "night_kill":
            self._apply_night_kill(state, action)
        elif action.action_type == "seer_check":
            self._apply_seer_check(state, action)
        elif action.action_type == "speak":
            self._apply_speak(state, action)
        elif action.action_type == "vote":
            self._apply_vote(state, action)

    def _apply_night_kill(self, state: GameState, action: WerewolfAction) -> None:
        if action.target_id is None:
            raise ValueError("night_kill requires target_id")
        target = action.target_id
        if not state.players[target].alive:
            raise ValueError(f"target {target} already dead")
        if state.players[target].role == Role.WEREWOLF:
            raise ValueError("werewolf cannot self-target")
        state.pending_kill = target

    def _apply_seer_check(self, state: GameState, action: WerewolfAction) -> None:
        if action.target_id is None:
            raise ValueError("seer_check requires target_id")
        target = action.target_id
        if not state.players[target].alive:
            raise ValueError(f"target {target} already dead")
        if target == action.player_id:
            raise ValueError("seer cannot check self")
        target_role = state.players[target].role.value
        state.private_logs[action.player_id].append(
            {
                "day": state.day,
                "note": f"Player {target} is a {target_role.upper()} (seer check).",
            }
        )

    def _apply_speak(self, state: GameState, action: WerewolfAction) -> None:
        text = action.content or ""
        actor = action.player_id
        # Capture suspicions for calibration reward — every Player N mentioned
        # is treated as an accusation (calibration scores against ground truth)
        for match in _PLAYER_MENTION_RE.finditer(text):
            pid = int(match.group(1))
            if (
                pid in state.players
                and pid != actor
                and pid not in state.players[actor].suspicions_stated
            ):
                state.players[actor].suspicions_stated.append(pid)

        state.public_log.append(
            {
                "day": state.day,
                "phase": Phase.DAY_DISCUSSION.value,
                "actor_id": actor,
                "kind": "speech",
                "text": text,
            }
        )
        state.players[actor].speeches_made += 1
        state.discussion_index += 1

    def _apply_vote(self, state: GameState, action: WerewolfAction) -> None:
        if action.target_id is None:
            raise ValueError("vote requires target_id")
        actor = action.player_id
        target = action.target_id

        # Defensive: if target invalid, redirect to a random alive non-self
        # and count a format violation. The parser normally handles this.
        if target == actor or not state.players[target].alive:
            cands = [pid for pid, p in state.players.items() if p.alive and pid != actor]
            target = cands[0] if cands else actor
            state.players[actor].format_violations += 1

        state.pending_votes[actor] = target
        state.players[actor].votes_cast += 1
        state.public_log.append(
            {
                "day": state.day,
                "phase": Phase.DAY_VOTE.value,
                "actor_id": actor,
                "kind": "vote",
                "target_id": target,
            }
        )

    def _build_default_action(
        self,
        state: GameState,
        player_id: int,
        legal: List[str],
        rng: random.Random,
    ) -> Optional[WerewolfAction]:
        if "vote" in legal:
            cands = [pid for pid, p in state.players.items() if p.alive and pid != player_id]
            target = rng.choice(cands) if cands else player_id
            return WerewolfAction(
                player_id=player_id, action_type="vote", target_id=target
            )
        if "speak" in legal:
            return WerewolfAction(
                player_id=player_id, action_type="speak", content="(no comment)"
            )
        if "night_kill" in legal:
            cands = [
                pid for pid, p in state.players.items()
                if p.alive and p.role != Role.WEREWOLF
            ]
            target = rng.choice(cands) if cands else player_id
            return WerewolfAction(
                player_id=player_id, action_type="night_kill", target_id=target
            )
        if "seer_check" in legal:
            cands = [pid for pid, p in state.players.items() if p.alive and pid != player_id]
            target = rng.choice(cands) if cands else player_id
            return WerewolfAction(
                player_id=player_id, action_type="seer_check", target_id=target
            )
        return None

    # ── Phase transitions ────────────────────────────────────────────────────

    def _maybe_advance(self, state: GameState) -> None:
        """Run phase transitions until we land on a phase that needs an actor
        or the game is over.
        """
        # We loop because some transitions chain (e.g., NIGHT_SEER skip if seer
        # dead, then resolve kill, then maybe end game).
        for _ in range(8):  # safety cap; phases-per-day is bounded
            advanced = self._advance_once(state)
            if not advanced:
                return

    def _advance_once(self, state: GameState) -> bool:
        if state.phase == Phase.NIGHT_WEREWOLF:
            if state.pending_kill is None:
                return False  # waiting for werewolf action
            self._transition_to(state, Phase.NIGHT_SEER)
            return True

        if state.phase == Phase.NIGHT_SEER:
            seer_alive = any(p.alive and p.role == Role.SEER for p in state.players.values())
            if seer_alive and not self._seer_acted_today(state):
                return False  # waiting for seer action
            # Either seer is dead OR seer just acted: resolve night kill, then move on
            self._resolve_kill(state)
            if self._check_win(state):
                state.phase = Phase.DONE
                return True
            self._transition_to(state, Phase.DAY_DISCUSSION)
            return True

        if state.phase == Phase.DAY_DISCUSSION:
            if state.discussion_index < len(state.discussion_order):
                return False  # waiting on next speaker
            self._transition_to(state, Phase.DAY_VOTE)
            return True

        if state.phase == Phase.DAY_VOTE:
            alive_count = sum(1 for p in state.players.values() if p.alive)
            if len(state.pending_votes) < alive_count:
                return False  # waiting on more votes
            self._resolve_votes(state)
            if self._check_win(state):
                state.phase = Phase.DONE
                return True
            if state.day >= MAX_DAYS:
                state.winner = "werewolf"
                state.phase = Phase.DONE
                return True
            state.day += 1
            self._transition_to(state, Phase.NIGHT_WEREWOLF)
            return True

        return False

    def _seer_acted_today(self, state: GameState) -> bool:
        """True if the seer has logged a check this day."""
        seer_id: Optional[int] = next(
            (pid for pid, p in state.players.items() if p.role == Role.SEER), None
        )
        if seer_id is None:
            return True  # no seer in game (shouldn't happen) — treat as acted
        notes = state.private_logs.get(seer_id, [])
        return any(
            n.get("day") == state.day and "seer check" in n.get("note", "").lower()
            for n in notes
        )

    def _transition_to(self, state: GameState, new_phase: Phase) -> None:
        state.phase = new_phase
        if new_phase == Phase.DAY_DISCUSSION:
            alive = sorted(pid for pid, p in state.players.items() if p.alive)
            state.discussion_order = alive
            state.discussion_index = 0
        elif new_phase == Phase.DAY_VOTE:
            state.pending_votes = {}
        elif new_phase == Phase.NIGHT_WEREWOLF:
            state.pending_kill = None

    # ── Resolution helpers ───────────────────────────────────────────────────

    def _resolve_kill(self, state: GameState) -> None:
        if state.pending_kill is not None:
            target = state.pending_kill
            if state.players[target].alive:
                state.players[target].alive = False
                state.players[target].day_died = state.day
                state.public_log.append(
                    {
                        "day": state.day,
                        "phase": Phase.RESOLUTION.value,
                        "actor_id": target,
                        "kind": "death_announcement",
                        "target_id": target,
                    }
                )
            state.pending_kill = None

    def _resolve_votes(self, state: GameState) -> None:
        if not state.pending_votes:
            return
        counts: dict[int, int] = {}
        for target in state.pending_votes.values():
            counts[target] = counts.get(target, 0) + 1
        max_votes = max(counts.values())
        candidates = sorted(pid for pid, c in counts.items() if c == max_votes)
        # Tie-break deterministically using a state-derived seed.
        rng = random.Random(f"{state.game_id}:{state.day}:tiebreak")
        eliminated = rng.choice(candidates)
        state.players[eliminated].alive = False
        state.players[eliminated].day_died = state.day
        state.public_log.append(
            {
                "day": state.day,
                "phase": Phase.RESOLUTION.value,
                "actor_id": eliminated,
                "kind": "death_announcement",
                "target_id": eliminated,
            }
        )
        state.pending_votes = {}

    def _check_win(self, state: GameState) -> bool:
        alive = [p for p in state.players.values() if p.alive]
        wolves = [p for p in alive if p.role == Role.WEREWOLF]
        villagers = [p for p in alive if ROLE_TO_FACTION[p.role] == Faction.VILLAGER]
        if not wolves:
            state.winner = "villager"
            return True
        if len(wolves) >= len(villagers):
            state.winner = "werewolf"
            return True
        return False
