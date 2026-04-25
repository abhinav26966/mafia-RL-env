"""Heuristic NPC players for the seats not played by the trained agent.

A single `HeuristicNPC` class adapts its behavior to the assigned role. Each
NPC at game start picks a random `suspicion_bias` (the player it's
predisposed to accuse) and a random `trusted_player` (whose accusations it
will follow). Discussion-day priorities, in order:

    1. If our trusted_player accused someone today → vote with them.
    2. If WE were accused today → redirect onto the accuser.
    3. Vote the most-mentioned alive non-self player so far today.
    4. Fall back to suspicion_bias.

`ParanoidVillager`, `LoyalVillager`, and `ConfusedWerewolf` are tiny
subclasses that flip priority weights — they exist mainly to give NPCs
some persona variety during training rollouts.

Output text always survives the strict parser (`parse_vote` / `parse_target`
/ `parse_speech`); NPCs never produce format violations.
"""

from __future__ import annotations

import random
import re
from typing import List, Optional

from werewolf_env.game.constants import Phase, Role
from werewolf_env.game.state import GameState
from werewolf_env.models import WerewolfAction


_PLAYER_MENTION_RE = re.compile(r"\bplayer\s+(\d+)\b", re.IGNORECASE)


class HeuristicNPC:
    """Default NPC. Adapts behaviour to its assigned role."""

    # Subclasses override to change priority weights. Default: balanced.
    PRIORITY_TRUSTED: float = 1.0
    PRIORITY_REDIRECT: float = 1.0
    PRIORITY_MENTIONED: float = 1.0

    def __init__(
        self,
        player_id: int,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.player_id = player_id
        self._rng = rng or random.Random()
        self._suspicion_bias: Optional[int] = None
        self._trusted_player: Optional[int] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def act(self, state: GameState) -> WerewolfAction:
        """Produce the next legal action for this seat."""
        self._lazy_init(state)
        my_role = state.players[self.player_id].role
        phase = state.phase

        if phase == Phase.NIGHT_WEREWOLF and my_role == Role.WEREWOLF:
            return self._kill(state)
        if phase == Phase.NIGHT_SEER and my_role == Role.SEER:
            return self._seer_check(state)
        if phase == Phase.DAY_DISCUSSION:
            return self._speak(state)
        if phase == Phase.DAY_VOTE:
            return self._vote(state)
        raise ValueError(
            f"NPC {self.player_id} (role={my_role}) cannot act in phase {phase}"
        )

    # ── Lazy init ────────────────────────────────────────────────────────────

    def _lazy_init(self, state: GameState) -> None:
        """Pick suspicion bias and trusted player on first call (when alive
        seats are known)."""
        if self._suspicion_bias is not None and self._trusted_player is not None:
            return
        candidates = [
            pid for pid in state.players
            if pid != self.player_id and state.players[pid].alive
        ]
        if not candidates:
            self._suspicion_bias = self.player_id
            self._trusted_player = self.player_id
            return
        if self._suspicion_bias is None:
            self._suspicion_bias = self._rng.choice(candidates)
        if self._trusted_player is None:
            # trusted != suspicion_bias when possible
            others = [p for p in candidates if p != self._suspicion_bias]
            self._trusted_player = (
                self._rng.choice(others) if others else self._suspicion_bias
            )

    # ── Action builders ──────────────────────────────────────────────────────

    def _kill(self, state: GameState) -> WerewolfAction:
        cands = [
            pid for pid, p in state.players.items()
            if p.alive and p.role != Role.WEREWOLF
        ]
        target = self._rng.choice(cands) if cands else self.player_id
        return WerewolfAction(
            player_id=self.player_id, action_type="night_kill", target_id=target
        )

    def _seer_check(self, state: GameState) -> WerewolfAction:
        cands = [
            pid for pid, p in state.players.items()
            if p.alive and pid != self.player_id
        ]
        target = self._rng.choice(cands) if cands else self.player_id
        return WerewolfAction(
            player_id=self.player_id, action_type="seer_check", target_id=target
        )

    def _speak(self, state: GameState) -> WerewolfAction:
        target = self._suspect_target(state)
        text = (
            f"I have been watching Player {target} closely. They have been "
            f"acting suspicious to me. Their tone does not add up. We should "
            f"vote them out today. [VOTE: {target}]"
        )
        return WerewolfAction(
            player_id=self.player_id, action_type="speak", content=text
        )

    def _vote(self, state: GameState) -> WerewolfAction:
        target = self._suspect_target(state)
        return WerewolfAction(
            player_id=self.player_id, action_type="vote", target_id=target
        )

    # ── Suspicion logic ──────────────────────────────────────────────────────

    def suspect_target(self, state: GameState) -> int:
        """Public alias for _suspect_target — useful for tests."""
        self._lazy_init(state)
        return self._suspect_target(state)

    def _suspect_target(self, state: GameState) -> int:
        # Priority 1: our trusted player accused someone today
        if self.PRIORITY_TRUSTED > 0:
            tt = self._trust_target(state)
            if tt is not None:
                return tt

        # Priority 2: redirect onto whoever accused us today
        if self.PRIORITY_REDIRECT > 0:
            accusers = self._find_accusers(state)
            if accusers:
                return self._rng.choice(accusers)

        # Priority 3: most-mentioned alive non-self player today
        if self.PRIORITY_MENTIONED > 0:
            best = self._most_mentioned(state)
            if best is not None and best != self.player_id and state.players[best].alive:
                return best

        # Fallback: suspicion bias if alive, else random alive non-self
        if (
            self._suspicion_bias is not None
            and state.players[self._suspicion_bias].alive
        ):
            return self._suspicion_bias
        cands = [
            pid for pid in state.players
            if pid != self.player_id and state.players[pid].alive
        ]
        return self._rng.choice(cands) if cands else self.player_id

    def _trust_target(self, state: GameState) -> Optional[int]:
        """Return the most recent target accused by self._trusted_player today."""
        if self._trusted_player is None:
            return None
        # Walk speeches today by trusted_player from latest backwards
        for evt in reversed(state.public_log):
            if evt.get("day") != state.day or evt.get("kind") != "speech":
                continue
            if evt.get("actor_id") != self._trusted_player:
                continue
            text = evt.get("text", "") or ""
            for m in _PLAYER_MENTION_RE.finditer(text):
                pid = int(m.group(1))
                if (
                    pid != self.player_id
                    and pid in state.players
                    and state.players[pid].alive
                ):
                    return pid
            break  # only inspect the latest speech by trusted player
        return None

    def _find_accusers(self, state: GameState) -> List[int]:
        """Players who mentioned us in a speech today."""
        accusers: list[int] = []
        for evt in state.public_log:
            if evt.get("day") != state.day or evt.get("kind") != "speech":
                continue
            actor = evt.get("actor_id")
            if actor is None or actor == self.player_id:
                continue
            text = (evt.get("text") or "").lower()
            if f"player {self.player_id}" in text:
                if actor not in accusers and state.players[actor].alive:
                    accusers.append(actor)
        return accusers

    def _most_mentioned(self, state: GameState) -> Optional[int]:
        counts: dict[int, int] = {}
        for evt in state.public_log:
            if evt.get("day") != state.day or evt.get("kind") != "speech":
                continue
            text = evt.get("text") or ""
            for m in _PLAYER_MENTION_RE.finditer(text):
                pid = int(m.group(1))
                if pid in state.players:
                    counts[pid] = counts.get(pid, 0) + 1
        if not counts:
            return None
        # Pick the highest count; break ties by lowest player_id for determinism
        best_pid = min(counts, key=lambda p: (-counts[p], p))
        return best_pid


# ── Persona subclasses (priority-weight variations) ──────────────────────────


class ParanoidVillager(HeuristicNPC):
    """Default profile — votes most-mentioned, redirects when accused."""

    PRIORITY_TRUSTED: float = 0.0  # ignores trusted player
    PRIORITY_REDIRECT: float = 1.0
    PRIORITY_MENTIONED: float = 1.0


class LoyalVillager(HeuristicNPC):
    """Follows their trusted player's accusations preferentially."""

    PRIORITY_TRUSTED: float = 1.0
    PRIORITY_REDIRECT: float = 0.0  # doesn't bother defending self
    PRIORITY_MENTIONED: float = 1.0


class ConfusedWerewolf(HeuristicNPC):
    """Aggressively redirects suspicion onto accusers; ignores consensus."""

    PRIORITY_TRUSTED: float = 0.0
    PRIORITY_REDIRECT: float = 1.0
    PRIORITY_MENTIONED: float = 0.0


# ── NPC pool (used by the env to fill non-spotlight seats) ───────────────────


class NPCPool:
    """One NPC per non-spotlight seat. Personas are assigned at game start."""

    PERSONAS: tuple[type, ...] = (ParanoidVillager, LoyalVillager, ConfusedWerewolf)

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()
        self._npcs: dict[int, HeuristicNPC] = {}

    def init_for_game(self, state: GameState, spotlight_id: int) -> None:
        """Seed NPCs for every non-spotlight seat. Werewolf NPCs get the
        ConfusedWerewolf persona; others are randomly Paranoid or Loyal."""
        self._npcs.clear()
        for pid, p in state.players.items():
            if pid == spotlight_id:
                continue
            cls: type
            if p.role == Role.WEREWOLF:
                cls = ConfusedWerewolf
            else:
                cls = self._rng.choice([ParanoidVillager, LoyalVillager])
            seat_rng = random.Random(self._rng.random())
            self._npcs[pid] = cls(player_id=pid, rng=seat_rng)

    def act(self, state: GameState, player_id: int) -> WerewolfAction:
        if player_id not in self._npcs:
            # Fallback: instantiate a default NPC on demand (used by self-play)
            self._npcs[player_id] = HeuristicNPC(
                player_id=player_id, rng=random.Random(self._rng.random())
            )
        return self._npcs[player_id].act(state)
