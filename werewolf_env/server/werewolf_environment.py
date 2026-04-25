"""WerewolfEnvironment — OpenEnv `Environment` subclass.

One episode = one full Werewolf game from the perspective of a "spotlight"
seat. The trainer's model controls seat 0 only; the other 4 seats are filled
by deterministic NPCs (`NPCPool`).

The spotlight role rotates per episode (the engine shuffles roles in
`new_game`), so seat 0 is sometimes Werewolf, sometimes Seer, sometimes
Villager. This is how a single trained model learns all three roles.

Format-violation contract:
    The trainer's rollout parses LLM text into a structured `WerewolfAction`.
    When parsing fell back to a default (e.g., missing `[VOTE: N]`), the
    trainer sets `action.metadata["format_violation"] = True`. The env logs
    the violation against the spotlight via `engine.record_format_violation`.

Death-of-spotlight contract:
    If the spotlight dies before a natural game end, the env keeps running
    NPC turns until the game resolves, then returns done=True with the
    correct `winning_faction` so the trainer can compute outcome reward.
"""

from __future__ import annotations

import random
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from werewolf_env.game.engine import GameEngine
    from werewolf_env.game.npc import NPCPool
    from werewolf_env.game.rewards import (
        reward_calibration,
        reward_format,
        reward_outcome,
        reward_survival,
    )
    from werewolf_env.game.state import GameState
    from werewolf_env.models import (
        PrivateNote,
        PublicEvent,
        WerewolfAction,
        WerewolfObservation,
        WerewolfState,
    )
except ModuleNotFoundError:  # pragma: no cover  — HF Space container path
    from game.engine import GameEngine  # type: ignore[import-not-found,no-redef]
    from game.npc import NPCPool  # type: ignore[import-not-found,no-redef]
    from game.rewards import (  # type: ignore[import-not-found,no-redef]
        reward_calibration,
        reward_format,
        reward_outcome,
        reward_survival,
    )
    from game.state import GameState  # type: ignore[import-not-found,no-redef]
    from models import (  # type: ignore[import-not-found,no-redef]
        PrivateNote,
        PublicEvent,
        WerewolfAction,
        WerewolfObservation,
        WerewolfState,
    )


# Reward composition weights — must match training/reward_funcs.py
_W_OUTCOME, _W_CALIBRATION, _W_SURVIVAL, _W_FORMAT = 0.50, 0.25, 0.15, 0.10

# Hard cap on NPC fast-forward iterations — guards against infinite loops
# from any logic bug in the engine. Realistic upper bound:
# 4 days × (1 night kill + 1 seer check + 5 speeches + 5 votes) = 48 actions.
_FAST_FORWARD_CAP = 200


class WerewolfEnvironment(Environment):
    """One episode = one Werewolf game from seat 0's POV."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    SPOTLIGHT_ID: int = 0  # the seat the trainer's model always controls

    def __init__(self) -> None:
        super().__init__()
        self._engine = GameEngine()
        self._state: Optional[GameState] = None
        self._npc_pool: Optional[NPCPool] = None
        self._rng: Optional[random.Random] = None
        self._episode_id: str = ""

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> WerewolfObservation:  # type: ignore[override]
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        self._rng = random.Random(seed)
        self._episode_id = str(uuid4())

        # Engine RNG and NPC RNG are derived from the seed so the whole
        # episode is reproducible from a single integer.
        engine_rng = random.Random(self._rng.randint(0, 2**31 - 1))
        npc_rng = random.Random(self._rng.randint(0, 2**31 - 1))

        self._state = self._engine.new_game(self._episode_id, engine_rng)
        self._npc_pool = NPCPool(rng=npc_rng)
        self._npc_pool.init_for_game(self._state, spotlight_id=self.SPOTLIGHT_ID)

        # Drive NPCs to the spotlight's first turn (or end of game)
        self._fast_forward_to_spotlight()

        return self._build_observation()

    def step(self, action: WerewolfAction) -> WerewolfObservation:  # type: ignore[override]
        assert self._state is not None and self._npc_pool is not None, "step before reset"

        # Trainer-flagged format violation (parser fell back to a default).
        # Even when format_violation is set, we still APPLY the action
        # because the parser already substituted a legal target — the
        # game must continue forward.
        format_flagged = bool(action.metadata.get("format_violation"))
        if format_flagged:
            self._engine.record_format_violation(self._state, self.SPOTLIGHT_ID)

        # Wrong actor — log a violation and let NPC fast-forward fix the turn
        if action.player_id != self.SPOTLIGHT_ID:
            self._engine.record_format_violation(self._state, self.SPOTLIGHT_ID)
        else:
            try:
                self._engine.apply_action(self._state, action)
            except Exception:
                # Engine rejected (illegal target, dead target, etc).
                # Log a violation and fall back to a default action.
                self._engine.apply_default_action(
                    self._state,
                    self.SPOTLIGHT_ID,
                    rng=self._rng,
                )

        self._fast_forward_to_spotlight()

        # If spotlight died but game not yet resolved, drive remaining NPC
        # turns until the game ends. This gives the trainer a clean
        # done=True signal with `winning_faction` populated.
        if not self._spotlight_alive() and not self._state.done:
            self._drive_to_end()

        return self._build_observation()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="DECEIT — Werewolf",
            description=(
                "5-player hidden-role social-deduction environment. The trainer's "
                "model controls seat 0 (the 'spotlight'); the other 4 seats are "
                "deterministic NPC bots. The spotlight role rotates each episode "
                "(werewolf / seer / villager) so a single trained model learns "
                "all three roles. Theme #1: Multi-Agent Interactions."
            ),
            version="0.1.0",
        )

    @property
    def state(self) -> WerewolfState:  # type: ignore[override]
        if self._state is None:
            return WerewolfState(
                episode_id="none",
                step_count=0,
                game_id="none",
            )
        s = self._state
        return WerewolfState(
            episode_id=self._episode_id,
            step_count=len(s.public_log),
            game_id=s.game_id,
            day=s.day,
            phase=s.phase.value,
            alive_player_ids=[pid for pid, p in s.players.items() if p.alive],
            dead_player_ids=[pid for pid, p in s.players.items() if not p.alive],
            roles={pid: p.role.value for pid, p in s.players.items()},
            history_length=len(s.public_log),
            winner=s.winner,
        )

    # ── Internal: fast-forward + observation building ───────────────────────

    def _fast_forward_to_spotlight(self) -> None:
        """Drive NPC actions until it's the spotlight's turn or game ends."""
        assert self._state is not None and self._npc_pool is not None
        for _ in range(_FAST_FORWARD_CAP):
            if self._state.done:
                return
            actor = self._engine.current_actor(self._state)
            if actor is None:
                return  # phase transition already happened inside apply_action
            if actor == self.SPOTLIGHT_ID:
                return
            action = self._npc_pool.act(self._state, actor)
            self._engine.apply_action(self._state, action)
        raise RuntimeError(
            f"Fast-forward exceeded {_FAST_FORWARD_CAP} steps — engine bug?"
        )

    def _drive_to_end(self) -> None:
        """Run remaining NPC turns to natural game end after spotlight dies."""
        assert self._state is not None and self._npc_pool is not None
        for _ in range(_FAST_FORWARD_CAP):
            if self._state.done:
                return
            actor = self._engine.current_actor(self._state)
            if actor is None:
                return
            # Even if "actor" is the dead spotlight, the engine's current_actor
            # only returns alive seats for vote/speech phases. For NIGHT
            # phases the actor is the alive werewolf or seer — both NPCs here
            # since spotlight is dead.
            action = self._npc_pool.act(self._state, actor)
            self._engine.apply_action(self._state, action)

    def _spotlight_alive(self) -> bool:
        assert self._state is not None
        return self._state.players[self.SPOTLIGHT_ID].alive

    def _build_observation(self) -> WerewolfObservation:
        s = self._state
        assert s is not None
        spot = s.players[self.SPOTLIGHT_ID]

        # Public log is visible to everyone; private log is just the spotlight's
        public = [PublicEvent(**e) for e in s.public_log]
        private = [PrivateNote(**n) for n in s.private_logs.get(self.SPOTLIGHT_ID, [])]

        done = s.done
        # When the spotlight is dead but the engine hasn't resolved yet, we
        # still mark done=True for the trainer's purposes (drive_to_end above
        # ensures the game finishes before returning).
        if not spot.alive and not done:
            done = True  # defensive — drive_to_end should have set this

        if done and not s.done:
            # Spotlight died but game continues — should not happen post-drive
            actor = None
            legal: list = []
        else:
            actor = self._engine.current_actor(s) if not done else None
            legal = self._engine.legal_action_types(s) if not done else []

        reward: Optional[float] = None
        if done:
            reward = self._compute_combined_reward()

        return WerewolfObservation(
            player_id=self.SPOTLIGHT_ID,
            role=spot.role.value,
            day=s.day,
            phase=s.phase.value,
            alive_player_ids=[pid for pid, p in s.players.items() if p.alive],
            dead_player_ids=[pid for pid, p in s.players.items() if not p.alive],
            public_log=public,
            private_log=private,
            current_actor_id=actor,
            legal_action_types=legal,
            winning_faction=s.winner,
            done=done,
            reward=reward,
        )

    def _compute_combined_reward(self) -> float:
        s = self._state
        assert s is not None
        pid = self.SPOTLIGHT_ID
        return (
            _W_OUTCOME * reward_outcome(s, pid)
            + _W_CALIBRATION * reward_calibration(s, pid)
            + _W_SURVIVAL * reward_survival(s, pid)
            + _W_FORMAT * reward_format(s, pid)
        )
