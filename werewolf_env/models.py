"""Wire-level Pydantic models for the Werewolf OpenEnv environment.

These are the action / observation / state shapes that flow over the
WebSocket between the trainer and the FastAPI server. They MUST inherit
from `openenv.core.env_server.types.{Action, Observation, State}`.

Phase 0: minimal-but-valid stubs so `pip install -e .` and `import` work.
Phase 1: full schema per MASTER_PLAN.md §5 lands here.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# ── Public events (visible in observation.public_log) ─────────────────────────


class PublicEvent(BaseModel):
    """One public game event — visible to all players."""

    day: int = Field(..., ge=0)
    phase: str
    actor_id: int = Field(..., ge=0, le=4)
    kind: Literal["speech", "vote", "death_announcement"]
    text: Optional[str] = None
    target_id: Optional[int] = Field(None, ge=0, le=4)


class PrivateNote(BaseModel):
    """One private observation — visible only to a single player.

    Examples: own role assignment, seer-check result, werewolf kill list.
    """

    day: int = Field(..., ge=0)
    note: str


# ── Action ───────────────────────────────────────────────────────────────────


class WerewolfAction(Action):
    """A single move by a single seat.

    Set `metadata['format_violation'] = True` when the trainer's parser fell
    back to a default target — the env will increment format_violations on
    the spotlight.
    """

    player_id: int = Field(..., ge=0, le=4, description="Seat 0-4")
    action_type: Literal["speak", "vote", "night_kill", "seer_check"]
    content: Optional[str] = Field(
        default=None,
        max_length=600,
        description="Speech text — only used when action_type='speak'",
    )
    target_id: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description="Target seat for vote / night_kill / seer_check",
    )


# ── Observation ──────────────────────────────────────────────────────────────


class WerewolfObservation(Observation):
    """What ONE player sees right now, plus turn metadata.

    Inherits `done`, `reward`, `metadata` from the OpenEnv base.
    """

    player_id: int = Field(..., ge=0, le=4)
    role: str = Field(..., description="Own role; only the requesting player's own role is revealed")

    day: int = Field(default=0, ge=0)
    phase: str = Field(default="night_werewolf")

    alive_player_ids: List[int] = Field(default_factory=list)
    dead_player_ids: List[int] = Field(default_factory=list)

    public_log: List[PublicEvent] = Field(default_factory=list)
    private_log: List[PrivateNote] = Field(default_factory=list)

    current_actor_id: Optional[int] = Field(
        default=None,
        description="Seat of the player whose turn it is, or None if game over",
    )
    legal_action_types: List[str] = Field(default_factory=list)

    winning_faction: Optional[str] = Field(
        default=None, description="'werewolf' | 'villager' | None — populated only on terminal step"
    )


# ── State (server-side, exposed via /state for trainers/debug) ───────────────


class WerewolfState(State):
    """Episode metadata + ground-truth roles. Used by the trainer to compute
    calibration rewards after the episode. NEVER leaked into observations.
    """

    game_id: Optional[str] = Field(default=None)
    day: int = Field(default=0, ge=0)
    phase: str = Field(default="none")
    alive_player_ids: List[int] = Field(default_factory=list)
    dead_player_ids: List[int] = Field(default_factory=list)
    roles: Dict[int, str] = Field(
        default_factory=dict,
        description="Ground-truth {player_id: role}. Server-only — exposed via /state, never via /step or /reset.",
    )
    history_length: int = Field(default=0, ge=0)
    winner: Optional[str] = Field(default=None)
