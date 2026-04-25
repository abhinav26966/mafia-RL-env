"""GameState dataclasses. Mutated by `engine.py`.

Phase 1: full state per MASTER_PLAN §6 state.py.
Phase 0: minimal stubs to keep imports clean.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from werewolf_env.game.constants import Phase, Role


@dataclass
class PlayerState:
    player_id: int
    role: Role
    alive: bool = True
    speeches_made: int = 0
    votes_cast: int = 0
    suspicions_stated: List[int] = field(default_factory=list)
    """Player IDs this player has accused — used for calibration reward."""
    format_violations: int = 0
    """Count of format/anti-hacking violations attributable to this player.
    Read by `reward_format`. Incremented by the engine on parser-flagged
    or default-action invocations."""
    day_died: Optional[int] = None
    """Day the player was eliminated, or None if alive at game end.
    Read by `reward_survival` to compute fraction of days survived."""


@dataclass
class GameState:
    game_id: str
    day: int = 1
    phase: Phase = Phase.NIGHT_WEREWOLF
    players: Dict[int, PlayerState] = field(default_factory=dict)
    public_log: List[dict] = field(default_factory=list)
    private_logs: Dict[int, List[dict]] = field(default_factory=dict)
    discussion_order: List[int] = field(default_factory=list)
    discussion_index: int = 0
    pending_votes: Dict[int, int] = field(default_factory=dict)
    pending_kill: Optional[int] = None
    winner: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE
