"""Roles, phases, factions, and game-config constants.

Phase 1: full enums + role distribution (per MASTER_PLAN §6 constants.py).
"""

from enum import Enum


class Role(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    VILLAGER = "villager"


class Phase(str, Enum):
    NIGHT_WEREWOLF = "night_werewolf"
    NIGHT_SEER = "night_seer"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    RESOLUTION = "resolution"
    DONE = "done"


class Faction(str, Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"


ROLE_TO_FACTION: dict[Role, Faction] = {
    Role.WEREWOLF: Faction.WEREWOLF,
    Role.SEER: Faction.VILLAGER,
    Role.VILLAGER: Faction.VILLAGER,
}

NUM_PLAYERS: int = 5
ROLE_DISTRIBUTION: list[Role] = [
    Role.WEREWOLF,
    Role.SEER,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
]
MAX_DAYS: int = 4
MAX_SPEECH_TOKENS: int = 80
