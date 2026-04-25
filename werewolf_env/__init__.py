"""DECEIT — Werewolf/Mafia social-deduction OpenEnv environment.

Public exports follow the canonical OpenEnv pattern: action class,
observation class, and the WerewolfEnv client. The environment server
class is intentionally NOT exported here — it lives in `werewolf_env.server`.
"""

from werewolf_env.client import WerewolfEnv
from werewolf_env.models import (
    PrivateNote,
    PublicEvent,
    WerewolfAction,
    WerewolfObservation,
    WerewolfState,
)

__version__ = "0.1.0"

__all__ = [
    "WerewolfAction",
    "WerewolfObservation",
    "WerewolfState",
    "WerewolfEnv",
    "PublicEvent",
    "PrivateNote",
]
