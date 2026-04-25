"""Per-role system prompts for training rollouts.

Phase 4 deliverable. Stub re-exports the strings from werewolf_env.game.prompts
so the trainer has a single import path.
"""

from werewolf_env.game.prompts import SEER_SYSTEM, VILLAGER_SYSTEM, WEREWOLF_SYSTEM

ROLE_TO_SYSTEM_PROMPT = {
    "werewolf": WEREWOLF_SYSTEM,
    "villager": VILLAGER_SYSTEM,
    "seer": SEER_SYSTEM,
}


def system_prompt_for_role(role: str) -> str:
    """Return the system prompt for the given role string."""
    if role not in ROLE_TO_SYSTEM_PROMPT:
        raise ValueError(f"Unknown role: {role!r}")
    return ROLE_TO_SYSTEM_PROMPT[role]
