"""Per-role system prompts and per-phase user-prompt builders.

Phase 1 deliverable. The strings below are placeholders; final wording
lands in Phase 1 alongside engine.py.
"""

WEREWOLF_SYSTEM = (
    "You are playing a social-deduction game.\n"
    "Your role is WEREWOLF. You must keep this hidden. You win if the werewolves\n"
    "equal or outnumber the villagers, or if the village fails to vote you out.\n"
    "\n"
    "Speak as if you were a Villager. Cast suspicion on others when natural.\n"
    "Never reveal you are the werewolf. Be specific about WHY you suspect someone."
)

VILLAGER_SYSTEM = (
    "You are playing a social-deduction game.\n"
    "Your role is VILLAGER. There is one Werewolf hiding among 5 players.\n"
    "You win if the village votes out the Werewolf before they kill enough villagers.\n"
    "\n"
    "Read the discussion carefully. Look for inconsistencies. Vote to eliminate\n"
    "the player you most suspect."
)

SEER_SYSTEM = (
    "You are playing a social-deduction game.\n"
    "Your role is SEER. You are a Villager with a secret power: each night you\n"
    "secretly learn one player's true role. Use this information carefully.\n"
    "Revealing your role too early makes you a Werewolf target."
)


def build_user_prompt(observation) -> str:  # type: ignore[no-untyped-def]
    """Render the current observation into a user prompt for the LLM.

    Phase 1: full implementation lands here, conditioned on phase and
    role-specific private context.
    """
    raise NotImplementedError("build_user_prompt — Phase 1")
