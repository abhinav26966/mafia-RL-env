"""Per-role system prompts and per-phase user-prompt builder.

The user prompt is assembled from:
  - role + day + phase header
  - private notes (only this player's)
  - recent public log
  - phase-specific instruction telling the LLM exactly what format to emit

The format expected by the parser is strict, so prompts include explicit
examples (e.g., `[VOTE: 3]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:  # avoid runtime circular imports
    from werewolf_env.models import WerewolfObservation


# ── System prompts (per role) ─────────────────────────────────────────────────


WEREWOLF_SYSTEM = (
    "You are playing a 5-player social-deduction game called Werewolf.\n"
    "Your role is WEREWOLF (impostor). One Seer and three Villagers are\n"
    "trying to vote you out. You win if alive werewolves equal alive\n"
    "villagers, or if the village fails to vote you out within 4 days.\n"
    "\n"
    "Speak as if you were a Villager. Cast suspicion on others when natural.\n"
    "Be specific about WHY you suspect someone — reference earlier speeches\n"
    "or behavior. Never reveal you are the werewolf. Never use the word\n"
    "'werewolf' to describe yourself."
)


VILLAGER_SYSTEM = (
    "You are playing a 5-player social-deduction game called Werewolf.\n"
    "Your role is VILLAGER. There is one Werewolf hiding among 4 other\n"
    "players. You win if the village votes the Werewolf out before they\n"
    "kill enough villagers. You have no special powers — read carefully\n"
    "and look for inconsistencies in others' speeches.\n"
    "\n"
    "When you suspect someone, reference Player N specifically and explain\n"
    "WHY. Be precise — vague accusations get ignored."
)


SEER_SYSTEM = (
    "You are playing a 5-player social-deduction game called Werewolf.\n"
    "Your role is SEER. You are a Villager with a secret power: each\n"
    "night you privately learn one player's true role. Your team wins\n"
    "if the Werewolf is voted out.\n"
    "\n"
    "Use your knowledge carefully — revealing your role too early makes\n"
    "you the Werewolf's next kill target. Reference Player N specifically\n"
    "when you accuse, but you don't always have to say HOW you know."
)


def system_prompt_for_role(role: str) -> str:
    """Return the system prompt for the given role string."""
    role = role.lower()
    if role == "werewolf":
        return WEREWOLF_SYSTEM
    if role == "seer":
        return SEER_SYSTEM
    if role == "villager":
        return VILLAGER_SYSTEM
    raise ValueError(f"Unknown role: {role!r}")


# ── User prompt builder ──────────────────────────────────────────────────────


_PUBLIC_LOG_WINDOW = 12
"""Recent public events to include verbatim. Older events are summarised."""


def _format_public_log(events: List, window: int = _PUBLIC_LOG_WINDOW) -> str:
    if not events:
        return "(none yet)"
    recent = events[-window:]
    lines = []
    for evt in recent:
        kind = evt.kind
        day = evt.day
        actor = evt.actor_id
        if kind == "speech":
            text = (evt.text or "").strip()
            # Trim very long speeches to keep prompt small
            if len(text) > 240:
                text = text[:237] + "…"
            lines.append(f"  Day {day} — Player {actor} says: {text}")
        elif kind == "vote":
            lines.append(f"  Day {day} — Player {actor} → vote Player {evt.target_id}")
        elif kind == "death_announcement":
            lines.append(f"  Day {day} — Player {evt.target_id} was eliminated")
        else:  # pragma: no cover
            lines.append(f"  Day {day} — Player {actor} {kind}")
    return "\n".join(lines)


def _format_private_log(notes: List) -> str:
    if not notes:
        return ""
    lines = [f"  Day {n.day}: {n.note}" for n in notes]
    return "\n".join(lines)


def _instruction_for_phase(obs: "WerewolfObservation") -> str:
    phase = obs.phase
    pid = obs.player_id
    others_alive = [p for p in obs.alive_player_ids if p != pid]

    if phase == "day_discussion":
        return (
            f"It is Day {obs.day} discussion. Make a SHORT speech (5-30 words)\n"
            f"explaining your suspicions or defending yourself. Reference\n"
            f"specific players by 'Player N' (where N is one of {others_alive})\n"
            f"so your accusations are clear. Do NOT reveal your role."
        )

    if phase == "day_vote":
        return (
            f"It is Day {obs.day} vote. You must vote to eliminate one alive\n"
            f"player. Briefly justify (one sentence) then end your message\n"
            f"with the EXACT bracket tag:\n"
            f"\n"
            f"    [VOTE: N]\n"
            f"\n"
            f"where N is one of {others_alive} (NEVER yourself)."
        )

    if phase == "night_werewolf":
        # Werewolf cannot self-target
        kill_candidates = [p for p in obs.alive_player_ids if p != pid]
        return (
            f"It is Night {obs.day}. You are the WEREWOLF. Pick one alive\n"
            f"player to kill. End your message with the EXACT bracket tag:\n"
            f"\n"
            f"    [TARGET: N]\n"
            f"\n"
            f"where N is one of {kill_candidates}. You may write a brief\n"
            f"justification first, but only the bracketed integer is parsed."
        )

    if phase == "night_seer":
        return (
            f"It is Night {obs.day}. You are the SEER. Pick one alive player\n"
            f"to investigate. Their true role will be revealed to you privately.\n"
            f"End your message with the EXACT bracket tag:\n"
            f"\n"
            f"    [TARGET: N]\n"
            f"\n"
            f"where N is one of {others_alive}. Choose someone you suspect."
        )

    return f"[unknown phase: {phase}]"


def build_user_prompt(obs: "WerewolfObservation") -> str:
    """Render an observation into a user-prompt string for the LLM."""
    pid = obs.player_id
    body_lines = [
        f"You are Player {pid}. Your role: {obs.role.upper()}.",
        f"Day {obs.day} — Phase: {obs.phase}",
        f"Alive players: {obs.alive_player_ids}",
    ]
    if obs.dead_player_ids:
        body_lines.append(f"Dead players: {obs.dead_player_ids}")

    private = _format_private_log(obs.private_log)
    if private:
        body_lines.append("")
        body_lines.append("Your private notes (only you see these):")
        body_lines.append(private)

    body_lines.append("")
    body_lines.append("Recent public events:")
    body_lines.append(_format_public_log(obs.public_log))

    body_lines.append("")
    body_lines.append(_instruction_for_phase(obs))

    return "\n".join(body_lines)
