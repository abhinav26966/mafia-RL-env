"""Parse free-form LLM output into structured actions.

Strict bracket format:
    Vote:   message must contain `[VOTE: N]`.
    Night:  message must contain `[TARGET: N]` (kill or check).
    Speech: free text; we extract `Player N` mentions and detect
            format violations (role leak, length floor, repeat).

All parsers return `(value, format_valid)` tuples where format_valid=False
signals that a default fallback was applied so the game can continue.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Regex constants ────────────────────────────────────────────────────────────

_VOTE_RE = re.compile(r"\[VOTE:\s*(\d+)\s*\]", re.IGNORECASE)
_TARGET_RE = re.compile(r"\[TARGET:\s*(\d+)\s*\]", re.IGNORECASE)
_PLAYER_MENTION_RE = re.compile(r"\bplayer\s+(\d+)\b", re.IGNORECASE)

# Substrings that count as a role leak. Substring match is case-insensitive.
_ROLE_LEAK_PHRASES: tuple[str, ...] = (
    "i am the werewolf",
    "i am a werewolf",
    "i'm the werewolf",
    "i'm a werewolf",
    "i am the seer",
    "i'm the seer",
)

# Speech length floor. Below this many tokens (whitespace-split) is a violation.
_MIN_SPEECH_TOKENS: int = 5

# Words that indicate negative/suspicious sentiment toward a mentioned player.
_ACCUSE_KEYWORDS: tuple[str, ...] = (
    "suspect", "wolf", "lying", "guilty", "kill", "strange", "weird",
    "odd", "off", "vote", "deceiv", "shifty", "nervous", "evasive",
)
# Words that indicate positive/defending sentiment.
_DEFEND_KEYWORDS: tuple[str, ...] = (
    "trust", "innocent", "good", "clean", "honest", "with me", "agree",
)


@dataclass
class ParsedSpeech:
    """Result of parsing a player's speech into structured fields."""

    text: str
    accuses: List[int] = field(default_factory=list)
    defends: List[int] = field(default_factory=list)
    claims_role: Optional[str] = None
    format_valid: bool = True


# ── Parsers ────────────────────────────────────────────────────────────────────


def parse_vote(
    text: str,
    alive_ids: List[int],
    self_id: int,
    rng: Optional[random.Random] = None,
) -> Tuple[int, bool]:
    """Parse `[VOTE: N]` from `text`. Returns `(target_id, format_valid)`.

    Format is invalid when the bracket pattern is missing, the target is
    not currently alive, or the target equals `self_id` (vote-self penalty).
    A random alive non-self target is returned as fallback.
    """
    rng = rng or random.Random()
    match = _VOTE_RE.search(text or "")
    if match is None:
        return _default_vote_target(alive_ids, self_id, rng), False
    target = int(match.group(1))
    if target not in alive_ids or target == self_id:
        return _default_vote_target(alive_ids, self_id, rng), False
    return target, True


def parse_target(
    text: str,
    valid_ids: List[int],
    rng: Optional[random.Random] = None,
) -> Tuple[int, bool]:
    """Parse `[TARGET: N]` from `text` for night actions.

    `valid_ids` is the caller's pre-filtered list of legal targets (e.g.,
    alive non-self for werewolf kill or seer check).
    """
    rng = rng or random.Random()
    match = _TARGET_RE.search(text or "")
    if match is None:
        return _default_target(valid_ids, rng), False
    target = int(match.group(1))
    if target not in valid_ids:
        return _default_target(valid_ids, rng), False
    return target, True


def parse_speech(
    text: str,
    speaker_id: int,
    alive_ids: List[int],
    prior_speeches: Optional[List[str]] = None,
) -> ParsedSpeech:
    """Parse a speech, extracting accuses/defends and detecting violations.

    `prior_speeches` is the list of this speaker's PRIOR speech texts in the
    same game; used to detect repeated-speech format violations.
    """
    text = text or ""
    prior_speeches = prior_speeches or []
    text_lower = text.lower()

    result = ParsedSpeech(text=text)

    # Violation 1: role leak (works on substring; lowercase check)
    if any(phrase in text_lower for phrase in _ROLE_LEAK_PHRASES):
        result.format_valid = False

    # Violation 2: length floor — fewer than _MIN_SPEECH_TOKENS whitespace tokens
    if len(text.split()) < _MIN_SPEECH_TOKENS:
        result.format_valid = False

    # Violation 3: repeated speech (case-insensitive, stripped match)
    text_norm = text.strip().lower()
    if text_norm and any(text_norm == s.strip().lower() for s in prior_speeches):
        result.format_valid = False

    # Extract Player N mentions and classify by surrounding context
    for match in _PLAYER_MENTION_RE.finditer(text):
        pid = int(match.group(1))
        if pid not in alive_ids or pid == speaker_id:
            continue
        ctx_start = max(0, match.start() - 30)
        ctx_end = min(len(text_lower), match.end() + 30)
        ctx = text_lower[ctx_start:ctx_end]
        if any(k in ctx for k in _DEFEND_KEYWORDS):
            if pid not in result.defends:
                result.defends.append(pid)
        elif any(k in ctx for k in _ACCUSE_KEYWORDS):
            if pid not in result.accuses:
                result.accuses.append(pid)
        else:
            # Neutral mention — default to accuse, since the prompt asks
            # players to be specific about WHY they suspect someone.
            if pid not in result.accuses:
                result.accuses.append(pid)

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────


def _default_vote_target(alive_ids: List[int], self_id: int, rng: random.Random) -> int:
    candidates = [i for i in alive_ids if i != self_id]
    if not candidates:
        return self_id  # degenerate; only ever hit if game is broken
    return rng.choice(candidates)


def _default_target(valid_ids: List[int], rng: random.Random) -> int:
    if not valid_ids:
        return -1  # degenerate
    return rng.choice(valid_ids)
