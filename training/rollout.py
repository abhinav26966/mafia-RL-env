"""Custom rollout function for GRPOTrainer.

ONE rollout = ONE full Werewolf game from the spotlight seat's POV. The
model takes 3-6 turns per game (one speech + one vote per day, plus a
night action if the spotlight is werewolf or seer). We concatenate all
turn token ids into a single (prompt_ids, completion_ids, logprobs)
trajectory per game and attach per-rubric reward fields.

This sidesteps TRL #4543 (multi-step `prompt_ids` issue) by collapsing
one game into one sample.

Design notes:
- Format violations and accusations are tracked CLIENT-SIDE during the
  game (no /state extension needed). Only ground-truth roles are pulled
  from /state for the calibration reward.
- The `generate_fn` argument is injected so the rollout can be unit-tested
  with a deterministic mock (no GPU needed).
- We pass `seed = base_seed * 1000 + global_step` so each step sees a
  fresh game distribution while remaining reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from werewolf_env.client import WerewolfEnv
from werewolf_env.game.constants import MAX_DAYS
from werewolf_env.game.parser import parse_speech, parse_target, parse_vote
from werewolf_env.game.prompts import build_user_prompt, system_prompt_for_role
from werewolf_env.models import WerewolfAction, WerewolfObservation, WerewolfState


# ── Trajectory modes ─────────────────────────────────────────────────────────

TRAJECTORY_MODE_PER_TURN = "per_turn"
"""Concatenate per-turn token ids into one (prompt, completion, logprobs)
trajectory per game. Master plan default."""

TRAJECTORY_MODE_WHOLE_GAME = "whole_game"
"""Fallback if TRL #4543 hits us: present the whole game state at once
and ask the model to output a single all-in-one decision plan. NOT
implemented — flip to this mode in Phase 4 debug if per_turn breaks."""

DEFAULT_TRAJECTORY_MODE = TRAJECTORY_MODE_PER_TURN


# ── Rubric reward computation (client-side) ───────────────────────────────────


@dataclass
class RubricRewards:
    outcome: float
    calibration: float
    survival: float
    format: float


def compute_rubric_rewards(
    state: WerewolfState,
    *,
    spotlight_role: str,
    suspicions: List[int],
    format_violations: int,
    spotlight_alive_at_end: bool,
    spotlight_day_died: Optional[int],
) -> RubricRewards:
    """Compute the four rubric rewards for the spotlight from final state."""
    spotlight_faction = "werewolf" if spotlight_role == "werewolf" else "villager"

    # Outcome
    if state.winner is None:
        outcome = 0.0
    else:
        outcome = 1.0 if state.winner == spotlight_faction else 0.0

    # Calibration — accuracy of stated suspicions vs ground-truth opposing faction
    if not suspicions:
        calibration = 0.5
    else:
        correct = 0
        for sid in suspicions:
            sid_role = state.roles.get(sid)
            if sid_role is None:
                continue
            sid_faction = "werewolf" if sid_role == "werewolf" else "villager"
            if sid_faction != spotlight_faction:
                correct += 1
        calibration = correct / len(suspicions)

    # Survival — fraction of days survived
    if spotlight_alive_at_end:
        survival = 1.0
    elif spotlight_day_died is None:
        survival = 0.0
    else:
        survival = min(1.0, spotlight_day_died / MAX_DAYS)

    # Format
    fmt_reward = max(0.0, 1.0 - 0.25 * format_violations)

    return RubricRewards(
        outcome=outcome,
        calibration=calibration,
        survival=survival,
        format=fmt_reward,
    )


# ── Action builder ───────────────────────────────────────────────────────────


def build_action_from_text(
    text: str,
    obs: WerewolfObservation,
    *,
    rng: Any = None,
) -> tuple[WerewolfAction, bool, List[int]]:
    """Map free model text into a structured WerewolfAction.

    Returns (action, format_valid, accuses) where:
        - format_valid: True if the parser found a clean bracketed target
          (vote/target) OR if the speech survived all format checks.
        - accuses: list of player ids the speech accused (for calibration).
    """
    legal = obs.legal_action_types
    pid = obs.player_id
    alive = obs.alive_player_ids
    others_alive = [p for p in alive if p != pid]

    if "speak" in legal:
        parsed = parse_speech(text or "", speaker_id=pid, alive_ids=alive)
        return (
            WerewolfAction(
                player_id=pid,
                action_type="speak",
                content=(text or "(no comment)")[:600],
                metadata={"format_violation": True} if not parsed.format_valid else {},
            ),
            parsed.format_valid,
            list(parsed.accuses),
        )

    if "vote" in legal:
        target, valid = parse_vote(text or "", alive_ids=alive, self_id=pid, rng=rng)
        return (
            WerewolfAction(
                player_id=pid,
                action_type="vote",
                target_id=target,
                metadata={"format_violation": True} if not valid else {},
            ),
            valid,
            [],
        )

    if "night_kill" in legal:
        target, valid = parse_target(text or "", valid_ids=others_alive, rng=rng)
        return (
            WerewolfAction(
                player_id=pid,
                action_type="night_kill",
                target_id=target,
                metadata={"format_violation": True} if not valid else {},
            ),
            valid,
            [],
        )

    if "seer_check" in legal:
        target, valid = parse_target(text or "", valid_ids=others_alive, rng=rng)
        return (
            WerewolfAction(
                player_id=pid,
                action_type="seer_check",
                target_id=target,
                metadata={"format_violation": True} if not valid else {},
            ),
            valid,
            [],
        )

    raise ValueError(f"No legal action types in {legal}")


# ── Generation interface ─────────────────────────────────────────────────────


GenerateFn = Callable[
    [Any, List[List[Dict[str, str]]]],
    List[Dict[str, Any]],
]
"""A function (trainer, list_of_messages_lists) → list of dicts with
prompt_ids, completion_ids, logprobs. Real impl uses TRL's
`generate_rollout_completions`; tests pass a deterministic mock."""


def _default_generate_fn(trainer: Any, messages_batch: List[List[Dict[str, str]]]):
    """The real production generator — wraps TRL's helper."""
    from trl.experimental.openenv import generate_rollout_completions

    return generate_rollout_completions(trainer, messages_batch)


# ── Rollout function ─────────────────────────────────────────────────────────


def rollout_func(
    prompts: Sequence[Dict[str, Any]],
    trainer: Any,
    env_url: Optional[str] = None,
    *,
    trajectory_mode: str = DEFAULT_TRAJECTORY_MODE,
    generate_fn: Optional[GenerateFn] = None,
    base_seed: int = 0,
) -> Dict[str, List]:
    """Run one Werewolf game per prompt, return per-game trajectory + rewards.

    Args:
        prompts: dataset rows. Each row is a dict; we use `seed` if present.
        trainer: GRPOTrainer instance (gives us tokenizer + model + step).
        env_url: OpenEnv server URL. Defaults to $WEREWOLF_ENV_URL or
            http://localhost:8000.
        trajectory_mode: per_turn (default) or whole_game (fallback).
        generate_fn: injectable generation function for testing.
        base_seed: added to dataset seeds for reproducibility across runs.

    Returns:
        dict with keys: prompt_ids, completion_ids, logprobs,
        outcome_reward, calibration_reward, survival_reward, format_reward.
        All values are lists of length len(prompts).
    """
    env_url = env_url or os.environ.get("WEREWOLF_ENV_URL", "http://localhost:8000")
    generate_fn = generate_fn or _default_generate_fn
    if trajectory_mode != TRAJECTORY_MODE_PER_TURN:
        raise NotImplementedError(
            f"trajectory_mode={trajectory_mode!r} not implemented yet — "
            "see TRL #4543 fallback notes in this file"
        )

    global_step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)

    out_p_ids: List[List[int]] = []
    out_c_ids: List[List[int]] = []
    out_lp: List[List[float]] = []
    out_outcome: List[float] = []
    out_calibration: List[float] = []
    out_survival: List[float] = []
    out_format: List[float] = []

    for row_idx, row in enumerate(prompts):
        row_seed = int(row.get("seed", row_idx)) if isinstance(row, dict) else row_idx
        seed = (base_seed + row_seed * 1000 + global_step) % (2**31)

        format_violations = 0
        suspicions: List[int] = []
        spotlight_role: Optional[str] = None
        spotlight_alive_at_end = True
        spotlight_day_died: Optional[int] = None
        last_alive: List[int] = []

        turn_prompt_ids: List[List[int]] = []
        turn_completion_ids: List[List[int]] = []
        turn_logprobs: List[List[float]] = []

        with WerewolfEnv(base_url=env_url).sync() as client:
            result = client.reset(seed=seed)
            obs = result.observation
            spotlight_role = obs.role
            last_alive = list(obs.alive_player_ids)

            while not result.done:
                if obs.current_actor_id != obs.player_id:
                    # Defensive — env auto-fast-forwards NPCs, so this should
                    # never fire, but break gracefully if it does.
                    break

                # Build chat-format prompt for the LLM
                messages = [
                    {"role": "system", "content": system_prompt_for_role(obs.role)},
                    {"role": "user", "content": build_user_prompt(obs)},
                ]

                # Generate one turn
                gen = generate_fn(trainer, [messages])
                if not gen:
                    break
                p_ids = list(gen[0]["prompt_ids"])
                c_ids = list(gen[0]["completion_ids"])
                lp = list(gen[0]["logprobs"])

                # Decode the completion text
                if tokenizer is not None and hasattr(tokenizer, "decode"):
                    text = tokenizer.decode(c_ids, skip_special_tokens=True)
                else:
                    text = gen[0].get("text", "")

                action, format_valid, accuses = build_action_from_text(text, obs)
                if not format_valid:
                    format_violations += 1
                if accuses:
                    for sid in accuses:
                        if sid not in suspicions:
                            suspicions.append(sid)

                turn_prompt_ids.append(p_ids)
                turn_completion_ids.append(c_ids)
                turn_logprobs.append(lp)

                # Step the env
                pre_alive = list(obs.alive_player_ids)
                pre_day = obs.day
                result = client.step(action)
                obs = result.observation
                last_alive = list(obs.alive_player_ids)

                # Detect spotlight death
                if (
                    spotlight_alive_at_end
                    and 0 in pre_alive
                    and 0 not in obs.alive_player_ids
                ):
                    spotlight_alive_at_end = False
                    spotlight_day_died = pre_day

            state: WerewolfState = client.state()

            # Final alive check (covers spotlight dying mid-fast-forward at reset)
            if 0 not in last_alive:
                spotlight_alive_at_end = False
                if spotlight_day_died is None:
                    spotlight_day_died = state.day

        rewards = compute_rubric_rewards(
            state,
            spotlight_role=spotlight_role or "villager",
            suspicions=suspicions,
            format_violations=format_violations,
            spotlight_alive_at_end=spotlight_alive_at_end,
            spotlight_day_died=spotlight_day_died,
        )

        # Stitch turns into one trajectory per game (master-plan approach).
        # We use the FIRST turn's prompt as the trajectory prompt and the
        # CONCATENATION of all completions as the trajectory completion.
        if turn_prompt_ids:
            game_p = turn_prompt_ids[0]
            game_c: List[int] = []
            for c in turn_completion_ids:
                game_c.extend(c)
            game_lp: List[float] = []
            for lp in turn_logprobs:
                game_lp.extend(lp)
        else:
            # No model turns — spotlight died during reset's NPC fast-forward.
            # Emit a 1-token placeholder trajectory so TRL doesn't choke.
            placeholder = (
                getattr(tokenizer, "bos_token_id", None) if tokenizer is not None else None
            ) or 0
            game_p = [placeholder]
            game_c = []
            game_lp = []

        out_p_ids.append(game_p)
        out_c_ids.append(game_c)
        out_lp.append(game_lp)
        out_outcome.append(rewards.outcome)
        out_calibration.append(rewards.calibration)
        out_survival.append(rewards.survival)
        out_format.append(rewards.format)

    return {
        "prompt_ids": out_p_ids,
        "completion_ids": out_c_ids,
        "logprobs": out_lp,
        "outcome_reward": out_outcome,
        "calibration_reward": out_calibration,
        "survival_reward": out_survival,
        "format_reward": out_format,
    }
