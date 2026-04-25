"""Custom rollout function for GRPOTrainer.

ONE rollout = ONE full Werewolf game from the spotlight seat's POV. The model
takes 3-6 turns per game (one speech + one vote per day, optionally a night
action if werewolf or seer). We concatenate the token ids of all turns into
a single (prompt_ids, completion_ids, logprobs) trajectory per game and
attach four per-rubric reward fields.

This sidesteps the multi-step rollout known issue (TRL #4543) by collapsing
one game into one sample.

Phase 4 deliverable. Stub raises NotImplementedError.
"""

from typing import Any, Dict, List

# Two trajectory modes — the second is a fallback if per-turn stitching breaks
# under TRL #4543 in our setup.
TRAJECTORY_MODE_PER_TURN = "per_turn"
TRAJECTORY_MODE_WHOLE_GAME = "whole_game"
DEFAULT_TRAJECTORY_MODE = TRAJECTORY_MODE_PER_TURN


def rollout_func(
    prompts: List[str],
    trainer: Any,
    env_url: str,
    trajectory_mode: str = DEFAULT_TRAJECTORY_MODE,
) -> Dict[str, List]:
    """Phase 4: implement per MASTER_PLAN §10 rollout.py."""
    raise NotImplementedError("rollout_func — Phase 4")
