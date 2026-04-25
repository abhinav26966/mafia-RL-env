"""Trivial dataset for GRPO — one row per game, prompt is built dynamically
inside the rollout function from observations. Phase 4 deliverable.
"""


def build_dataset(n_games: int = 1024):
    """Return a HF Dataset with N empty seed rows. Each row triggers one game."""
    from datasets import Dataset

    return Dataset.from_dict({"prompt": [""] * n_games, "seed": list(range(n_games))})
