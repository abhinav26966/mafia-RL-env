"""Main entrypoint for GRPO post-training of the Werewolf agent.

Two compute targets:
    - T4 (Colab Free, 16 GB)        — debug + first run with Gemma-3 1B
    - A100 (Colab Pro / HF, 40 GB)  — final run with Qwen3-1.7B

Run via:
    python -m training.train_grpo                           # uses T4 defaults
    HARDWARE=a100 python -m training.train_grpo             # promote
    WEREWOLF_ENV_URL=http://localhost:8000 python -m ...    # local env

Notebook usage:
    from training.train_grpo import run_training
    run_training(profile_name="t4", n_games=64, max_steps=20)
"""

from __future__ import annotations

import os
import sys as _sys
from unittest.mock import MagicMock as _MagicMock

# IMPORTANT: set BEFORE importing unsloth, per Unsloth memory-efficient RL guide
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

# Stub out mergekit BEFORE any TRL import.
# TRL >= 0.20's grpo_trainer module unconditionally imports mergekit. Recent
# mergekit versions define Pydantic models with raw `torch.Tensor` fields,
# which fails to generate a schema under pydantic >= 2.12. We don't use
# mergekit's LoRA-merge feature — only the import statement has to succeed.
# Pre-registering MagicMocks satisfies any `import mergekit.X` or
# `from mergekit.X import Y` in TRL's code.
for _mod in [
    "mergekit", "mergekit.config", "mergekit.merge", "mergekit.options",
    "mergekit.io", "mergekit.architecture", "mergekit.graph",
    "mergekit.merge_methods", "mergekit.scripts", "mergekit.tokenizer",
    "mergekit.plan", "mergekit.common",
]:
    _sys.modules.setdefault(_mod, _MagicMock())


# ── Compute-tier defaults ────────────────────────────────────────────────────

T4_MODEL = "unsloth/gemma-3-1b-it"
A100_MODEL = "Qwen/Qwen3-1.7B"

# Default OpenEnv server URL — points at local uvicorn for training (low latency).
# Override to the live HF Space for one-off experiments:
#   WEREWOLF_ENV_URL=https://abhinav2696-werewolf-env.hf.space python -m training.train_grpo
DEFAULT_LOCAL_URL = "http://localhost:8000"
SPACE_URL = "https://abhinav2696-werewolf-env.hf.space"
ENV_URL: str = os.environ.get("WEREWOLF_ENV_URL", DEFAULT_LOCAL_URL)


# ── Public entrypoint ────────────────────────────────────────────────────────


def run_training(
    profile_name: str = "t4",
    *,
    model_name: str | None = None,
    n_games: int = 64,
    max_steps: int | None = 20,
    output_dir: str = "outputs/werewolf_grpo",
    env_url: str | None = None,
    base_seed: int = 0,
    save_adapter: bool = True,
):
    """Build the trainer and run training.

    Args:
        profile_name: 't4' or 'a100' — picks hardware profile from trainer_config.
        model_name: HF model id. Defaults to T4_MODEL or A100_MODEL based on profile.
        n_games: dataset size (rows). Each row = one game during training.
        max_steps: cap training at this many steps (None = run full epoch). 20 for debug.
        output_dir: where to save the LoRA adapter.
        env_url: OpenEnv URL. Defaults to $WEREWOLF_ENV_URL or http://localhost:8000.
        base_seed: stable across runs; mixed with row_seed + step inside the rollout.
        save_adapter: if True, save the LoRA after training.

    Returns:
        The trained `GRPOTrainer` instance.
    """
    # Lazy imports so the module imports cleanly on machines without Unsloth/TRL
    from functools import partial

    from datasets import Dataset
    from trl import GRPOTrainer
    from unsloth import FastLanguageModel

    from training.rollout import rollout_func
    from training.trainer_config import (
        A100_PROFILE,
        T4_PROFILE,
        build_config,
    )

    profile = T4_PROFILE if profile_name.lower() == "t4" else A100_PROFILE
    model_id = model_name or (T4_MODEL if profile.name == "t4" else A100_MODEL)
    url = env_url or ENV_URL

    print(f"[train] profile={profile.name} model={model_id} n_games={n_games} url={url}")

    # ── Load model + LoRA ────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=profile.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,                 # vLLM colocate
        gpu_memory_utilization=profile.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=base_seed,
    )

    # ── Dataset (one row per game) ───────────────────────────────────────────
    dataset = Dataset.from_dict(
        {
            "prompt": [""] * n_games,
            "seed": list(range(n_games)),
        }
    )

    # ── Trainer config ───────────────────────────────────────────────────────
    config = build_config(output_dir=output_dir, profile=profile)
    if max_steps is not None:
        config.max_steps = max_steps

    # ── Rollout binding (inject env_url + base_seed) ─────────────────────────
    bound_rollout = partial(rollout_func, env_url=url, base_seed=base_seed)

    # ── Reward funcs (registered as 4 independent rubrics for Trackio) ───────
    from training.reward_funcs import (
        DEFAULT_REWARD_WEIGHTS,
        reward_calibration_fn,
        reward_format_fn,
        reward_outcome_fn,
        reward_survival_fn,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_outcome_fn,
            reward_calibration_fn,
            reward_survival_fn,
            reward_format_fn,
        ],
        reward_weights=DEFAULT_REWARD_WEIGHTS,
        rollout_func=bound_rollout,
        train_dataset=dataset,
        args=config,
    )

    print("[train] starting training…")
    trainer.train()

    if save_adapter:
        adapter_path = os.path.join(output_dir, "final_adapter")
        # Per Unsloth RL guide: save lora-only first; verify inference; then maybe merge
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"[train] saved LoRA adapter to {adapter_path}")

    return trainer


def main() -> None:  # pragma: no cover
    profile = os.environ.get("HARDWARE", "t4").lower()
    n_games = int(os.environ.get("N_GAMES", "64"))
    max_steps_env = os.environ.get("MAX_STEPS")
    max_steps = int(max_steps_env) if max_steps_env else 20
    run_training(
        profile_name=profile,
        n_games=n_games,
        max_steps=max_steps,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
