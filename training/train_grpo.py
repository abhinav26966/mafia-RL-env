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

import importlib.abc as _ila
import importlib.util as _ilu
import os
import sys as _sys
import types as _types
from unittest.mock import MagicMock as _MagicMock

# IMPORTANT: set BEFORE importing unsloth, per Unsloth memory-efficient RL guide
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

# Disable Unsloth's source-introspection compiler — it calls
# `inspect.getsource(Trainer.training_step)` which can fail on Colab with
# `OSError: could not get source code`. See unslothai/unsloth#1224, #742.
# Unsloth still applies its model-load optimizations without this compiler.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")


# ── TRL optional-deps stub (must run BEFORE any `import trl`) ────────────────
#
# TRL >= 0.20's grpo_trainer module unconditionally imports several optional
# packages (mergekit for LoRA merge, llm_blender for ensemble ranking,
# liger_kernel for CUDA kernels). We don't use any of those features — only
# the import has to succeed.
#
# Mergekit specifically defines Pydantic models with raw `torch.Tensor`
# fields which fail under pydantic >= 2.12 even if the package is installed.
#
# A meta-path import hook auto-mocks anything under known prefixes. It uses
# real `types.ModuleType` instances (not MagicMock) so `module.__spec__` is
# set correctly by importlib — `importlib.util.find_spec` can then read it
# without a "ValueError: __spec__ is not set". Per-module attribute access
# falls back to MagicMock via PEP 562's module-level `__getattr__`.


class _StubLoader(_ila.Loader):
    def create_module(self, spec):  # type: ignore[override]
        m = _types.ModuleType(spec.name)
        m.__path__ = []  # mark as package — submodule imports keep going
        # Pre-set common metadata so callers expecting strings don't choke
        m.__version__ = "0.0.0"
        m.__author__ = ""
        m.__file__ = "<stub>"
        m.__all__: list = []

        def _module_getattr(name):  # PEP 562 fallback
            # String defaults for metadata dunders — TRL does
            # `if mod.__version__ >= "0.1"` and a MagicMock comparison
            # against a string raises TypeError.
            if name == "__version__":
                return "0.0.0"
            if name in ("__author__", "__doc__", "__license__",
                        "__email__", "__url__"):
                return ""
            if name == "__all__":
                return []
            return _MagicMock()

        m.__getattr__ = _module_getattr  # type: ignore[attr-defined]
        return m

    def exec_module(self, module):  # type: ignore[override]
        pass


class _StubFinder(_ila.MetaPathFinder):
    # Optional packages that TRL's grpo_trainer imports unconditionally.
    # NOTE: we do NOT stub `vllm` itself — Unsloth's monkey-patcher needs to
    # set `__init__` on real vllm classes, which fails on MagicMock. Instead
    # we let real vllm 0.19+ load and apply a `GuidedDecodingParams` alias
    # at the end of this file (see _apply_vllm_shim).
    PREFIXES: tuple[str, ...] = (
        "mergekit",          # LoRA merging
        "llm_blender",       # PairRanker ensemble
        "liger_kernel",      # CUDA fused kernels
        "weave",             # W&B observability
        "comet_ml",          # tracker
        "swanlab",           # tracker
        "vllm_ascend",       # Huawei NPU plugin (irrelevant on CUDA)
    )

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[override]
        for prefix in self.PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                return _ilu.spec_from_loader(fullname, _StubLoader(), is_package=True)
        return None


# Wipe any prior broken entries (e.g., bare MagicMocks left by an earlier shim
# without a real __spec__). The finder will recreate them properly on import.
for _k in list(_sys.modules):
    if any(_k == _p or _k.startswith(_p + ".") for _p in _StubFinder.PREFIXES):
        del _sys.modules[_k]

if not any(isinstance(_f, _StubFinder) for _f in _sys.meta_path):
    _sys.meta_path.insert(0, _StubFinder())


def _apply_vllm_shim() -> None:
    """Alias `vllm.sampling_params.GuidedDecodingParams` to its renamed
    successor `StructuredOutputsParams` (vllm 0.12+). TRL 0.20 imports the
    old name unconditionally; this lets the import succeed without needing
    to stub all of vllm.

    Falls back to a dummy class if vllm has neither name (defense in depth)."""
    try:
        import vllm.sampling_params as _vsp
    except Exception:  # pragma: no cover  — vllm not installed at all
        return
    if hasattr(_vsp, "GuidedDecodingParams"):
        return  # already there (older vllm)
    new_cls = getattr(_vsp, "StructuredOutputsParams", None)
    if new_cls is not None:
        _vsp.GuidedDecodingParams = new_cls
        return

    class _GuidedDecodingParamsShim:  # last-resort stub
        def __init__(self, *args, **kwargs):
            pass

    _vsp.GuidedDecodingParams = _GuidedDecodingParamsShim


_apply_vllm_shim()


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
    # NOTE: fast_inference=False (no vLLM colocate). Reason:
    # TRL 0.20.0 imports `GuidedDecodingParams` from vLLM, which only exists
    # in vLLM <0.12. vLLM <0.12 is built against torch 2.8 and breaks with
    # Colab's torch 2.10 (C++ ABI mismatch). Until TRL ships a version that
    # uses `StructuredOutputsParams` (the new vLLM name), we go without vLLM
    # colocate and use HF transformers.generate for rollouts. Slower but
    # actually works on current Colab.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=profile.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
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
