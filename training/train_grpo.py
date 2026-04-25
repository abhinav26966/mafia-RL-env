"""Main entrypoint for GRPO post-training.

Two compute targets:
    - T4 (Colab Free)        — debug run + first attempt with Gemma-3 1B
    - A100 (Colab Pro / HF)  — final run with Qwen3-1.7B

Phase 4 deliverable. Stub raises NotImplementedError so the file imports.
"""

import os

# IMPORTANT: set BEFORE importing unsloth, per Unsloth memory-efficient RL guide
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")


T4_MODEL = "unsloth/gemma-3-1b-it"
A100_MODEL = "Qwen/Qwen3-1.7B"


def main():  # pragma: no cover
    raise NotImplementedError(
        "train_grpo.main — Phase 4. See notebooks/02_train.ipynb for the canonical entrypoint."
    )


if __name__ == "__main__":
    main()
