#!/usr/bin/env bash
# Colab T4 bootstrap — battle-tested install order for the DECEIT training stack.
#
# Run with:
#     !bash scripts/colab_bootstrap.sh
#
# After it prints "── done ──", proceed to:
#   1. Launch local uvicorn
#   2. Run the stub-finder cell
#   3. Run training (training/train_grpo.py:run_training)
#
# Pinned versions (battle-tested 2026-04-26 on Colab Free T4):
#   transformers : whatever Unsloth picks (4.55-4.57 range)
#   trl          : 0.20.0   (--no-deps, so it doesn't drag transformers)
#   pydantic     : 2.10.6   (mergekit's torch.Tensor schemas need <2.12)
#   vllm         : 0.10.2   (TRL 0.20.0's expected version; 0.19+ removed
#                            `GuidedDecodingParams` from sampling_params)
#   unsloth      : git main (regex bug for newer transformers fixed there)

set -e

echo "── upgrading pip ──"
pip install --quiet --upgrade pip

echo "── installing unsloth (Colab-friendly install) ──"
pip install --quiet "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

echo "── installing trl + peft no-deps ──"
pip install --quiet --no-deps "trl==0.20.0" "peft>=0.12"

echo "── installing pydantic pin + soft deps ──"
pip install --quiet "pydantic==2.10.6" datasets accelerate trackio matplotlib pandas seaborn huggingface-hub

echo "── installing vLLM 0.19.1 (with all deps) ──"
# NOTE: vLLM 0.19.1 is the latest torch-2.10-compatible version. We let pip
# resolve all its transitive deps (cbor2, msgpack, etc.) — DO NOT use
# --no-deps here, that leaves vllm half-broken at import time.
# We don't actually USE vllm at training time (use_vllm=False) but it has to
# be importable because:
#   1. Unsloth checks `importlib.metadata.version('vllm')` at load
#   2. TRL's grpo_trainer imports `from vllm.sampling_params import ...`
# The notebook's Cell B aliases GuidedDecodingParams -> StructuredOutputsParams
# so step 2 succeeds despite vLLM 0.19's renaming.
pip install --quiet "vllm==0.19.1"

echo "── installing env package (editable) ──"
pip install --quiet -e .

echo "── verifying versions ──"
python -c "import transformers, trl, pydantic; print(f'transformers={transformers.__version__}, trl={trl.__version__}, pydantic={pydantic.VERSION}')"

echo "── done ──"
