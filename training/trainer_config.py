"""GRPOConfig builders. Two profiles to match our compute targets:

    - T4 (Colab Free, 16 GB) — debug + first run with Gemma-3 1B
    - A100 (Colab Pro / HF compute, 40 GB) — final run with Qwen3-1.7B

Phase 4 deliverable. Constants here are first-pass defaults; tune after the
20-step debug run.
"""

from dataclasses import dataclass


@dataclass
class HardwareProfile:
    name: str
    gpu_memory_utilization: float
    max_seq_length: int
    max_prompt_length: int
    max_completion_length: int
    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int


T4_PROFILE = HardwareProfile(
    name="t4",
    gpu_memory_utilization=0.55,    # vLLM colocate is tight on 16 GB
    max_seq_length=2048,
    max_prompt_length=1024,
    max_completion_length=256,
    num_generations=4,              # GRPO group size — keep small on T4
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # effective batch = 8 games per step
)


A100_PROFILE = HardwareProfile(
    name="a100",
    gpu_memory_utilization=0.4,
    max_seq_length=4096,
    max_prompt_length=2048,
    max_completion_length=512,
    num_generations=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # effective batch = 32 games per step
)


def build_config(output_dir: str, profile: HardwareProfile = T4_PROFILE):
    """Returns a GRPOConfig.

    NOTE: vLLM is intentionally disabled. TRL 0.20.0's `GuidedDecodingParams`
    import is incompatible with vLLM 0.12+, and vLLM <0.12 is incompatible
    with Colab's torch 2.10. We use HF transformers.generate via the
    rollout function instead — slower but actually works.
    """
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        learning_rate=5e-6,
        per_device_train_batch_size=profile.per_device_train_batch_size,
        gradient_accumulation_steps=profile.gradient_accumulation_steps,
        num_generations=profile.num_generations,
        max_prompt_length=profile.max_prompt_length,
        max_completion_length=profile.max_completion_length,
        warmup_steps=10,
        logging_steps=1,
        save_steps=50,
        eval_steps=50,
        use_vllm=False,
        report_to=["trackio"],
        seed=42,
    )
