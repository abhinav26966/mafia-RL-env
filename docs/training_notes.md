# Training notes

## Compute targets

| Profile  | Hardware             | Use                                       | Model           |
| -------- | -------------------- | ----------------------------------------- | --------------- |
| T4       | Colab Free, 16 GB    | Phase 4 debug + first real training run   | Gemma-3 1B (4-bit) |
| A100     | Colab Pro / HF compute, 40 GB | Phase 5 final run                         | Qwen3-1.7B (4-bit) |

## T4 recipe (do this first)

```bash
!pip install --upgrade pip
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "trl>=0.20" "peft>=0.12"
!pip install -r requirements-train.txt
!pip install -e .
```

Use the `T4_PROFILE` from `training/trainer_config.py`:

- `gpu_memory_utilization=0.55` (vLLM colocate is tight on 16 GB)
- `max_seq_length=2048`, `max_prompt_length=1024`, `max_completion_length=256`
- `num_generations=4` (GRPO group size)
- `gradient_accumulation_steps=8`

Run 20 GRPO steps as a debug. Watch for:

- `format_reward` rising fast → parser is accepting outputs.
- `outcome_reward` non-zero on at least 1 in 8 rollouts → spotlight ever
  wins occasionally.
- No NaN losses, no empty completions.

If `format_reward` stays at 0 for 30+ steps, the parser is rejecting
everything. Symptom + fix: relax the strict bracket regex, log a parser-
failure trace every 10 steps.

## A100 upgrade (for the final run)

Switch model to `Qwen/Qwen3-1.7B`, switch profile to `A100_PROFILE`. Run
200–400 steps. ~2–3 h on A100.

## Saving — DO NOT naively merge 4-bit → 16-bit

Per Unsloth's RL guide:

```python
# Step 1: save LoRA-only first and verify inference works
trainer.save_model("outputs/werewolf_grpo/final")
# Step 2: only after step 1 verifies — merge to 16-bit if needed
model.save_pretrained_merged("outputs/werewolf_grpo/merged_16bit",
                             tokenizer, save_method="merged_16bit")
```

Naive `model.merge_and_unload()` on a 4-bit base will silently corrupt
quality. We test post-save inference before declaring success.

## Episode-length distribution (Phase 2 finding)

The spotlight (seat 0) gets a random role each episode. ~20% of episodes
have the spotlight killed on night 1 (NPC werewolf picks a random
non-werewolf, and seat 0 is one of 4 candidates when spotlight is
villager-faction, which is 4/5 of episodes → 4/5 × 1/4 = 20%). In those
episodes the trainer's model takes **zero** actions and the env returns
done=True from `reset()` with a terminal reward.

Why we keep this behaviour:
- GRPO uses *relative* advantage within a group of 8 samples. Out of 8,
  expected 1.6 are zero-action; the other ~6.4 still produce variance.
- Wall-time-wise, zero-action episodes are nearly free (no model
  generations required).
- A "spotlight-protection on night 1" mitigation is implemented in
  `werewolf_env/game/npc.py` as an option but disabled by default.
  Re-enable in Phase 4 if reward signal turns out to be too sparse.

If during Phase 4 the trainer's reward stays flat for 50+ steps, flip
`NPCPool(protect_spotlight_day1=True)` (TODO) and re-run.

## What can go wrong (decision log)

| Problem                                 | Why                                    | Fix                                                     |
| --------------------------------------- | -------------------------------------- | ------------------------------------------------------- |
| `format_reward` flat at 0               | Parser too strict                      | Relax regex; add tolerant default-vote fallback         |
| `outcome_reward` flat                   | NPCs too smart; spotlight never wins   | Replace one Villager with `GullibleVillager` heuristic  |
| GRPO collapses to single strategy       | LR too high, group too small           | Drop LR to 2e-6; increase `num_generations` if VRAM allows |
| Context overflow on day 4               | Public log unbounded                   | Truncate oldest events first; cap prompt at `max_prompt_length` |
| Trainer's vLLM mode mismatch            | Wrong colocate config                  | `vllm_mode="colocate"`, set `gpu_memory_utilization` correctly  |
| Multi-step rollout fails (TRL #4543)    | Stitching `prompt_ids` across turns    | Switch `TRAJECTORY_MODE` to `whole_game` in `rollout.py`         |
| LoRA save corruption                    | 4-bit → 16-bit naive merge             | Save `lora` first, verify inference, then `merged_16bit`         |
| HF Space cold-start during training     | Free Space sleeps after 48 h idle      | Duplicate to team account, set `max_concurrent_envs` high        |
| Time runs out before Qwen finishes      | Training takes 2–3 h on A100           | Submit Gemma-3 1B mini-run as fallback                           |
