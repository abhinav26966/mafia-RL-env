# Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Trainer (Colab T4 / A100)                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  GRPOTrainer (TRL)                                         │     │
│  │   ├── reward_funcs = [outcome, calibration, survival, fmt] │     │
│  │   ├── rollout_func: 1 game per row                         │     │
│  │   └── colocated vLLM (Unsloth, 4-bit LoRA)                 │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                  WebSocket (sync wrapper)                           │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
┌──────────────────────────────┼──────────────────────────────────────┐
│              HF Space (Docker, FastAPI server)                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  WerewolfEnvironment(Environment)                          │     │
│  │   ├── reset(seed) -> Observation (POV of seat 0)           │     │
│  │   ├── step(action) -> Observation                          │     │
│  │   ├── state property -> WerewolfState (incl. ground truth) │     │
│  │   └── _fast_forward_to_spotlight() drives NPC seats        │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  GameEngine (pure Python, no openenv deps)                 │     │
│  │   ├── new_game / current_actor / legal_action_types        │     │
│  │   ├── apply_action -> validates and mutates GameState      │     │
│  │   └── advance_phase + _check_win                           │     │
│  └────────────────────────────────────────────────────────────┘     │
│                              │                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │  NPCPool (3 personas) + Parser (strict bracket format)     │     │
│  │  Reward functions (outcome, calibration, survival, format) │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## Spotlight rollout

GRPO requires N independent rollouts per group; true multi-agent self-play
(OMAR-style) needs PPO-with-hierarchical-advantage which TRL does not
currently support. We sidestep by making *one* model-controlled seat (the
"spotlight") and filling the other 4 with deterministic NPCs. This:

1. Makes each rollout independent (the only stochastic source for the same
   seed is the model's own sampling).
2. Lets us train against a fixed-but-not-trivial opponent pool.
3. Trades off some emergent dynamics for hackathon-grade simplicity.

Stretch: replace NPCs with a frozen reference checkpoint of the model
itself (mid-training snapshot) for richer opponents.

## Reward composition

Four pure functions over the final `GameState`, each in [0, 1]:

| Reward      | Weight | What it measures                                                          |
| ----------- | -----: | ------------------------------------------------------------------------- |
| outcome     |   0.50 | 1 if the spotlight's faction won, else 0                                  |
| calibration |   0.25 | Mean accuracy of stated suspicions vs ground truth                        |
| survival    |   0.15 | Fraction of game days the spotlight was alive                             |
| format      |   0.10 | 1.0 - 0.25 × format_violations, floored at 0                              |

They're registered as four SEPARATE `reward_funcs` in TRL so each shows up
as its own column in Trackio, and so judges can see independent rubric
movement. See `docs/reward_design.md` for anti-hacking details.

## Why this scores well on the hackathon rubric

| Rubric (weight)              | How we earn it                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------- |
| Innovation (40%)             | Hidden-role multi-agent deception — no existing OpenEnv covers it.              |
| Storytelling (30%)           | Side-by-side baseline vs trained transcripts; 90s video; clear README.          |
| Reward improvement (20%)     | Four independent reward curves + baseline-vs-trained win-rate plot.             |
| Reward & pipeline (10%)      | Multiple independent rubrics, anti-hacking checks, clean GRPO rollout func.    |
