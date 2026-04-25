# DECEIT — Werewolf social-deduction OpenEnv environment

A multi-agent social-deduction RL environment (5-player Werewolf/Mafia) that teaches an LLM to **lie convincingly when impostor and detect lies when villager**. Submitted to the Meta OpenEnv Hackathon (April 2026) under **Theme #1 — Multi-Agent Interactions**.

> **Status:** in development. This README is a placeholder; results, plots, and links go in here as we ship phases.

## Why this environment

Most OpenEnv submissions in social/strategic spaces are perfect-information games (chess, snake, gridworlds). Werewolf is **hidden-role, partially observable, and language-native** — you can't win without theory of mind. That makes it a clean training target for a class of behavior current LLMs are *bad at*: maintaining a consistent fake identity, building a case against another speaker over multiple turns, and updating beliefs from indirect cues.

## Game in one paragraph

5 seats, 1 Werewolf + 1 Seer + 3 Villagers. Phases cycle: NIGHT (Werewolf privately picks a kill, Seer privately checks one player's role) → DAY_DISCUSSION (each alive player speaks once, ≤80 tokens) → DAY_VOTE (each alive player votes one player out) → RESOLUTION. Werewolf wins by reaching parity with villagers; villagers win by voting the Werewolf out. Max 4 days.

## Quickstart

```bash
git clone <repo-url> && cd mafia-RL-project
pip install -e ".[dev]"
pytest                                            # unit tests for game logic
uvicorn werewolf_env.server.app:app --port 8000   # local OpenEnv server
python scripts/play_self_random.py --games 100    # smoke-test 100 NPC games
```

Training (Colab):

```python
!pip install -r requirements-train.txt
!pip install -e .
# then open notebooks/02_train.ipynb
```

## Repository layout

```
werewolf_env/      # OpenEnv-compliant env package (this is the HF Space contents)
├── game/          # pure game logic, no openenv deps
├── server/        # FastAPI app + Dockerfile
├── models.py      # WerewolfAction / WerewolfObservation
└── client.py      # WerewolfEnv (EnvClient subclass)

tests/             # pytest suite — green gate at end of each phase
scripts/           # CLI tools: play_self_random, eval_model, compare_runs
training/          # GRPO trainer + rollout function
notebooks/         # 01_walkthrough, 02_train (submission), 03_compare
demo/              # transcripts + plots for the writeup
docs/              # architecture, reward design, training notes, API deltas
```

## Links (filled in as we ship)

- HF Space: _TBD_
- Training notebook (Colab): _TBD_
- 90-second video: _TBD_
- Plots: see `demo/plots/`
- Transcripts: see `demo/transcripts/`

## License

Apache-2.0.
