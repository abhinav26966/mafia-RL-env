---
title: DECEIT — Werewolf Env
emoji: 🐺
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
license: apache-2.0
tags:
  - openenv
  - multi-agent
  - social-deduction
  - rl
---

# DECEIT — Werewolf social-deduction environment

5-player hidden-role Werewolf/Mafia, OpenEnv-compliant. The trainer's model controls one "spotlight" seat; the other 4 are filled by deterministic NPC bots.

## Quickstart

```python
from werewolf_env import WerewolfEnv, WerewolfAction

with WerewolfEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42)
    obs = result.observation
    print(f"You are Player {obs.player_id}, role={obs.role}, phase={obs.phase}")
```

## Endpoints

- `POST /reset` — start a new episode
- `POST /step` — apply an action
- `GET /state` — full ground-truth state (for trainers, debug, demo)
- `WS /ws` — persistent session (recommended for training)
- `GET /web` — interactive web interface
- `GET /docs` — OpenAPI/Swagger

## Action / Observation

See `werewolf_env/models.py` for the canonical Pydantic schemas.

## License

Apache-2.0.
