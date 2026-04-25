# OpenEnv 0.2.3 API deltas vs MASTER_PLAN.md

This file captures every place the master plan's code differs from the
canonical `openenv init` output and the actual openenv-core 0.2.3 surface.
**Read this before Phase 1/2.** Each delta is something we already
absorbed into the scaffold; future code in those files should follow
the canonical convention, not the master plan.

## 1. Project layout — flat, not src-style

| Master plan                         | Canonical 0.2.3                        |
| ----------------------------------- | -------------------------------------- |
| `src/werewolf_env/...`              | `werewolf_env/...` (flat at repo root) |
| `src/werewolf_env/server/environment.py` | `werewolf_env/server/werewolf_environment.py` (env-name-prefixed) |

We followed canonical. `pyproject.toml` at the env-package root uses explicit
`package-dir` mapping; the repo-root `pyproject.toml` uses `find` to discover
`werewolf_env*`.

## 2. `create_app` import path and signature

Master plan code:

```python
from openenv.core.env_server import create_app
app = create_app(
    env_factory=lambda: WerewolfEnvironment(),
    action_cls=WerewolfAction,
    observation_cls=WerewolfObservation,
    max_concurrent_envs=64,
)
```

Canonical 0.2.3:

```python
from openenv.core.env_server.http_server import create_app
app = create_app(
    WerewolfEnvironment,            # CLASS (positional), not factory
    WerewolfAction,
    WerewolfObservation,
    env_name="werewolf_env",
    max_concurrent_envs=8,
)
```

Two changes:
- Import is `openenv.core.env_server.http_server.create_app`, not `openenv.core.env_server.create_app`.
- First arg is the env CLASS, not a factory. The server instantiates per session when `SUPPORTS_CONCURRENT_SESSIONS=True`.

## 3. `Environment` base class — kwargs-flexible

`Environment.reset` and `step` accept `**kwargs` in the base. Subclasses can
match the simpler `reset(self, seed=None)` / `step(self, action)` form, but
must add `# type: ignore[override]` if they don't accept `episode_id` /
`timeout_s`. We do this in `werewolf_environment.py`.

## 4. Concurrency flag

Set `SUPPORTS_CONCURRENT_SESSIONS = True` on the env class so the FastAPI
factory can serve independent episodes in parallel during GRPO rollout
collection. Without it, only one client can connect at a time.

## 5. WebSocket-first transport

`EnvClient` connects via WebSocket (`/ws`), not request/response HTTP. The
client class still lives at `openenv.core.EnvClient` (re-exported) and
subclasses still implement `_step_payload`, `_parse_result`, `_parse_state`
exactly as the master plan describes.

`base_url` accepts `http://...` or `ws://...`; it's normalized to
`ws://...//ws` internally.

For sync wrapper:

```python
with WerewolfEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42)
```

The `.sync()` is required when calling from synchronous code (rollouts in
the GRPOTrainer rollout function).

## 6. `openenv.yaml` schema

| Master plan field                  | Canonical 0.2.3 field          |
| ---------------------------------- | ------------------------------ |
| `client.class_name`, `client.module` | _(removed)_                  |
| `action.class_name`, `action.module` | _(removed)_                  |
| `observation.class_name`, ...        | _(removed)_                  |
| `default_image: ...`                 | _(removed)_                  |
| —                                    | `type: space`                |
| —                                    | `runtime: fastapi`           |
| —                                    | `app: server.app:app`        |
| —                                    | `port: 8000`                 |

The new manifest is much smaller. The action/observation/client modules are
discovered through Python imports, not YAML.

## 7. HF Space README front matter

Required for HF Space rendering:

```yaml
---
title: ...
emoji: ...
colorFrom: ...
colorTo: ...
sdk: docker
pinned: false
app_port: 8000
base_path: /web
license: apache-2.0
tags:
  - openenv
---
```

`base_path: /web` is the convention OpenEnv uses to expose the gradio web
interface at `<space-url>/web`. Without it the web UI won't render.

## 8. Dockerfile uses `openenv-base` image + uv

Canonical Dockerfile uses `ghcr.io/meta-pytorch/openenv-base:latest` plus
`uv sync` for reproducible installs. Our `werewolf_env/server/Dockerfile`
matches verbatim. Don't replace with a stock `python:3.11-slim` — the
`openenv-base` image bakes in helpful runtime support (web UI, /health, etc.).

## 9. State subclass — pydantic, not dataclass

`openenv.core.env_server.types.State` is a `pydantic.BaseModel` with
`extra="allow"`. Master plan describes it correctly but the actual fields
are `episode_id`, `step_count` only. Custom fields like `game_id`, `roles`
need to be declared on the subclass with `Field(...)` defaults — which we do.

## 10. Observation `reward` type

`Observation.reward` is `bool | int | float | None`. Master plan says
`float | None` — close enough. Use `float` in our case.

---

**TL;DR for Phase 1/2 implementers:** the *concepts* in MASTER_PLAN §5–§9
are correct, but the *imports* and *shapes* should follow the scaffold in
this repo, which already absorbed every delta above.
