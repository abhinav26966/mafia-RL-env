"""FastAPI application for the Werewolf environment.

Created via `openenv.core.env_server.http_server.create_app`, which wires:
    POST /reset        — reset the environment
    POST /step         — apply an action
    GET  /state        — full ground-truth state (used by trainer for rewards)
    GET  /schema       — action/observation JSON schemas
    GET  /web          — interactive web interface (HF Space)
    GET  /docs         — OpenAPI / Swagger
    WS   /ws           — persistent session for low-latency rollouts

Run locally:
    uvicorn werewolf_env.server.app:app --port 8000
or:
    python -m werewolf_env.server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core[core] is required. Install via `pip install -e .` from the repo root."
    ) from e

try:
    from werewolf_env.models import WerewolfAction, WerewolfObservation
    from werewolf_env.server.werewolf_environment import WerewolfEnvironment
except ModuleNotFoundError:  # pragma: no cover  — HF Space container path
    from models import WerewolfAction, WerewolfObservation  # type: ignore[import-not-found,no-redef]
    from server.werewolf_environment import WerewolfEnvironment  # type: ignore[import-not-found,no-redef]


app = create_app(
    WerewolfEnvironment,
    WerewolfAction,
    WerewolfObservation,
    env_name="werewolf_env",
    # Set high enough to match GRPO generation_batch_size during training.
    # Phase 4 may bump this to 64 once the env is stable.
    max_concurrent_envs=8,
)


def main() -> None:
    """Entry point for `python -m werewolf_env.server.app` and `uv run`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
