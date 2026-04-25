"""WerewolfEnv — client for the Werewolf OpenEnv server.

Subclasses `openenv.core.EnvClient`. The base class handles WebSocket
connection management; subclasses provide payload encoding and result parsing.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from werewolf_env.models import WerewolfAction, WerewolfObservation, WerewolfState


class WerewolfEnv(EnvClient[WerewolfAction, WerewolfObservation, WerewolfState]):
    """Client for the Werewolf environment.

    Example:
        with WerewolfEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(seed=42)
            while not result.done:
                action = pick_action(result.observation)
                result = env.step(action)
    """

    def _step_payload(self, action: WerewolfAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[WerewolfObservation]:
        obs_data = payload.get("observation", {})
        observation = WerewolfObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> WerewolfState:
        return WerewolfState(**payload)
