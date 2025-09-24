from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State


class SingleTurnEnv(MultiTurnEnv):
    """
    Environment for single-turn tasks (chat or completion).
    """

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        return len(state["responses"]) > 0

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        # never called in MultiTurnEnv.rollout
        return [{"role": "user", "content": ""}], state

    def attach_client(self, client, model: str | None = None):
        """Attach a remote inference client for policy generation."""
        self.client = client  # type: ignore
        if model is not None:
            self.model = model  # type: ignore
