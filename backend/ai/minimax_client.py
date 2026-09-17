"""
MiniMax Client (Backwards-Compatible Wrapper around OpenAICompatibleClient)
"""

from backend.ai.openai_client import OpenAICompatibleClient


class MiniMaxClient(OpenAICompatibleClient):
    """
    Subclass of OpenAICompatibleClient pre-configured for MiniMax provider.
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider="minimax",
        )
