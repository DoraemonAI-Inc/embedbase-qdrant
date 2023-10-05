from typing import List, Union, Optional

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from embedbase.embedding.base import Embedder
from embedbase.settings import Settings

try:
    import openai
except:
    pass


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=3),
    stop=stop_after_attempt(3),
    # TODO: send pr/issue on https://github.com/openai/openai-python/blob/94428401b4f71596e4a1331102a6beee9d8f0bc4/openai/__init__.py#L25
    # To expose openai.AuthenticationError
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
)
def embed_retry(
    data: List[str],
    deployment_id: str = None,
) -> List[dict]:
    """
    Embed a list of sentences and retry on failure
    :param data: list of sentences to embed
    :param provider: which provider to use
    :return: list of embeddings
    """
    if deployment_id:
        return [
            e["embedding"]
            for e in openai.Embedding.create(
                input=data,
                deployment_id=deployment_id,
            )["data"]
        ]
    else:
        return [
            e["embedding"]
            for e in openai.Embedding.create(input=data, model="text-embedding-ada-002")[
                "data"
            ]
        ]


class OpenAI(Embedder):
    """
    OpenAI Embedder
    """

    EMBEDDING_MODEL = "text-embedding-ada-002"
    EMBEDDING_CTX_LENGTH = 8191
    EMBEDDING_ENCODING = "cl100k_base"
    deployment_id: Optional[str] = None

    def __init__(
        self, settings: Settings
    ):
        super().__init__()
        try:
            import openai
            import tiktoken
        except ImportError:
            raise ImportError(
                "OpenAI is not installed. Install it with `pip install openai tiktoken`"
            )

        self.encoding = tiktoken.get_encoding(self.EMBEDDING_ENCODING)
        if settings.api_type in ["open_ai", "OpenAI", "openai"]:
            openai.api_key = settings.openai_api_key
            openai.organization = settings.openai_organization
        elif settings.api_type in ["azure"]:
            openai.api_key = settings.azure_api_key
            openai.api_base = settings.azure_api_base
            openai.api_version = settings.azure_api_version
            self.deployment_id = settings.azure_deployment_id
            openai.api_type = "azure"
        else:
            raise ValueError(
                f"Unknown OpenAI API type {settings.api_type}. Please set api_type to open_ai or azure, if you are using Cohere, please use the CohereEmbedder"
            )

    @property
    def dimensions(self) -> int:
        return 1536

    def is_too_big(self, text: str) -> bool:
        tokens = self.encoding.encode(text)
        if len(tokens) > self.EMBEDDING_CTX_LENGTH:
            return True

        return False

    async def embed(self, data: Union[List[str], str]) -> List[List[float]]:
        return embed_retry(data, self.deployment_id)
