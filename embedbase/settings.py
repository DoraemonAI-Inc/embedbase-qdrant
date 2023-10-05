from enum import Enum
from functools import lru_cache
import typing
import os
# from pydantic_yaml import YamlModel
from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as


class VectorDatabaseEnum(str, Enum):
    pinecone = "pinecone"
    supabase = "supabase"
    weaviate = "weaviate"
    postgres = "postgres"


# an enum to pick from openai or cohere
class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"


class Settings(BaseModel):
    # default as OpenAI, choices are open_ai, cohere or azure(Azure OpenAI API)
    api_type: str = "open_ai"

    # OpenAI api settings
    openai_api_key: typing.Optional[str] = None
    openai_organization: typing.Optional[str] = None

    # Azure openai api settings
    azure_api_key:  typing.Optional[str] = None
    azure_api_base:  typing.Optional[str] = None
    azure_api_version:  typing.Optional[str] = None
    azure_deployment_id: typing.Optional[str] = None

    # supabase settings
    supabase_url: typing.Optional[str] = None
    supabase_key: typing.Optional[str] = None

    # qdrant settings
    qdrant_url: typing.Optional[str] = None
    qdrant_api_key: typing.Optional[str] = None

    # logging settings
    log_level: str = "INFO"
    auth: typing.Optional[str] = None
    firebase_service_account_path: typing.Optional[str] = None

@lru_cache()
def get_settings_from_file(path: str = "config.yaml"):
    """
    Read settings from a file, only supports yaml for now
    """
    # settings = Settings.parse_file(path)
    settings = parse_yaml_file_as(Settings, path)

    # TODO: move
    # if firebase, init firebase
    if settings.auth and settings.auth == "firebase":
        import firebase_admin
        from firebase_admin import credentials

        cred = credentials.Certificate(settings.firebase_service_account_path)
        firebase_admin.initialize_app(cred)
    return settings
