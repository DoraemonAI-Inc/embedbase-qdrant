import os
import uvicorn
from embedbase import get_app
from embedbase.embedding.openai import OpenAI
from embedbase.database.qdrant_db import Qdrant
from embedbase.settings import Settings

settings = Settings()

settings.api_type = os.environ.get("API_TYPE")

settings.openai_api_key = os.environ.get("OPENAI_API_KEY")
settings.openai_organization = os.environ.get("OPENAI_ORGANIZATION")

settings.azure_api_key = os.environ.get("AZURE_API_KEY")
settings.azure_api_base = os.environ.get("AZURE_API_BASE")
settings.azure_api_version = os.environ.get("AZURE_API_VERSION")
settings.azure_deployment_id = os.environ.get("AZURE_DEPLOYMENT_ID")

settings.supabase_url = os.environ.get("SUPABASE_URL")
settings.supabase_key = os.environ.get("SUPABASE_KEY")

settings.qdrant_url = os.environ.get("QDRANT_URL")
settings.qdrant_api_key = os.environ.get("QDRANT_API_KEY")

# here we use openai to create embeddings and qdrant to store the data
app = get_app().use_embedder(OpenAI(settings)).use_db(Qdrant(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key
)).run()

if __name__ == "__main__":
    uvicorn.run(app)
