import os
import uvicorn
from embedbase import get_app
from embedbase.embedding.openai import OpenAI
from embedbase.database.qdrant_db import Qdrant
from embedbase.settings import get_settings_from_file
from embedbase.database.memory_db import MemoryDatabase

settings = get_settings_from_file()

# here we use openai to create embeddings and qdrant to store the data
# app = get_app().use_embedder(OpenAI(os.environ["OPENAI_API_KEY"])).use_db(Qdrant()).run()
app = get_app().use_embedder(OpenAI(settings)).use_db(MemoryDatabase()).run()

if __name__ == "__main__":
    uvicorn.run(app)