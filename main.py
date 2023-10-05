import os
import uvicorn
from embedbase import get_app
from embedbase.embedding.openai import OpenAI
from embedbase.database.qdrant_db import Qdrant

# here we use openai to create embeddings and qdrant to store the data
app = get_app().use_embedder(OpenAI(os.environ["OPENAI_API_KEY"])).use_db(Qdrant()).run()

if __name__ == "__main__":
    uvicorn.run(app)