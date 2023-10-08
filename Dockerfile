FROM python:3.10-slim
RUN apt-get update && apt-get install -y git gcc && apt-get clean && \
    pip install openai uvicorn fastapi pandas pydantic-yaml pydantic tenacity qdrant-client tiktoken

COPY main.py main.py
COPY embedbase embedbase

CMD ["python3", "main.py"]