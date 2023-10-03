FROM python:3.10-slim
RUN apt-get update && apt-get install -y git gcc && apt-get clean && \
    pip install openai uvicorn fastapi pandas pydantic-yaml

COPY main.py main.py

CMD ["embedbase"]