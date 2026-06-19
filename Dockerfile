FROM python:3.11-slim

WORKDIR /app

COPY src/ ./src/

CMD ["python", "-c", "import src.reporag; print('API started')"]
