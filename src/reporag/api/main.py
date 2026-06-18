"""FastAPI application entrypoint.

Configures the FastAPI app with routes, middleware, CORS, and lifespan
events. Run with: uvicorn src.reporag.api.main:app --reload
"""

# TODO: Implement in Issue 26
# - Create FastAPI app with metadata (title, version, description)
# - Include routers: repos, query, health, auth
# - Register middleware: CORS, auth, rate limiter, logging, error handler
# - Lifespan events: connect to Neo4j + Qdrant on startup, close on shutdown
# - OpenAPI docs at /docs


"""FastAPI application entrypoint.

Temporary placeholder app for infrastructure validation.
Full implementation will be added in Issue 26.
"""

from fastapi import FastAPI

app = FastAPI(
    title="RepoRAG AI",
    version="0.1.0",
    description="Temporary scaffold app for Issue 1",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}