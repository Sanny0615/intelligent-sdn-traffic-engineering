"""
Main FastAPI Application Entrypoint.
Exposes RESTful endpoints for network simulation, telemetry, ML prediction, TE routing, and What-If Digital Twin scenarios.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.topology import router as topology_router
from src.api.routes.telemetry import router as telemetry_router
from src.api.routes.prediction import router as prediction_router
from src.api.routes.routing import router as routing_router
from src.api.routes.scenarios import router as scenarios_router
from src.api.routes.explain import router as explain_router

app = FastAPI(
    title="Intelligent SDN Traffic Engineering API",
    description="REST API backend for simulation telemetry, ML link congestion prediction, TE routing optimization, What-If Digital Twin scenarios, and AI RAG explanations.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Streamlit frontend and local/container API clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers under /api/v1 prefix
app.include_router(health_router, prefix="/api/v1")
app.include_router(topology_router, prefix="/api/v1")
app.include_router(telemetry_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(routing_router, prefix="/api/v1")
app.include_router(scenarios_router, prefix="/api/v1")
app.include_router(explain_router, prefix="/api/v1")

# Also expose /health at root for convenience
app.include_router(health_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "title": "Intelligent SDN Traffic Engineering API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health"
    }
