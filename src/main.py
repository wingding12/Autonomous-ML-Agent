"""
Main FastAPI application for the Autonomous ML Agent.

This module serves as the entry point for the FastAPI backend service,
providing REST API endpoints for pipeline management, model training,
and inference services.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import get_settings
from .api.routes import pipeline, models, inference
from .core.agent import AutonomousMLAgent
from .utils.logger import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Global agent instance
agent: AutonomousMLAgent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global agent
    
    # Startup
    logger.info("Starting Autonomous ML Agent...")
    try:
        settings = get_settings()
        agent = AutonomousMLAgent(settings)
        await agent.initialize()
        logger.info("Autonomous ML Agent started successfully")
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Autonomous ML Agent...")
    if agent:
        await agent.cleanup()
    logger.info("Autonomous ML Agent shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Autonomous ML Agent",
    description="An intelligent, end-to-end machine learning pipeline orchestrated by LLMs",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(pipeline.router, prefix="/api/v1/pipeline", tags=["pipeline"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["inference"])

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint providing API information."""
    return {
        "message": "Autonomous ML Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    global agent
    
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        agent_status = await agent.get_status()
        return {
            "status": "healthy",
            "agent": agent_status,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

def main():
    """Main function to run the FastAPI application."""
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=settings.API_WORKERS if not settings.DEBUG else 1
    )

if __name__ == "__main__":
    main()
