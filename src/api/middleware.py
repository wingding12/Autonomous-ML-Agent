"""
API Middleware for the Autonomous ML Agent.

This module provides middleware functionality for the FastAPI
application including authentication, logging, and error handling.
"""

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import logging
import time

logger = logging.getLogger(__name__)

def setup_middleware(app):
    """Setup middleware for the FastAPI application."""
    app.add_middleware(LoggingMiddleware)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
        
        return response
