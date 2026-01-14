"""
SmartJal FastAPI Application

Main entry point for the groundwater prediction API.
"""
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .models.schemas import HealthResponse
from .api.routes import predictions, villages, aquifers, analytics, geojson

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    'data_loader': None,
    'predictor': None,
    'model_loaded': False,
    'data_loaded': False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting SmartJal API...")

    # Initialize data loader
    try:
        from ml.data.loader import DataLoader
        app_state['data_loader'] = DataLoader()
        # Pre-load some data
        app_state['data_loader'].load_aquifers()
        app_state['data_loaded'] = True
        logger.info("Data loader initialized")
    except Exception as e:
        logger.error(f"Failed to initialize data loader: {e}")

    # Initialize predictor (try to load pre-trained model)
    try:
        from ml.models.ensemble import GroundwaterPredictor
        models_path = settings.MODELS_DIR
        if models_path.exists() and (models_path / 'ensemble_weights.json').exists():
            app_state['predictor'] = GroundwaterPredictor(models_path)
            app_state['predictor'].load()
            app_state['model_loaded'] = True
            logger.info("Loaded pre-trained model")
        else:
            logger.info("No pre-trained model found. Model training required.")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")

    yield

    # Cleanup
    logger.info("Shutting down SmartJal API...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    SmartJal API - AI-driven Groundwater Level Prediction System

    This API provides:
    - Village-level groundwater predictions
    - Aquifer-based analysis
    - Temporal forecasting
    - Recharge zone identification
    - Analytics and visualizations

    For Krishna District, Andhra Pradesh.
    """,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=app_state['model_loaded'],
        data_loaded=app_state['data_loaded'],
        timestamp=datetime.utcnow()
    )


@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(
    predictions.router,
    prefix=f"{settings.API_V1_PREFIX}/predictions",
    tags=["Predictions"]
)

app.include_router(
    villages.router,
    prefix=f"{settings.API_V1_PREFIX}/villages",
    tags=["Villages"]
)

app.include_router(
    aquifers.router,
    prefix=f"{settings.API_V1_PREFIX}/aquifers",
    tags=["Aquifers"]
)

app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_PREFIX}/analytics",
    tags=["Analytics"]
)

app.include_router(
    geojson.router,
    prefix=f"{settings.API_V1_PREFIX}/geojson",
    tags=["GeoJSON"]
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def get_app_state():
    """Get application state (for dependency injection)"""
    return app_state


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
