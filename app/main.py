from fastapi import FastAPI, HTTPException, Depends
from app.schemas import (
    RiskPredictionRequest, RiskPredictionResponse,
    ReturnPredictionRequest, ReturnPredictionResponse,
    RecommendationRequest, RecommendationResponse
)
from app.dependencies import get_models
from app.services import PredictionService
from ml.models import load_latest_metrics
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Plain ASCII on purpose: an emoji here crashes startup when stdout is a
    # Windows pipe or log file (cp1252 cannot encode it).
    print("RiskGuard AI is running.")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Stock Risk Forecasting API",
    description="End-to-End MLOps Project for Stock Risk Classification and Forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Mount Static Files (Frontend)
# Ensure directory exists to prevent startup error
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_root():
    return RedirectResponse(url="/static/index.html")

@app.post("/predict_risk", response_model=RiskPredictionResponse)
def predict_risk(request: RiskPredictionRequest, models = Depends(get_models)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    service = PredictionService(models)
    try:
        result = service.predict_risk(request.ticker)
        return {
            "ticker": request.ticker,
            "risk_class": result["risk_class"],
            "probabilities": result["probabilities"],
            "volatility": result["volatility"],
            "confidence_score": result["confidence_score"],
            "recommendation": result["recommendation"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_return", response_model=ReturnPredictionResponse)
def predict_return(request: ReturnPredictionRequest, models = Depends(get_models)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    service = PredictionService(models)
    try:
        pred = service.predict_return(request.ticker)
        return {
            "ticker": request.ticker,
            "predicted_next_day_return": pred
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend_similar") # GET for simpler query
def recommend_similar(ticker: str, risk_preference: str = None, models = Depends(get_models)):
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    service = PredictionService(models)
    try:
        recs = service.recommend_similar(ticker, risk_preference)
        return {
            "input_ticker": ticker,
            "recommendations": recs
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Evaluation results of the model being served (saved next to it at training time)."""
    metrics = load_latest_metrics()
    if metrics is None:
        raise HTTPException(status_code=404, detail="No metrics found")
    return metrics


@app.get("/health")
def health(models = Depends(get_models)):
    return {"status": "ok", "model_loaded": bool(models)}
