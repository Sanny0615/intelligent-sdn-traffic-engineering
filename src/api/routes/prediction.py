"""
ML Prediction route handler.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.services import get_service_manager, APIServiceManager
from src.ml.predictor import predict_link_congestion

router = APIRouter(tags=["Prediction"])


@router.post("/predict/congestion", response_model=PredictionResponse, summary="Predict Future Link Congestion Risk (t+1)")
def predict_congestion(
    req: PredictionRequest,
    services: APIServiceManager = Depends(get_service_manager)
):
    """
    Accepts current link feature vector at tick t and returns predicted congestion class (0/1)
    and probability P(congestion=1) at t+1 using the pre-trained Phase 3 Random Forest model.
    """
    if not services.ml_model.is_fitted:
        raise HTTPException(status_code=500, detail="ML CongestionModel is not trained.")

    feature_dict = req.model_dump()
    result = predict_link_congestion(services.ml_model, feature_dict)

    return PredictionResponse(
        link_id=req.link_id,
        predicted_congestion_class=result["predicted_congestion_class"],
        congestion_probability=result["congestion_probability"],
        prediction_horizon="t+1"
    )
