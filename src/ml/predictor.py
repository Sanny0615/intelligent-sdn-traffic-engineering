"""
Prediction interface module for querying future congestion predictions on current link feature vectors.
"""

from typing import Dict, Any
import pandas as pd
from src.ml.dataset import ML_FEATURE_COLUMNS
from src.ml.model import CongestionModel


def predict_link_congestion(
    model: CongestionModel,
    link_features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Accepts current link feature dictionary at tick t, formats it into a single-row DataFrame,
    and returns predicted future congestion class (0/1) and congestion probability at t+1.
    """
    if not model.is_fitted:
        raise RuntimeError("Provided CongestionModel is not trained.")

    # Format input into 1-row DataFrame matching ML_FEATURE_COLUMNS
    input_row = {col: float(link_features.get(col, 0.0)) for col in ML_FEATURE_COLUMNS}
    input_df = pd.DataFrame([input_row])

    pred_class = int(model.predict(input_df)[0])
    probas = model.predict_proba(input_df)[0]

    # Find probability of positive congestion class (1)
    pos_idx = list(model.model.classes_).index(1) if 1 in model.model.classes_ else 0
    cong_prob = float(probas[pos_idx]) if pos_idx < len(probas) else float(pred_class)

    return {
        "predicted_congestion_class": pred_class,
        "congestion_probability": round(cong_prob, 4),
        "input_features": input_row
    }
