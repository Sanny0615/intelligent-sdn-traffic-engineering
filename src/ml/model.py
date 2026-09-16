"""
Random Forest ML model wrapper for future link congestion prediction.
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


class CongestionModel:
    """
    Random Forest Classifier for predicting future link congestion risk (t+1).
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: Optional[int] = 5,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits the Random Forest model on training data."""
        if X_train.empty or len(y_train) == 0:
            raise ValueError("Training dataset is empty.")
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary future congestion class (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for congestion."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call fit() before predict_proba().")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates model predictions on test set using actual empirical sklearn metrics.
        Returns accuracy, precision, recall, f1, confusion matrix, and roc_auc (if valid).
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call fit() before evaluate().")

        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Calculate ROC-AUC if both classes (0 and 1) exist in test set
        unique_classes = np.unique(y_test)
        roc_auc: Optional[float] = None
        if len(unique_classes) == 2:
            y_proba = self.predict_proba(X_test)
            # Find index of positive class (1)
            pos_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else 1
            if y_proba.shape[1] > pos_idx:
                roc_auc = float(roc_auc_score(y_test, y_proba[:, pos_idx]))

        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
            "test_sample_count": len(y_test),
            "test_class_distribution": dict(pd.Series(y_test).value_counts().to_dict())
        }

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Returns feature importances mapped to feature names, sorted descending.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained.")
        importances = self.model.feature_importances_
        mapped = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}
        sorted_mapped = dict(sorted(mapped.items(), key=lambda item: item[1], reverse=True))
        return sorted_mapped

    def save(self, filepath: str) -> None:
        """Saves model instance to file via joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CongestionModel":
        """Loads model instance from file via joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at {filepath}")
        return joblib.load(filepath)
