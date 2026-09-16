"""
Unit test suite for Phase 3 ML-Based Future Congestion Prediction.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor
from src.ml.dataset import SupervisedDatasetBuilder, ML_FEATURE_COLUMNS, TARGET_COLUMN
from src.ml.model import CongestionModel
from src.ml.predictor import predict_link_congestion


def generate_test_df():
    """Helper function to generate a multi-tick feature DataFrame for tests."""
    engine = NetworkXEngine(seed=42)
    collector = TelemetryCollector()
    extractor = FeatureExtractor()
    traffic_gen = DeterministicTrafficGenerator(seed=42)

    flows = traffic_gen.generate_dynamic_time_series_flows(num_ticks=15)
    flows_by_tick = {}
    for f in flows:
        flows_by_tick.setdefault(f.start_tick, []).append(f)

    for tick in range(15):
        if tick in flows_by_tick:
            for f in flows_by_tick[tick]:
                engine.add_flow(f)
        snapshot = engine.step()
        collector.collect(snapshot)

    return extractor.to_dataframe(collector.get_history())


def test_supervised_dataset_construction_and_leakage_prevention():
    """Verify t -> t+1 target shifting, row reduction, and data leakage prevention."""
    raw_df = generate_test_df()
    builder = SupervisedDatasetBuilder()
    X, y = builder.create_supervised_dataset(raw_df)

    # Check shape reduction (dropped last tick for each of the 12 links)
    assert len(X) == len(raw_df) - 12
    assert len(y) == len(X)

    # Check feature matrix X columns
    assert list(X.columns) == ML_FEATURE_COLUMNS

    # Ensure no data leakage (identifiers and targets are NOT in X)
    forbidden_cols = {"tick", "link_id", "source", "target", "is_congested", TARGET_COLUMN}
    assert forbidden_cols.intersection(set(X.columns)) == set()


def test_dataset_both_class_validation():
    """Verify that dynamic traffic simulation produces both congested (1) and non-congested (0) target classes."""
    raw_df = generate_test_df()
    builder = SupervisedDatasetBuilder()
    _, y = builder.create_supervised_dataset(raw_df)

    unique_classes = set(y.unique())
    assert unique_classes == {0, 1}, f"Expected both classes {{0, 1}}, got {unique_classes}"


def test_random_forest_training_and_reproducibility():
    """Verify RandomForestClassifier fitting, binary predictions, and seed reproducibility."""
    raw_df = generate_test_df()
    builder = SupervisedDatasetBuilder()
    X, y = builder.create_supervised_dataset(raw_df)

    model1 = CongestionModel(n_estimators=20, random_state=42)
    model1.fit(X, y)
    preds1 = model1.predict(X)

    assert len(preds1) == len(X)
    assert set(np.unique(preds1)).issubset({0, 1})

    # Test reproducibility with identical seed
    model2 = CongestionModel(n_estimators=20, random_state=42)
    model2.fit(X, y)
    preds2 = model2.predict(X)

    assert np.array_equal(preds1, preds2)


def test_eval_metrics_and_feature_importances():
    """Verify evaluate() metrics structure and feature importances format."""
    raw_df = generate_test_df()
    builder = SupervisedDatasetBuilder()
    X, y = builder.create_supervised_dataset(raw_df)

    model = CongestionModel(n_estimators=20, random_state=42)
    model.fit(X, y)

    metrics = model.evaluate(X, y)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    importances = model.get_feature_importances(ML_FEATURE_COLUMNS)
    assert len(importances) == len(ML_FEATURE_COLUMNS)
    assert pytest.approx(sum(importances.values()), abs=1e-3) == 1.0


def test_prediction_api_interface():
    """Verify predict_link_congestion API wrapper."""
    raw_df = generate_test_df()
    builder = SupervisedDatasetBuilder()
    X, y = builder.create_supervised_dataset(raw_df)

    model = CongestionModel(n_estimators=20, random_state=42)
    model.fit(X, y)

    sample_features = {
        "utilization": 0.90,
        "prev_utilization": 0.75,
        "delta_utilization": 0.15,
        "current_load_mbps": 900.0,
        "capacity_mbps": 1000.0,
        "delay_ms": 2.0,
        "drop_rate": 0.0,
        "active_flows_count": 2.0
    }

    result = predict_link_congestion(model, sample_features)
    assert "predicted_congestion_class" in result
    assert "congestion_probability" in result
    assert result["predicted_congestion_class"] in [0, 1]
    assert 0.0 <= result["congestion_probability"] <= 1.0


def test_model_persistence(tmp_path):
    """Verify saving and loading model artifacts using joblib."""
    raw_df = generate_test_df()
    builder = SupervisedDatasetBuilder()
    X, y = builder.create_supervised_dataset(raw_df)

    model = CongestionModel(n_estimators=20, random_state=42)
    model.fit(X, y)
    preds_before = model.predict(X)

    save_path = str(tmp_path / "congestion_rf.joblib")
    model.save(save_path)

    assert os.path.exists(save_path)

    loaded_model = CongestionModel.load(save_path)
    preds_after = loaded_model.predict(X)

    assert np.array_equal(preds_before, preds_after)
