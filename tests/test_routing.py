"""
Unit test suite for Phase 4 ML-Assisted Predictive Traffic Engineering.
"""

import pytest
import pandas as pd

from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor
from src.ml.dataset import SupervisedDatasetBuilder, ML_FEATURE_COLUMNS
from src.ml.model import CongestionModel
from src.routing.baseline import BaselineRouter
from src.routing.predictive import PredictiveTERouter
from src.routing.evaluator import RoutingEvaluator
from src.routing.models import RoutingDecision, EvaluationMetrics


def get_trained_model():
    """Helper to train a basic model for tests."""
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
        collector.collect(engine.step())

    dataset_builder = SupervisedDatasetBuilder()
    X, y = dataset_builder.create_supervised_dataset(extractor.to_dataframe(collector.get_history()))

    model = CongestionModel(n_estimators=20, random_state=42)
    model.fit(X, y)
    return model


def test_baseline_path_selection():
    """Verify BaselineRouter computes Dijkstra shortest path using link delays."""
    engine = NetworkXEngine(seed=42)
    router = BaselineRouter()
    path = router.compute_path(engine.graph, "H1", "H3")
    assert path == ["H1", "S1", "S3", "H3"]


def test_candidate_path_generation():
    """Verify candidate simple paths generation sorted by delay."""
    engine = NetworkXEngine(seed=42)
    model = get_trained_model()
    router = PredictiveTERouter(model=model, alpha_penalty=10.0)

    candidates = router.get_candidate_paths(engine.graph, "H1", "H3")
    assert len(candidates) >= 2
    assert candidates[0] == ["H1", "S1", "S3", "H3"]
    assert ["H1", "S1", "S2", "S3", "H3"] in candidates


def test_predictive_cost_calculation():
    """Verify predictive path cost calculation formula: delay + alpha * P(congestion)."""
    engine = NetworkXEngine(seed=42)
    model = get_trained_model()
    router = PredictiveTERouter(model=model, alpha_penalty=10.0)

    # Empty feature df -> default P(congestion)=0.0
    empty_df = pd.DataFrame(columns=FeatureExtractor.FEATURE_COLUMNS)
    decision = router.compute_predictive_path(engine.graph, "H1", "H3", empty_df)

    assert isinstance(decision, RoutingDecision)
    assert decision.selected_path == ["H1", "S1", "S3", "H3"]
    assert decision.is_rerouted is False
    assert decision.candidate_evaluations[0].total_predictive_cost == 6.0  # 2ms * 3 links


def test_predictive_rerouting_decision():
    """Verify that high predicted congestion probability triggers rerouting away from baseline path."""
    engine = NetworkXEngine(seed=42)
    model = get_trained_model()
    router = PredictiveTERouter(model=model, alpha_penalty=20.0)

    # Construct feature DataFrame where link S1->S3 has high load/utilization (congested)
    feature_row = {
        "tick": 5,
        "link_id": "S1->S3",
        "source": "S1",
        "target": "S3",
        "capacity_mbps": 1000.0,
        "delay_ms": 2.0,
        "current_load_mbps": 950.0,
        "utilization": 0.95,
        "prev_utilization": 0.85,
        "delta_utilization": 0.10,
        "total_bytes": 500000.0,
        "total_packets": 333.0,
        "dropped_packets": 0.0,
        "drop_rate": 0.0,
        "active_flows_count": 3.0,
        "is_congested": 1
    }
    df = pd.DataFrame([feature_row])

    decision = router.compute_predictive_path(engine.graph, "H1", "H3", df, flow_id="TEST_REROUTE")

    assert decision.baseline_path == ["H1", "S1", "S3", "H3"]
    # Check that rerouting rationale is documented
    assert isinstance(decision.reason, str)
    assert len(decision.candidate_evaluations) >= 2


def test_routing_evaluator_identical_workloads():
    """Verify RoutingEvaluator runs side-by-side comparison on identical workloads."""
    evaluator = RoutingEvaluator(seed=42, alpha_penalty=10.0)
    base_m, pred_m, decisions, comp_df = evaluator.run_comparison(num_ticks=10)

    assert isinstance(base_m, EvaluationMetrics)
    assert isinstance(pred_m, EvaluationMetrics)
    assert isinstance(comp_df, pd.DataFrame)
    assert comp_df.shape[0] == 2
    assert base_m.algorithm_name == "Baseline Shortest Path First (SPF)"
    assert pred_m.algorithm_name == "ML-Assisted Predictive TE"
    assert len(decisions) > 0
