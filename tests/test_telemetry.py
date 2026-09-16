"""
Unit test suite for Phase 2 Telemetry Collection and Feature Engineering.
"""

import pytest
import pandas as pd

from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor


def test_collector_buffering():
    """Verify TelemetryCollector buffers snapshots correctly."""
    engine = NetworkXEngine(seed=42)
    collector = TelemetryCollector(max_history_size=5)

    assert len(collector.get_history()) == 0
    assert collector.get_latest_snapshot() is None

    for _ in range(3):
        snapshot = engine.step()
        collector.collect(snapshot)

    assert len(collector.get_history()) == 3
    assert collector.get_latest_snapshot().timestamp_tick == 3

    collector.reset()
    assert len(collector.get_history()) == 0


def test_feature_extraction_schema():
    """Verify FeatureExtractor produces correct columns and DataFrame structure."""
    engine = NetworkXEngine(seed=42)
    collector = TelemetryCollector()
    extractor = FeatureExtractor()

    snapshot = engine.step()
    collector.collect(snapshot)

    df = extractor.to_dataframe(collector.get_history())

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 12  # 12 directed links in topology
    assert list(df.columns) == FeatureExtractor.FEATURE_COLUMNS

    # Check required data types
    assert df["tick"].dtype in ["int64", "int32"]
    assert df["utilization"].dtype in ["float64", "float32"]
    assert df["is_congested"].dtype in ["int64", "int32"]


def test_time_series_lag_features():
    """Verify prev_utilization and delta_utilization calculations across ticks."""
    engine = NetworkXEngine(seed=42)
    collector = TelemetryCollector()
    extractor = FeatureExtractor()

    # Add flow on H1 -> H3
    traffic_gen = DeterministicTrafficGenerator(seed=42)
    flow = traffic_gen.generate_fixed_flow_set()[0]  # Demand: 600 Mbps
    engine.add_flow(flow)

    # Tick 1
    s1 = engine.step()
    collector.collect(s1)

    # Tick 2: Increase flow demand to 900 Mbps
    flow.demand_mbps = 900.0
    s2 = engine.step()
    collector.collect(s2)

    df = extractor.to_dataframe(collector.get_history())

    # Filter for link H1->S1
    h1_s1_df = df[df["link_id"] == "H1->S1"].sort_values("tick")
    assert len(h1_s1_df) == 2

    row1 = h1_s1_df.iloc[0]
    row2 = h1_s1_df.iloc[1]

    # Tick 1: prev_util should be 0.0, delta should equal utilization
    assert row1["tick"] == 1
    assert row1["utilization"] == 0.6
    assert row1["prev_utilization"] == 0.0
    assert row1["delta_utilization"] == 0.6

    # Tick 2: prev_util should equal tick 1 utilization (0.6), delta should be 0.9 - 0.6 = 0.3
    assert row2["tick"] == 2
    assert row2["utilization"] == 0.9
    assert row2["prev_utilization"] == 0.6
    assert round(row2["delta_utilization"], 4) == 0.3


def test_drop_rate_and_congestion_flags():
    """Verify drop rate and congestion flag logic."""
    engine = NetworkXEngine(seed=42)
    collector = TelemetryCollector()
    extractor = FeatureExtractor()

    # Create an overloading flow demand (1500 Mbps over 1000 Mbps capacity)
    traffic_gen = DeterministicTrafficGenerator(seed=42)
    overload_flow = traffic_gen.generate_random_flows(
        count=1, host_nodes=["H1", "H3"], min_demand_mbps=1500.0, max_demand_mbps=1500.0
    )[0]
    engine.add_flow(overload_flow)

    s1 = engine.step()
    collector.collect(s1)

    df = extractor.to_dataframe(collector.get_history())
    congested_link = df[df["link_id"] == "H1->S1"].iloc[0]

    assert congested_link["utilization"] == 1.5
    assert congested_link["is_congested"] == 1
    assert congested_link["dropped_packets"] > 0
    assert congested_link["drop_rate"] > 0.0


def test_feature_dataframe_determinism():
    """Verify that repeated runs with seed=42 yield identical DataFrames."""
    def run_sim():
        engine = NetworkXEngine(seed=42)
        collector = TelemetryCollector()
        extractor = FeatureExtractor()
        traffic_gen = DeterministicTrafficGenerator(seed=42)

        for flow in traffic_gen.generate_fixed_flow_set():
            engine.add_flow(flow)

        for _ in range(3):
            collector.collect(engine.step())

        return extractor.to_dataframe(collector.get_history())

    df1 = run_sim()
    df2 = run_sim()

    assert df1.equals(df2)
