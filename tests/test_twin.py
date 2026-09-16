"""
Unit test suite for Phase 5 What-If Network Digital Twin / Scenario Simulation.
"""

import pytest
import pandas as pd

from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.simulation.models import TrafficFlow
from src.twin.models import WhatIfScenario, WhatIfExecutionResult
from src.twin.engine import create_digital_twin, WhatIfNetworkDigitalTwin
from src.twin.scenarios import apply_scenario


def test_digital_twin_creation_and_deep_copy_isolation():
    """Verify deep-copy isolation between live simulation engine and digital twin."""
    live_engine = NetworkXEngine(seed=42)
    traffic_gen = DeterministicTrafficGenerator(seed=42)
    for f in traffic_gen.generate_fixed_flow_set():
        live_engine.add_flow(f)
    live_engine.step()

    live_tick_before = live_engine.current_tick
    live_flows_count = len(live_engine.active_flows)
    orig_capacity = live_engine.graph["S1"]["S3"]["capacity"]

    # Create twin
    twin_engine = create_digital_twin(live_engine)

    # Mutate twin state
    twin_engine.step()
    twin_engine.add_flow(TrafficFlow("test_twin_flow", "H1", "H2", 300.0))
    twin_engine.graph["S1"]["S3"]["capacity"] = 200.0

    # Verify live engine is 100% UNTOUCHED
    assert live_engine.current_tick == live_tick_before
    assert len(live_engine.active_flows) == live_flows_count
    assert live_engine.graph["S1"]["S3"]["capacity"] == orig_capacity
    assert "test_twin_flow" not in live_engine.active_flows


def test_scenario_traffic_demand_increase():
    """Verify traffic demand increase scenario on twin."""
    engine = NetworkXEngine(seed=42)
    flow = TrafficFlow("F1", "H1", "H3", 400.0)
    engine.add_flow(flow)

    scenario = WhatIfScenario("Surge", "traffic_demand_increase", "F1", "demand_mbps", 0.50)
    apply_scenario(engine, scenario)

    assert engine.active_flows["F1"].demand_mbps == 600.0  # 400 * 1.5


def test_scenario_add_flow():
    """Verify add flow scenario on twin."""
    engine = NetworkXEngine(seed=42)
    scenario = WhatIfScenario("NewFlow", "add_flow", "H2_H3", "flow", {
        "source": "H2", "destination": "H3", "demand_mbps": 500.0
    })
    apply_scenario(engine, scenario)

    assert len(engine.active_flows) == 1
    assert "whatif_flow_H2_H3" in engine.active_flows


def test_scenario_link_capacity_reduction():
    """Verify link capacity reduction scenario on twin."""
    engine = NetworkXEngine(seed=42)
    orig_cap = engine.graph["S1"]["S3"]["capacity"]

    scenario = WhatIfScenario("CapacityCut", "link_capacity_reduction", "S1->S3", "capacity_mbps", 300.0)
    apply_scenario(engine, scenario)

    assert engine.graph["S1"]["S3"]["capacity"] == 300.0
    assert engine.graph["S1"]["S3"]["object"].capacity_mbps == 300.0
    assert orig_cap == 1000.0


def test_scenario_link_delay_increase():
    """Verify link delay increase scenario on twin."""
    engine = NetworkXEngine(seed=42)
    scenario = WhatIfScenario("DelaySpike", "link_delay_increase", "S1->S3", "delay_ms", 15.0)
    apply_scenario(engine, scenario)

    assert engine.graph["S1"]["S3"]["delay"] == 15.0
    assert engine.graph["S1"]["S3"]["weight"] == 15.0


def test_deterministic_scenario_execution():
    """Verify scenario execution is 100% deterministic given seed=42."""
    live_engine = NetworkXEngine(seed=42)
    traffic_gen = DeterministicTrafficGenerator(seed=42)
    for f in traffic_gen.generate_fixed_flow_set():
        live_engine.add_flow(f)
    live_engine.step()

    scenario = WhatIfScenario("Surge", "traffic_demand_increase", "F1_H1_H3", "demand_mbps", 0.50)
    twin_controller = WhatIfNetworkDigitalTwin(seed=42)

    res1 = twin_controller.execute_scenario(live_engine, scenario, routing_strategy="baseline", num_ticks=5)
    res2 = twin_controller.execute_scenario(live_engine, scenario, routing_strategy="baseline", num_ticks=5)

    assert res1.to_dict() == res2.to_dict()
