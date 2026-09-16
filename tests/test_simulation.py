"""
Unit test suite for Phase 1 Deterministic Network Simulation Foundation.
"""

import pytest
import networkx as nx

from src.simulation.models import TrafficFlow, Link
from src.simulation.topology import build_custom_6node_topology
from src.simulation.traffic import DeterministicTrafficGenerator
from src.simulation.engine import NetworkXEngine


def test_topology_structure():
    """Verify that custom 6-node topology builds correctly with nodes and directed edges."""
    graph = build_custom_6node_topology()
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() == 6
    assert graph.number_of_edges() == 12

    expected_nodes = {"H1", "H2", "S1", "S2", "S3", "H3"}
    assert set(graph.nodes()) == expected_nodes

    # Check node types
    assert graph.nodes["H1"]["node_type"] == "host"
    assert graph.nodes["S1"]["node_type"] == "switch"
    assert graph.nodes["S3"]["node_type"] == "switch"

    # Check edge attributes
    edge_data = graph["H1"]["S1"]
    assert "capacity" in edge_data
    assert "delay" in edge_data
    assert "object" in edge_data
    assert isinstance(edge_data["object"], Link)


def test_traffic_generator_determinism():
    """Verify that DeterministicTrafficGenerator produces identical flows with the same seed."""
    gen1 = DeterministicTrafficGenerator(seed=42)
    gen2 = DeterministicTrafficGenerator(seed=42)

    flows1 = gen1.generate_random_flows(count=5)
    flows2 = gen2.generate_random_flows(count=5)

    assert len(flows1) == len(flows2) == 5
    for f1, f2 in zip(flows1, flows2):
        assert f1.source == f2.source
        assert f1.destination == f2.destination
        assert f1.demand_mbps == f2.demand_mbps


def test_shortest_path_calculation():
    """Verify Dijkstra shortest path routing computation in NetworkXEngine."""
    engine = NetworkXEngine(seed=42)
    path = engine.compute_shortest_path("H1", "H3")
    # Path should be H1 -> S1 -> S3 -> H3 (delay: 2.0 + 2.0 + 2.0 = 6.0 ms vs S1->S2->S3 delay: 2.0 + 3.0 + 3.0 = 8.0 ms)
    assert path == ["H1", "S1", "S3", "H3"]


def test_flow_addition_and_removal():
    """Verify registration, path assignment, and removal of traffic flows."""
    engine = NetworkXEngine(seed=42)
    flow = TrafficFlow(
        flow_id="test_flow_1",
        source="H1",
        destination="H3",
        demand_mbps=300.0
    )
    path = engine.add_flow(flow)
    assert path == ["H1", "S1", "S3", "H3"]
    assert "test_flow_1" in engine.active_flows

    # Step simulation and check link load
    snapshot = engine.step()
    assert snapshot.active_flows_count == 1
    assert snapshot.links_telemetry["H1->S1"]["current_load_mbps"] == 300.0
    assert snapshot.links_telemetry["S1->S3"]["current_load_mbps"] == 300.0

    # Remove flow
    removed = engine.remove_flow("test_flow_1")
    assert removed is True
    assert "test_flow_1" not in engine.active_flows

    snapshot2 = engine.step()
    assert snapshot2.active_flows_count == 0
    assert snapshot2.links_telemetry["H1->S1"]["current_load_mbps"] == 0.0


def test_capacity_saturation_and_drop_tracking():
    """Verify packet drop counter increments when link capacity is exceeded."""
    engine = NetworkXEngine(seed=42)
    # Add flow exceeding default 1000 Mbps capacity
    overload_flow = TrafficFlow(
        flow_id="overload_1",
        source="H1",
        destination="H3",
        demand_mbps=1500.0
    )
    engine.add_flow(overload_flow)

    snapshot = engine.step()
    assert snapshot.links_telemetry["H1->S1"]["current_load_mbps"] == 1500.0
    assert snapshot.links_telemetry["H1->S1"]["is_congested"] == 1.0
    assert snapshot.total_dropped_packets > 0


def test_engine_reset():
    """Verify resetting engine clears tick counter and active flows."""
    engine = NetworkXEngine(seed=42)
    flow = TrafficFlow(flow_id="f1", source="H1", destination="H3", demand_mbps=200.0)
    engine.add_flow(flow)
    engine.step()
    assert engine.current_tick == 1

    engine.reset(seed=42)
    assert engine.current_tick == 0
    assert len(engine.active_flows) == 0
