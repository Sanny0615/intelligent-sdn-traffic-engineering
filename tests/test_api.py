"""
Unit test suite for Phase 6 FastAPI Backend API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /api/v1/health returns status ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["simulation"] == "active"


def test_topology_endpoint():
    """Verify GET /api/v1/topology returns network nodes and links."""
    response = client.get("/api/v1/topology")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["nodes_count"] == 6
    assert data["edges_count"] == 12
    assert len(data["nodes"]) == 6
    assert len(data["links"]) == 12


def test_telemetry_endpoint():
    """Verify GET /api/v1/telemetry returns link metrics snapshot."""
    response = client.get("/api/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp_tick" in data
    assert "total_throughput_mbps" in data
    assert len(data["links"]) == 12


def test_prediction_endpoint():
    """Verify POST /api/v1/predict/congestion predicts future link congestion."""
    payload = {
        "link_id": "S1->S3",
        "utilization": 0.88,
        "prev_utilization": 0.65,
        "delta_utilization": 0.23,
        "current_load_mbps": 880.0,
        "capacity_mbps": 1000.0,
        "delay_ms": 2.0,
        "drop_rate": 0.0,
        "active_flows_count": 3.0
    }
    response = client.post("/api/v1/predict/congestion", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["link_id"] == "S1->S3"
    assert data["predicted_congestion_class"] in [0, 1]
    assert 0.0 <= data["congestion_probability"] <= 1.0


def test_routing_optimize_baseline():
    """Verify POST /api/v1/routing/optimize with baseline strategy."""
    payload = {
        "source": "H1",
        "destination": "H3",
        "demand_mbps": 400.0,
        "strategy": "baseline"
    }
    response = client.post("/api/v1/routing/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "baseline"
    assert data["selected_path"] == ["H1", "S1", "S3", "H3"]


def test_routing_optimize_predictive():
    """Verify POST /api/v1/routing/optimize with predictive strategy."""
    payload = {
        "source": "H1",
        "destination": "H3",
        "demand_mbps": 400.0,
        "strategy": "predictive",
        "alpha_penalty": 10.0
    }
    response = client.post("/api/v1/routing/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "predictive"
    assert "selected_path" in data
    assert "reason" in data


def test_routing_invalid_node_error():
    """Verify POST /api/v1/routing/optimize returns 400 on invalid node."""
    payload = {
        "source": "INVALID_NODE",
        "destination": "H3",
        "demand_mbps": 400.0,
        "strategy": "predictive"
    }
    response = client.post("/api/v1/routing/optimize", json=payload)
    assert response.status_code == 400
    assert "Source node 'INVALID_NODE' does not exist" in response.json()["detail"]


def test_routing_invalid_strategy_error():
    """Verify POST /api/v1/routing/optimize returns 422 on invalid strategy."""
    payload = {
        "source": "H1",
        "destination": "H3",
        "demand_mbps": 400.0,
        "strategy": "invalid_strategy"
    }
    response = client.post("/api/v1/routing/optimize", json=payload)
    assert response.status_code == 422


def test_what_if_scenario_endpoint():
    """Verify POST /api/v1/simulation/what-if executes scenario on Digital Twin."""
    payload = {
        "scenario_name": "API_Test_Surge",
        "scenario_type": "traffic_demand_increase",
        "target_id": "F1_H1_H3",
        "parameter_name": "demand_mbps",
        "value": 0.50,
        "routing_strategy": "predictive",
        "num_ticks": 5
    }
    response = client.post("/api/v1/simulation/what-if", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_name"] == "API_Test_Surge"
    assert data["routing_strategy"] == "predictive"
    assert "throughput_delta_mbps" in data


def test_what_if_invalid_scenario_type():
    """Verify POST /api/v1/simulation/what-if returns 422 on invalid scenario type."""
    payload = {
        "scenario_name": "Bad_Scenario",
        "scenario_type": "invalid_type",
        "target_id": "S1->S3",
        "parameter_name": "bandwidth",
        "value": 100.0
    }
    response = client.post("/api/v1/simulation/what-if", json=payload)
    assert response.status_code == 422
