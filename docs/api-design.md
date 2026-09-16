# API Design Specification

## 1. REST API Overview

The FastAPI backend exposes RESTful endpoints for controlling the network simulation, querying real-time telemetry, triggering ML predictions, executing traffic engineering optimizations, running What-If scenario simulations, and interacting with the RAG explainer.

- **Base URL**: `http://localhost:8000/api/v1`
- **Content Type**: `application/json`

---

## 2. Endpoint Specifications

### 2.1 Topology Management

#### `POST /api/v1/topology/generate`
Generates a new network topology graph.

- **Request Body** (`TopologyCreateRequest`):
```json
{
  "topology_type": "abilene",
  "num_nodes": 12,
  "default_bandwidth_mbps": 1000.0,
  "default_delay_ms": 2.0,
  "seed": 42
}
```

- **Response** (`TopologyResponse`):
```json
{
  "status": "success",
  "topology_type": "abilene",
  "nodes_count": 12,
  "edges_count": 30,
  "links": [
    {"source": "N1", "target": "N2", "capacity_mbps": 1000.0, "delay_ms": 2.0}
  ]
}
```

---

### 2.2 Telemetry & State

#### `GET /api/v1/telemetry/snapshot`
Fetches current link utilization, drop rates, and traffic state.

- **Response** (`TelemetrySnapshotResponse`):
```json
{
  "timestamp_tick": 145,
  "active_flows_count": 24,
  "total_throughput_mbps": 450.5,
  "links": [
    {
      "source": "N1",
      "target": "N2",
      "load_mbps": 750.0,
      "capacity_mbps": 1000.0,
      "utilization": 0.75,
      "queue_occupancy": 12,
      "drop_rate": 0.0
    }
  ]
}
```

---

### 2.3 ML Prediction

#### `POST /api/v1/predict/congestion`
Triggers prediction of link utilization and congestion risk for the next time window $\Delta t$.

- **Request Body** (`PredictionRequest`):
```json
{
  "time_horizon_seconds": 5
}
```

- **Response** (`PredictionResponse`):
```json
{
  "time_horizon_seconds": 5,
  "high_risk_links_count": 2,
  "predictions": [
    {
      "source": "N1",
      "target": "N2",
      "current_utilization": 0.75,
      "predicted_utilization": 0.92,
      "congestion_risk_score": 0.88,
      "is_congested": true
    }
  ]
}
```

---

### 2.4 Traffic Engineering & Path Optimization

#### `POST /api/v1/te/optimize`
Calculates and applies forwarding paths using specified algorithm (Baseline Shortest Path vs. ML-CSPF).

- **Request Body** (`TERouteRequest`):
```json
{
  "algorithm": "ml_cspf",
  "source_node": "N1",
  "destination_node": "N8",
  "demand_mbps": 250.0
}
```

- **Response** (`TERouteResponse`):
```json
{
  "algorithm_used": "ml_cspf",
  "computed_path": ["N1", "N3", "N7", "N8"],
  "total_path_delay_ms": 6.5,
  "bottleneck_predicted_utilization": 0.64,
  "rerouted_flows_count": 1
}
```

---

### 2.5 Predictive What-If Digital Twin Simulation

#### `POST /api/v1/simulation/what-if`
Executes hypothetical scenario on cloned Network Digital Twin.

- **Request Body** (`WhatIfRequest`):
```json
{
  "scenario_type": "link_failure",
  "failed_link": {"source": "N1", "target": "N2"},
  "traffic_multiplier": 1.5,
  "eval_algorithm": "ml_cspf"
}
```

- **Response** (`WhatIfResponse`):
```json
{
  "scenario_type": "link_failure",
  "baseline_metrics": {
    "avg_latency_ms": 14.2,
    "packet_loss_rate": 0.08,
    "max_link_utilization": 0.98
  },
  "what_if_metrics": {
    "avg_latency_ms": 8.1,
    "packet_loss_rate": 0.00,
    "max_link_utilization": 0.71
  },
  "recommended_action": "Apply ML-CSPF re-route via path N1-N4-N8 prior to cutting link N1-N2."
}
```

---

### 2.6 RAG Explanation

#### `POST /api/v1/rag/explain`
Queries the out-of-band RAG explainer.

- **Request Body** (`RAGQueryRequest`):
```json
{
  "question": "Why did ML-CSPF reroute flow F12 away from link N1-N2?"
}
```

- **Response** (`RAGQueryResponse`):
```json
{
  "question": "Why did ML-CSPF reroute flow F12 away from link N1-N2?",
  "answer": "Link N1-N2 was predicted to hit 92% utilization within the next 5 seconds due to a Pareto elephant flow surge. ML-CSPF rerouted flow F12 through path N1-N3-N7-N8 to keep peak link utilization below the 80% threshold.",
  "sources": ["docs/architecture.md#2.4", "logs/te_events.log"]
}
```

---

### 2.7 System Health

#### `GET /api/v1/health`
- **Response**: `{"status": "healthy", "version": "1.0.0", "simulation_running": true}`
