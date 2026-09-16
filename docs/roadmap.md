# Project Roadmap & Implementation Timeline

## 1. Overview & Phased Priority Order

Development strictly follows a phased implementation order:

```
Network Simulation Core (Phase 1)
  └─► Telemetry & Baseline Routing (Phase 2)
        └─► ML Prediction & Predictive TE (Phase 3)
              └─► Digital Twin, FastAPI & Streamlit UI (Phase 4)
                    └─► RAG Layer & Benchmarking Package (Phase 5)
```

---

## 2. Phase-by-Phase Roadmap

### Phase 1: Deterministic Network Simulation Foundation **[CORE MVP]**
- **Focus**: Building a deterministic NetworkX simulation core on Windows.
- **Scope**:
  - Network topology abstraction (`NetworkXEngine`).
  - Small deterministic 6-node custom topology.
  - Node and directed link representations with bandwidth capacities ($C_e$) and propagation delays ($d_e$).
  - Basic traffic flow representation (source, destination, demand Mbps).
  - Basic forwarding / path simulation logic.
  - Telemetry counters (byte counters, packet counters) needed for later telemetry extraction.
  - Deterministic random seed support (`seed=42`) for reproducible experiments.
  - Automated `pytest` unit tests for simulation core.
- **Explicit Exclusions**: *No ML, FastAPI, Streamlit, RAG, Digital Twin, Mininet, or OpenFlow in Phase 1.*
- **Phase 1 DoD**: `pytest tests/test_simulation.py` passes 100% verifying topology construction, deterministic flow forwarding, and counter tracking.

---

### Phase 2: Telemetry Pipeline & Baseline Routing **[CORE MVP]**
- **Focus**: Telemetry feature extraction & baseline routing.
- **Scope**:
  - Telemetry collector (`TelemetryPipeline`) logging rolling window features.
  - Baseline Shortest Path First (SPF / Dijkstra) routing module.
  - Dataset generation scripts for saving telemetry CSVs using fixed seeds.
- **Phase 2 DoD**: Telemetry pipeline logs link utilization time-series data cleanly under baseline SPF routing.

---

### Phase 3: Single-Task ML Prediction & Predictive TE Engine **[CORE MVP]**
- **Focus**: ML congestion risk prediction and path cost optimization.
- **Scope**:
  - Single-task `scikit-learn` `RandomForestClassifier` (or Regressor) predicting link congestion risk $\hat{R}_e$.
  - Model serialization to `models/congestion_rf.joblib`.
  - ML-Assisted Predictive Traffic Engineering router (`ML_Predictive_Cost_Routing`).
- **Phase 3 DoD**: Predictive router dynamically re-routes traffic away from links with high predicted risk scores.

---

### Phase 4: Network Digital Twin, FastAPI Service & Streamlit UI **[CORE MVP / ADVANCED]**
- **Focus**: User interface, backend REST API, and scenario simulation.
- **Scope**:
  - Network Digital Twin (`WhatIfSimulator`) deep-cloning graphs for in-memory What-If testing **[ADVANCED]**.
  - FastAPI backend service exposing REST endpoints (`/topology`, `/telemetry`, `/predict`, `/te/optimize`, `/simulation/what-if`).
  - Interactive Streamlit dashboard with real-time topology heatmaps and comparative charts.
- **Phase 4 DoD**: Streamlit dashboard renders live network updates and What-If scenario results cleanly.

---

### Phase 5: RAG Intelligence Layer, Benchmarking & Viva Package **[ADVANCED]**
- **Focus**: Out-of-band RAG explainer, empirical research benchmarking, and viva preparation.
- **Scope**:
  - Standalone RAG explainer indexing project documentation **[ADVANCED]**.
  - Multi-topology benchmark suite (Abilene, GEANT, Fat-Tree).
  - Empirical metric evaluation (Latency, Throughput, Packet Loss, Link Utilization Variance).
  - Final viva slides and research report.
- **Phase 5 DoD**: Benchmark script exports clean comparative charts; RAG explainer answers project architecture queries accurately.
