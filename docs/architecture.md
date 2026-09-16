# System Architecture Specification

## 1. High-Level Architecture Overview

The **ML-Assisted Predictive Traffic Engineering System** is built using a simulation-first modular architecture. It cleanly separates network simulation, telemetry collection, ML prediction, path-cost optimization, REST API delivery, user dashboard visualization, and out-of-band AI explanation.

```
+-----------------------------------------------------------------------------------+
|                        PRESENTATION LAYER [CORE MVP / ADVANCED]                  |
|                                                                                   |
|   +----------------------------------+   +------------------------------------+   |
|   | Streamlit Interactive Dashboard  |   | Out-of-Band RAG Explainer Panel    |   |
|   | [CORE MVP - Phase 4]             |   | [ADVANCED - Phase 5]               |   |
|   +----------------------------------+   +------------------------------------+   |
+-----------------------------------------|-----------------------------------------+
                                          | REST HTTP / JSON
+-----------------------------------------v-----------------------------------------+
|                        FASTAPI SERVICE LAYER [CORE MVP - Phase 4]                 |
|                                                                                   |
|  +--------------------+  +-------------------+  +------------------------------+  |
|  | Topology API       |  | Telemetry API     |  | What-If Engine API           |  |
|  +--------------------+  +-------------------+  +------------------------------+  |
|  | Prediction API     |  | Routing TE API    |  | RAG Explainer API            |  |
|  +--------------------+  +-------------------+  +------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          | In-Memory Python Integration
+-----------------------------------------v-----------------------------------------+
|                        CORE ENGINE & LOGIC LAYER [CORE MVP / ADVANCED]            |
|                                                                                   |
|  +---------------------+   +---------------------+   +-------------------------+  |
|  | Network Digital Twin|   | ML Predictor        |   | ML Predictive Router    |  |
|  | (Scenario Twin)     |   | (Scikit-Learn RF)   |   | (Cost Optimizer)        |  |
|  | [ADVANCED - Ph. 4]  |   | [CORE MVP - Ph. 3]  |   | [CORE MVP - Ph. 3]      |  |
|  +---------------------+   +---------------------+   +-------------------------+  |
|             |                         |                           |               |
|             +-------------------------+---------------------------+               |
|                                       |                                           |
|                   +-------------------+-------------------+                       |
|                   |  Telemetry Extractor & Aggregator     |                       |
|                   |  [CORE MVP - Phase 2]                 |                       |
|                   +-------------------+-------------------+                       |
+---------------------------------------|-------------------------------------------+
                                        | Direct Python Interface
+---------------------------------------v-------------------------------------------+
|                   SIMULATION CORE LAYER [CORE MVP - Phase 1]                      |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   | NetworkX Simulation Engine (Small Custom Topology, Bytes/Packets, Seeds)   |   |
|   +---------------------------------------------------------------------------+   |
|   | Optional Mininet / OpenFlow Adapter Interface [FUTURE - Post-MVP]         |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Implementation Priority & Component Responsibilities

Implementation proceeds strictly according to core architectural priority:

### Priority 1: Network Simulation Engine (`NetworkXEngine`) **[CORE MVP - Phase 1]**
- **Role**: Maintains live directed network topology graph $G=(V, E)$.
- **Responsibilities**:
  - Initializes small deterministic 6-node custom topology with link bandwidths ($C_e$), delays ($d_e$), and queues.
  - Generates synthetic flow demands using explicit random seeds (`seed=42`) for 100% experiment reproducibility.
  - Forwards traffic along computed paths and updates link byte/packet counters per simulation tick.

### Priority 2: Telemetry Collector (`TelemetryPipeline`) **[CORE MVP - Phase 2]**
- **Role**: Periodically polls network state and computes feature matrices.
- **Responsibilities**:
  - Logs rolling windows ($t-k \dots t$) of link utilization $\rho_e(t)$, load deltas $\Delta \rho_e$, and edge centrality.
  - Formats feature vectors for single-task ML model training and inference.

### Priority 3: Baseline & ML Predictive Router (`TERouter`) **[CORE MVP - Phase 2 & 3]**
- **Role**: Calculates forwarding paths using baseline or predictive algorithms.
- **Responsibilities**:
  - **Baseline Dijkstra SPF**: Calculates paths based on static link delays ($W_e = d_e$).
  - **ML-Assisted Predictive Traffic Engineering (`ML_Predictive_Cost_Routing`)**: Calculates paths using dynamic link weights:
    $$W_e = d_e + \alpha \cdot \hat{R}_e$$
    where $d_e$ is base delay, $\hat{R}_e \in [0, 1]$ is predicted congestion risk score, and $\alpha$ is a scaling penalty factor.

### Priority 4: ML Inference Engine (`CongestionPredictor`) **[CORE MVP - Phase 3]**
- **Role**: Predicts future link congestion risk scores.
- **Responsibilities**:
  - Fits a single-task `scikit-learn` `RandomForestClassifier` or `RandomForestRegressor` on historical telemetry.
  - Predicts future link congestion risk $\hat{R}_e(t+\Delta t)$ for upcoming time windows.

### Priority 5: Network Digital Twin & What-If Engine (`WhatIfSimulator`) **[ADVANCED - Phase 4]**
- **Role**: Executes hypothetical scenario simulations in-memory after base TE engine is stable.
- **Responsibilities**:
  - Deep-clones active graph state $G_{\text{live}} \to G_{\text{twin}}$.
  - Injects scenario overrides (e.g., link failure, 2x traffic surge) on $G_{\text{twin}}$.
  - Evaluates baseline vs. predictive TE paths on $G_{\text{twin}}$ and reports expected performance deltas.

### Priority 6: FastAPI Service Layer (`app.main`) **[CORE MVP - Phase 4]**
- **Role**: Exposes RESTful API endpoints for topology, telemetry, prediction, routing, and What-If queries.

### Priority 7: Streamlit Dashboard (`dashboard.py`) **[CORE MVP - Phase 4]**
- **Role**: Renders interactive topology heatmaps, metric charts, and What-If scenario controls.

### Priority 8: Out-of-Band RAG Explainer (`RAGExplainer`) **[ADVANCED - Phase 5]**
- **Role**: Standalone educational assistant answering user queries about project documentation and routing actions. Strictly isolated from core routing logic.

---

## 3. End-to-End Data Flow Pipeline

```
[Simulation Tick (t)]
       |
       v
[NetworkX Simulation Engine] -> Updates Link Loads & Telemetry Counters
       |
       v
[Telemetry Collector] --------> Computes Feature Matrix: [rho_t, delta_rho, centrality]
       |
       v
[ML Congestion Predictor] ----> Outputs Predicted Congestion Risk Score R_e (t+5s)
       |
       +------------------------------------+
       |                                    |
       v                                    v
[Live Predictive TE Router]       [Network Digital Twin (Phase 4)]
  - Updates dynamic edge weights     - Injects scenario override
  - Re-routes active flows           - Evaluates candidate paths
       |                             - Produces predicted report
       v                                    |
[Forwarding Rules Updated]                  v
       |                          [Exposes REST API (FastAPI)]
       +------------------------------------>+
                                             |
                                             v
                                 [Streamlit Dashboard & RAG]
```

---

## 4. Architectural Appendix: Scope & Classification

### A. Final Recommended Architecture
Layered simulation-first architecture with pure Python NetworkX simulation core, single-task Scikit-learn ML predictor, FastAPI REST API, Streamlit dashboard, and out-of-band RAG explainer.

### B. Final Recommended Technology Stack
- **CORE MVP**: `Python 3.11`, `NetworkX 3.6`, `NumPy 2.4`, `Pandas 3.0`, `Scikit-learn 1.9`, `FastAPI 0.141`, `Uvicorn 0.53`, `Streamlit 1.64`, `Pytest 9.1`.
- **ADVANCED**: `SentenceTransformers` / `FAISS` (Phase 5 RAG only).
- **OPTIONAL**: `LightGBM` (post-MVP comparison model), SQLite logging.
- **FUTURE**: Mininet / OpenFlow / Ryu adapter interface (`ISDNControllerAdapter`).

### C. Technologies Explicitly Rejected & Justifications

| Rejected Technology | Scope | Justification for Rejection |
| :--- | :--- | :--- |
| **Docker / WSL Mandatory Setup** | REMOVE | NetworkX provides native Windows execution without Linux environment overhead. |
| **Microservices Architecture** | REMOVE | Unnecessary overhead for single-host academic micro-project. |
| **LLM-controlled Core Routing** | REMOVE | Core routing decisions must be 100% mathematical and deterministic. |
| **SentenceTransformers in Phase 1-4**| REMOVE | Deferred to Phase 5 to avoid premature dependency burden during core engine build. |

### D. Major Technical Risks & Mitigation Strategies

| Technical Risk | Likelihood | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Routing Oscillation** | Medium | High | Apply exponential smoothing and hysteresis thresholds ($\epsilon = 0.05$) to dynamic link weights. |
| **Over-complex Initial Setup** | Medium | Medium | Restrict Phase 1 strictly to a small 6-node custom topology and deterministic seed execution. |
| **Dependency Conflicts** | Low | Low | Keep initial dependencies minimal (standard `scikit-learn` stack). |
