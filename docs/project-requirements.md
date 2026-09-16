# Project Requirements Specification

## 1. Problem Definition & Research Context

### 1.1 Background
Software-Defined Networking (SDN) decouples the network control plane from the data forwarding plane. Traditional SDN routing relies on static or reactive algorithms (such as Dijkstra's Shortest Path First), updating routes only **after** packet loss or link congestion has occurred.

### 1.2 Core Problem Statement
*How can an SDN-based system leverage real-time network telemetry and Machine Learning (ML) to predict future link congestion and intelligently optimize traffic paths before performance degradation occurs?*

This project implements **ML-Assisted Predictive Traffic Engineering**, anticipating congestion risks, evaluating routing paths, and proactively adjusting traffic allocation.

---

## 2. Project Objectives

1. **Deterministic Network Simulation Foundation**: Develop a pure-Python network simulation core using `NetworkX` supporting a small deterministic topology, traffic flows, packet forwarding, link capacities, delays, and telemetry counters with full random seed reproducibility.
2. **Predictive Telemetry & ML Engine**: Formulate a lightweight ML pipeline using `scikit-learn` to predict future link congestion risk ($R_e \in [0.0, 1.0]$).
3. **Predictive What-If Scenario Simulator**: Implement a Network Digital Twin to evaluate candidate rerouting decisions and hypothetical traffic surges in-memory before committing traffic engineering rules.
4. **Reproducible Experimental Benchmark**: Conduct controlled comparative experiments comparing Baseline Shortest Path routing against proposed ML-Assisted Predictive Traffic Engineering under identical traffic scenarios.
5. **Out-of-Band Explainable AI Layer**: Provide an optional RAG-based explanation interface (deferred to Phase 5) that explains routing rationale grounded in project documentation.

---

## 3. Scope & Feature Tagging Matrix

| Feature / Module | Category | Implementation Phase |
| :--- | :--- | :--- |
| Network Topology Abstraction & Small Custom Topology | **[CORE MVP]** | Phase 1 |
| Deterministic Traffic Flows & Link Telemetry Counters | **[CORE MVP]** | Phase 1 & Phase 2 |
| Baseline Shortest Path (SPF) Routing | **[CORE MVP]** | Phase 2 |
| Single-Task ML Congestion Risk Predictor (`RandomForest`) | **[CORE MVP]** | Phase 3 |
| ML-Assisted Predictive Traffic Engineering Algorithm | **[CORE MVP]** | Phase 3 |
| FastAPI REST API Service | **[CORE MVP]** | Phase 4 |
| Streamlit Interactive Dashboard | **[CORE MVP]** | Phase 4 |
| Predictive What-If Digital Twin Scenario Simulator | **[ADVANCED]** | Phase 4 |
| Additional Standard Topologies (Abilene, GEANT, Fat-Tree) | **[ADVANCED]** | Phase 5 |
| Out-of-Band RAG / LLM Explanation Agent | **[ADVANCED]** | Phase 5 |
| LightGBM / GNN Model Comparisons | **[OPTIONAL]** | Post-MVP / Optional |
| Mininet / OpenFlow Adapter Interface (`ISDNControllerAdapter`)| **[FUTURE]** | Post-Micro-project |

---

## 4. Functional Requirements

| ID | Module | Description | Scope | Phase |
| :--- | :--- | :--- | :--- | :--- |
| **FR-1.1** | Topology Engine | System shall construct directed topology graphs with node capacities, link bandwidths (Mbps), propagation delays (ms), and deterministic seed support. | **[CORE MVP]** | Phase 1 |
| **FR-1.2** | Telemetry Collection | System shall record fine-grained time-series link metrics: byte counters, packet counters, current utilization ($\rho_e$), and drop rates. | **[CORE MVP]** | Phase 2 |
| **FR-2.1** | Traffic Generation | System shall synthesize flow demands (mice and elephant flows) reproducibly using configurable random seeds. | **[CORE MVP]** | Phase 1 & 2 |
| **FR-3.1** | ML Congestion Prediction | System shall train a single-task `scikit-learn` model to predict future link congestion risk score ($R_e \in [0, 1]$). | **[CORE MVP]** | Phase 3 |
| **FR-4.1** | Baseline Routing | System shall compute flow paths using standard Shortest Path First (Dijkstra) based on static link delays. | **[CORE MVP]** | Phase 2 |
| **FR-4.2** | ML-Assisted Predictive TE | System shall compute flow paths using dynamic ML-assisted link cost optimization (`ML_Predictive_Cost_Routing`). | **[CORE MVP]** | Phase 3 |
| **FR-5.1** | What-If Digital Twin | System shall execute scenario simulations (link failure, traffic spike) on cloned in-memory graphs. | **[ADVANCED]** | Phase 4 |
| **FR-6.1** | REST API Service | FastAPI backend shall expose endpoints for topology, telemetry, prediction, routing, and What-If queries. | **[CORE MVP]** | Phase 4 |
| **FR-6.2** | Interactive Dashboard | Streamlit dashboard shall display real-time topology heatmaps, telemetry metrics, and side-by-side comparative charts. | **[CORE MVP]** | Phase 4 |
| **FR-7.1** | RAG Explainer | Out-of-band RAG agent shall answer documentation queries explaining TE actions. | **[ADVANCED]** | Phase 5 |

---

## 5. Non-Functional Requirements

- **NFR-1 (Performance & Responsiveness)**: Simulation tick and ML prediction inference must complete within $\le 100\text{ ms}$ for demo topologies.
- **NFR-2 (Strict Reproducibility)**: Synthetic traffic generation and graph operations must support explicit random seeds to ensure 100% reproducible baseline vs. proposed comparisons.
- **NFR-3 (Native Windows Execution)**: Core simulation runs natively on Windows 10/11 using standard Python 3.11 without requiring WSL, Docker, or external emulators.
- **NFR-4 (Modularity & Decoupling)**: Simulation engine, ML logic, TE path selection, API handlers, and UI must remain strictly decoupled via standard Python module interfaces.
- **NFR-5 (Empirical Rigor)**: All performance metrics (latency, throughput, packet loss) must be empirically measured during experiments; no hardcoded success claims permitted.

---

## 6. Definition of Done (DoD) per Phase

- **Phase 1: Deterministic Network Simulation Foundation**: Small 6-node custom topology built, basic forwarding simulation working, telemetry counters active, random seed support verified, `pytest` suite passing.
- **Phase 2: Telemetry Pipeline & Baseline Routing**: Telemetry feature extraction logging data, static Shortest Path First (SPF) routing operational under dynamic flows.
- **Phase 3: ML Congestion Prediction & Predictive TE**: Single-task `scikit-learn` model predicting congestion risk score, `ML_Predictive_Cost_Routing` rerouting flows away from predicted high-risk links.
- **Phase 4: FastAPI Backend, Streamlit UI & Digital Twin**: REST API functional, Streamlit dashboard rendering live topology, in-memory Digital Twin executing What-If scenarios.
- **Phase 5: Benchmarking, RAG Layer & Viva Package**: Comparative experiments executed on identical traffic matrices, performance metrics documented, RAG explainer responding to project queries, viva presentation ready.
