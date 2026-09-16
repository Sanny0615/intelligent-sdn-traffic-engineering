# Machine Learning Strategy & Telemetry Specification

## 1. Simplified ML Problem Formulation

The primary goal of the ML pipeline is to predict impending link congestion risk to enable proactive traffic engineering.

### 1.1 Single Primary Prediction Task **[CORE MVP]**
To keep the MVP focused and robust, the system implements **one primary prediction task**:

- **Target Variable**: Link Congestion Risk Score $\hat{R}_e(t + \Delta t) \in [0.0, 1.0]$ for directed edge $e=(u, v)$ at future time window $\Delta t = 5\text{s}$.
- **Formulation**: Binary classification where $Y_e(t+\Delta t) = 1$ if link utilization $\rho_e(t+\Delta t) \ge 0.85$, or continuous risk score prediction via probability calibration $P(Y_e=1)$.
- **Primary Model**: `scikit-learn` `RandomForestClassifier` (or `RandomForestRegressor`).

*Note: Multi-target models and LightGBM model comparisons are classified as **[OPTIONAL]** post-MVP extensions.*

---

## 2. Telemetry Feature Engineering **[CORE MVP]**

Features are extracted per directed link $e = (u, v)$ at time tick $t$:

| Feature Name | Symbol | Description | Scope |
| :--- | :--- | :--- | :--- |
| **Link Utilization** | $\rho_e(t)$ | Current byte load divided by link capacity. | **[CORE MVP]** |
| **Queue Occupancy** | $q_e(t)$ | Current packet count in link buffer queue. | **[CORE MVP]** |
| **1-step Load Delta** | $\Delta \rho_1(t)$ | $\rho_e(t) - \rho_e(t-1)$ | **[CORE MVP]** |
| **3-step Load Delta** | $\Delta \rho_3(t)$ | $\rho_e(t) - \rho_e(t-3)$ | **[CORE MVP]** |
| **5-step Rolling Mean** | $\mu_{\rho, 5}(t)$ | Moving average over past 5 ticks. | **[CORE MVP]** |
| **Edge Betweenness** | $C_B(e)$ | Shortest-path betweenness centrality of edge $e$. | **[CORE MVP]** |
| **Active Flow Count** | $N_f(e)$ | Number of active traffic flows traversing link $e$.| **[CORE MVP]** |

---

## 3. Model Candidates & Justification

| Model Architecture | Suitability | Scope Category | Justification |
| :--- | :--- | :--- | :--- |
| **Scikit-Learn Random Forest** | High | **[CORE MVP]** | Fast inference ($<1\text{ms}$), handles non-linear step dynamics, no complex tuning needed. |
| **LightGBM** | High | **[OPTIONAL]** | Fast gradient boosting; optional comparison post-MVP. |
| **Graph Neural Network (GNN)** | Low | **[FUTURE]** | High complexity; deferred to future academic extension. |

---

## 4. Dataset Generation & Reproducibility Strategy

1. **Topology**: Initial training telemetry generated using the small deterministic custom topology (Phase 1). Additional topologies added in Phase 5.
2. **Deterministic Seed Control**: Synthetic flow generation uses explicit random seeds (`seed=42`) to guarantee identical telemetry generation across repeated training runs.
3. **Train / Test Split**: 70% Training, 15% Validation, 15% Test using strict temporal time-series splitting to prevent data leakage.

---

## 5. Model Evaluation Metrics

Performance will be empirically measured during Phase 3 experiments:

- **Classification Task Metrics**: ROC-AUC Score, F1-Score, Precision, Recall, Confusion Matrix (*Targets to be measured empirically during experiments*).
- **Regression Task Metrics (if evaluated)**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) (*Targets to be measured empirically during experiments*).
- **Inference Speed**: Average inference time per link vector ($\le 1\text{ ms}$ budget).
