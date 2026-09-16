# Experimentation & Benchmarking Plan

## 1. Experimental Methodology Overview

The evaluation framework compares **Baseline Shortest Path First (Dijkstra SPF)** against **ML-Assisted Predictive Traffic Engineering (`ML_Predictive_Cost_Routing`)** under dynamic, bursty traffic workloads.

To ensure research validity, all comparisons run under **identical traffic scenarios** using fixed random seeds.

---

## 2. Test Topologies

1. **Small Custom Topology** **[CORE MVP - Phase 1]**: 6 nodes, 8 directed links (used for baseline verification, debugging, and initial pipeline validation).
2. **Abilene Core Network** **[ADVANCED - Phase 5]**: 12 nodes, 30 directed links.
3. **GEANT Network** **[ADVANCED - Phase 5]**: 23 nodes, 74 directed links.
4. **Synthetic Fat-Tree Network** **[ADVANCED - Phase 5]**: 16 nodes, 48 directed links.

---

## 3. Compared Routing Algorithms

1. **Static Shortest Path First (Baseline SPF)** **[CORE MVP]**:
   - Computes paths based on static link propagation delays ($W_e = d_e$).
   - Ignores link utilization until buffer queues overflow.

2. **ML-Assisted Predictive Traffic Engineering (`ML_Predictive_Cost_Routing`)** **[CORE MVP]**:
   - Computes paths using dynamic cost function:
     $$W_e = d_e + \alpha \cdot \hat{R}_e(t+5\text{s})$$
     where $\hat{R}_e$ is predicted link congestion risk score.
   - Proactively routes flows away from predicted bottleneck links.

---

## 4. Quantitative Metrics (Empirical Measurement Plan)

All numerical results will be gathered empirically during Phase 5 experiments:

| Metric Name | Symbol / Formula | Measurement Status |
| :--- | :--- | :--- |
| **Average Latency** | $\bar{L} = \frac{1}{N_p} \sum_{i=1}^{N_p} (t_{\text{recv}, i} - t_{\text{sent}, i})$ | *To be measured during experiment* |
| **Tail Latency (p95 / p99)** | $95^{\text{th}} / 99^{\text{th}}$ percentile latency | *To be measured during experiment* |
| **Aggregate Throughput** | $TH = \frac{\text{Bytes Received} \times 8}{T \times 10^6} \text{ (Mbps)}$ | *To be measured during experiment* |
| **Packet Loss Rate** | $PLR = \frac{N_{\text{dropped}}}{N_{\text{sent}}} \times 100\%$ | *To be measured during experiment* |
| **Link Utilization Variance** | $\operatorname{Var}(\rho) = \frac{1}{\|E\|} \sum_{e \in E} (\rho_e - \bar{\rho})^2$ | *To be measured during experiment* |
| **Congestion Duration** | Ticks where any link $\rho_e \ge 0.85$ | *To be measured during experiment* |
| **ML Risk Prediction Accuracy**| ROC-AUC / F1-Score / MAE | *To be measured during experiment* |

---

## 5. Predictive What-If Experiment Protocol **[ADVANCED - Phase 4]**

```
Step 1: Initialize small custom topology with fixed seed (seed=42).
Step 2: Trigger What-If Scenario: "Simulate failure of link N1-N2 at tick t=100".
Step 3: What-If Engine clones active graph state G_live -> G_twin.
Step 4: Execute Baseline SPF on G_twin post-failure.
Step 5: Execute ML-Assisted Predictive TE on G_twin.
Step 6: Compute comparative metrics (Latency, Drops, Utilization) on G_twin.
Step 7: Render comparative charts on Streamlit dashboard.
```
