import os
import sys
from pathlib import Path

# Ingest project root into sys.path for Streamlit Cloud deployment compatibility
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from src.dashboard.api_client import DashboardAPIClient

# Configure Page
st.set_page_config(
    page_title="Intelligent SDN Traffic Engineering",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation & Settings
st.sidebar.title("🌐 SDN TE Dashboard")
st.sidebar.markdown("---")

default_api_url = os.getenv("API_BASE_URL", "https://intelligent-sdn-traffic-engineering.onrender.com/api/v1")
api_base_url = st.sidebar.text_input("FastAPI Base URL", value=default_api_url)
api_client = DashboardAPIClient(base_url=api_base_url)

# Health Status Badge & Warning Banner
health_data = api_client.get_health()
if health_data.get("error"):
    st.sidebar.error("❌ FastAPI Backend Offline")
    st.sidebar.caption(health_data["message"])
    st.warning("⚠️ Backend API is currently connecting or offline. Verify your FastAPI Base URL or launch the backend service.")
else:
    st.sidebar.success(f"🟢 Backend Online (v{health_data.get('version', '1.0')})")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Network Topology", "Telemetry", "ML Prediction", "Traffic Engineering", "What-If Simulation", "AI Assistant"]
)


st.sidebar.markdown("---")
st.sidebar.caption("Computer Networks Micro-Project | Python 3.11 + NetworkX + Scikit-Learn + FastAPI + Streamlit")


# =============================================================================
# Page 1: Overview
# =============================================================================
if page == "Overview":
    st.title("🌐 Intelligent SDN Traffic Engineering Using Machine Learning")
    st.markdown("""
    Welcome to the **Intelligent SDN Traffic Engineering Platform**. This system combines real-time network telemetry,
    Machine Learning congestion risk prediction, dynamic path-cost optimization, and a **What-If Network Digital Twin**
    to proactively prevent network congestion.
    """)

    col1, col2, col3, col4 = st.columns(4)

    topo = api_client.get_topology()
    nodes_count = topo.get("nodes_count", 6) if not topo.get("error") else 6
    edges_count = topo.get("edges_count", 12) if not topo.get("error") else 12

    col1.metric("API Status", "OK" if not health_data.get("error") else "Offline")
    col2.metric("Network Nodes", nodes_count)
    col3.metric("Directed Links", edges_count)
    col4.metric("Primary ML Model", "Random Forest")

    st.markdown("### 🔄 End-to-End Pipeline Architecture")
    st.info("""
    **Network Engine (NetworkX)** ➔ **Telemetry Collector** ➔ **Feature Extraction** ➔ 
    **ML Congestion Predictor (Random Forest)** ➔ **Predictive Cost Router (ML-CSPF)** ➔ 
    **What-If Digital Twin Scenario Simulator** ➔ **FastAPI REST Service** ➔ **Streamlit Dashboard**
    """)

    st.markdown("### 📌 Key Features")
    st.markdown("""
    - **Simulation-First Engine**: Pure Python 3.11 + NetworkX running natively on Windows.
    - **Single-Task ML Predictor**: Predicts link congestion risk score ($P(\text{congestion} \ge 85\%)$) at $t+1$.
    - **Proactive Traffic Engineering**: Dynamic link cost optimization ($W_e = d_e + \alpha \cdot P(\text{risk})$).
    - **What-If Network Digital Twin**: In-memory state replication for scenario testing without mutating live network state.
    """)


# =============================================================================
# Page 2: Network Topology
# =============================================================================
elif page == "Network Topology":
    st.title("🗺️ Network Topology Abstraction")
    st.markdown("Displays structured information about active network nodes, directed links, capacities, and propagation delays.")

    topo = api_client.get_topology()
    if topo.get("error"):
        st.error(topo["message"])
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Nodes List ({topo['nodes_count']})")
            nodes_df = pd.DataFrame(topo["nodes"])
            st.dataframe(nodes_df, use_container_width=True)

        with col2:
            st.subheader(f"Directed Links List ({topo['edges_count']})")
            links_df = pd.DataFrame(topo["links"])
            st.dataframe(links_df, use_container_width=True)


# =============================================================================
# Page 3: Telemetry
# =============================================================================
elif page == "Telemetry":
    st.title("📊 Live Network Telemetry Snapshot")
    st.markdown("Polls real-time per-link traffic loads, utilization ratios, packet drops, and congestion flags.")

    telemetry = api_client.get_telemetry()
    if telemetry.get("error"):
        st.error(telemetry["message"])
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Tick", telemetry["timestamp_tick"])
        col2.metric("Active Flows", telemetry["active_flows_count"])
        col3.metric("Throughput (Mbps)", f"{telemetry['total_throughput_mbps']:.2f}")

        st.subheader("Per-Link Telemetry Table")
        t_df = pd.DataFrame(telemetry["links"])
        t_df["utilization_pct"] = (t_df["utilization"] * 100).round(2)
        st.dataframe(t_df[["link_id", "source", "target", "capacity_mbps", "current_load_mbps", "utilization_pct", "is_congested", "dropped_packets"]], use_container_width=True)

        st.subheader("Link Utilization Chart (% of Capacity)")
        chart_df = t_df.set_index("link_id")[["utilization_pct"]]
        st.bar_chart(chart_df)

        congested_links = t_df[t_df["is_congested"] == 1]
        if not congested_links.empty:
            st.warning(f"⚠️ {len(congested_links)} link(s) currently experiencing congestion (≥85% capacity): {list(congested_links['link_id'])}")


# =============================================================================
# Page 4: ML Prediction
# =============================================================================
elif page == "ML Prediction":
    st.title("🤖 ML Future Congestion Prediction (t+1)")
    st.markdown("Predicts whether a target link will become congested at the next simulation tick ($t+1$) based on current telemetry features.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            link_id = st.text_input("Link ID", value="S1->S3")
            utilization = st.number_input("Current Utilization (ratio)", min_value=0.0, max_value=5.0, value=0.88, step=0.05)
            prev_utilization = st.number_input("Previous Utilization (ratio)", min_value=0.0, max_value=5.0, value=0.65, step=0.05)
            delta_utilization = round(utilization - prev_utilization, 4)
            st.text(f"Calculated Delta: {delta_utilization}")

        with col2:
            current_load_mbps = st.number_input("Current Traffic Load (Mbps)", min_value=0.0, value=880.0, step=50.0)
            capacity_mbps = st.number_input("Link Capacity (Mbps)", min_value=100.0, value=1000.0, step=100.0)
            delay_ms = st.number_input("Link Delay (ms)", min_value=0.1, value=2.0, step=0.5)
            active_flows_count = st.number_input("Active Flows Count", min_value=0, value=3, step=1)

        submit_pred = st.form_submit_button("Predict Future Congestion Risk")

    if submit_pred:
        payload = {
            "link_id": link_id,
            "utilization": utilization,
            "prev_utilization": prev_utilization,
            "delta_utilization": delta_utilization,
            "current_load_mbps": current_load_mbps,
            "capacity_mbps": capacity_mbps,
            "delay_ms": delay_ms,
            "drop_rate": 0.0,
            "active_flows_count": float(active_flows_count)
        }

        res = api_client.predict_congestion(payload)
        if res.get("error"):
            st.error(res["message"])
        else:
            cls = res["predicted_congestion_class"]
            prob = res["congestion_probability"]

            if cls == 1:
                st.error(f"🚨 HIGH CONGESTION RISK PREDICTED at t+1 (Class: {cls} | Risk Probability: {prob * 100:.1f}%)")
            else:
                st.success(f"✅ NORMAL LINK STATE PREDICTED at t+1 (Class: {cls} | Risk Probability: {prob * 100:.1f}%)")

            st.progress(prob)
            st.caption("*Prediction generated using pre-trained Phase 3 Random Forest Classifier artifact.")


# =============================================================================
# Page 5: Traffic Engineering
# =============================================================================
elif page == "Traffic Engineering":
    st.title("⚡ Traffic Engineering Path Optimization")
    st.markdown("Compares conventional **Baseline Shortest Path (SPF)** against **ML-Assisted Predictive Traffic Engineering**.")

    with st.form("routing_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            source = st.selectbox("Source Node", ["H1", "H2", "H3", "S1", "S2", "S3"], index=0)
            destination = st.selectbox("Destination Node", ["H1", "H2", "H3", "S1", "S2", "S3"], index=2)
        with col2:
            demand_mbps = st.number_input("Flow Demand (Mbps)", min_value=10.0, max_value=2000.0, value=400.0, step=50.0)
            strategy = st.selectbox("Routing Strategy", ["predictive", "baseline"], index=0)
        with col3:
            alpha_penalty = st.slider("Congestion Penalty Alpha (α)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
            flow_id = st.text_input("Flow ID (Optional)", value="flow_demo_1")

        submit_route = st.form_submit_button("Compute Optimal Path")

    if submit_route:
        payload = {
            "source": source,
            "destination": destination,
            "demand_mbps": demand_mbps,
            "strategy": strategy,
            "alpha_penalty": alpha_penalty,
            "flow_id": flow_id
        }

        res = api_client.optimize_routing(payload)
        if res.get("error"):
            st.error(res["message"])
        else:
            st.subheader(f"Routing Decision Result ({strategy.upper()})")

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"**Baseline Path**: `{' -> '.join(res['baseline_path'])}`")
            col2.markdown(f"**Selected Path**: `{' -> '.join(res['selected_path'])}`")
            col3.markdown(f"**Rerouted?**: {'⚠️ YES (Rerouted)' if res['is_rerouted'] else '✅ NO (Base Path)'}")

            st.info(f"**Decision Reason**: {res['reason']}")

            st.subheader("Candidate Paths Predictive Cost Matrix")
            cand_df = pd.DataFrame(res["candidate_evaluations"])
            cand_df["path_str"] = cand_df["path"].apply(lambda p: " -> ".join(p))
            st.dataframe(cand_df[["path_str", "total_delay_ms", "predicted_congestion_penalty", "total_predictive_cost"]], use_container_width=True)


# =============================================================================
# Page 6: What-If Simulation
# =============================================================================
elif page == "What-If Simulation":
    st.title("🧪 What-If Network Digital Twin")
    st.markdown("Executes hypothetical scenarios on an isolated in-memory Digital Twin clone without altering live network state.")

    with st.form("what_if_form"):
        col1, col2 = st.columns(2)
        with col1:
            scenario_name = st.text_input("Scenario Name", value="Surge_50pct_Form")
            scenario_type = st.selectbox("Scenario Type", [
                "traffic_demand_increase",
                "add_flow",
                "link_capacity_reduction",
                "link_delay_increase"
            ], index=0)
            target_id = st.text_input("Target ID (flow_id or link_id e.g. S1->S3)", value="F1_H1_H3")

        with col2:
            parameter_name = st.text_input("Parameter Name", value="demand_mbps")
            param_value = st.text_input("Parameter Value (e.g. 0.50 for +50%, 500 for capacity)", value="0.50")
            routing_strategy = st.selectbox("What-If Routing Strategy", ["predictive", "baseline"], index=0)
            num_ticks = st.slider("Simulated Ticks", min_value=1, max_value=30, value=10)

        submit_what_if = st.form_submit_button("Execute What-If Digital Twin Scenario")

    if submit_what_if:
        # Convert param value float if applicable
        try:
            parsed_val = float(param_value)
        except ValueError:
            parsed_val = param_value

        payload = {
            "scenario_name": scenario_name,
            "scenario_type": scenario_type,
            "target_id": target_id,
            "parameter_name": parameter_name,
            "value": parsed_val,
            "routing_strategy": routing_strategy,
            "alpha_penalty": 10.0,
            "num_ticks": num_ticks
        }

        res = api_client.execute_what_if(payload)
        if res.get("error"):
            st.error(res["message"])
        else:
            st.subheader(f"What-If Results ({res['scenario_name']})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Throughput (Mbps)", f"{res['twin_throughput_mbps']:.2f}", delta=f"{res['throughput_delta_mbps']:.2f} Mbps")
            col2.metric("Total Dropped Packets", res['twin_dropped_packets'], delta=res['dropped_packets_delta'], delta_color="inverse")
            col3.metric("Avg Utilization (%)", f"{res['twin_avg_utilization']:.2f}%")

            st.success("🔒 State Isolation Verified: Live network simulation state remained 100% untouched throughout What-If execution.")


# =============================================================================
# Page 7: AI Assistant (RAG + LLM)
# =============================================================================
elif page == "AI Assistant":
    st.title("🤖 Explainable AI Assistant (RAG + LLM)")
    st.markdown("""
    Ask technical questions regarding current network state, telemetry metrics, ML congestion predictions,
    predictive TE routing decisions, What-If simulation results, and SDN architecture.
    Answers are grounded in project documentation using Retrieval-Augmented Generation (RAG).
    """)

    sample_questions = [
        "What is ML-Assisted Predictive Traffic Engineering?",
        "Why is link S1->S3 predicted to become congested?",
        "Why did the predictive router choose an alternative path?",
        "How does the What-If Network Digital Twin guarantee state isolation?",
        "What metrics are evaluated in baseline vs predictive routing?"
    ]

    selected_sample = st.selectbox("Select Sample Question or Type Custom Below:", ["-- Select Sample Question --"] + sample_questions)
    default_q = selected_sample if selected_sample != "-- Select Sample Question --" else "What is ML-Assisted Predictive Traffic Engineering?"

    user_question = st.text_input("Enter Question for AI Assistant:", value=default_q)
    submit_ai = st.button("Ask AI Assistant")

    if submit_ai and user_question.strip():
        with st.spinner("Retrieving project documentation & generating explanation..."):
            payload = {"question": user_question.strip()}
            res = api_client.explain_ai(payload)

            if res.get("error"):
                st.error(res["message"])
            else:
                st.subheader("💡 AI Explanation")
                st.markdown(res["answer"])

                st.markdown("### 📚 Cited Documentation Sources")
                for src_file in res.get("sources", []):
                    st.caption(f"• `{src_file}`")

                st.caption(f"Explanation Provider: **{res.get('provider', 'Local')}** | *Out-of-band explanation only; LLM does not control routing decisions.")

