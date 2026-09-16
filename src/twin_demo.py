"""
Phase 5 Runnable Demonstration Script.
Demonstrates What-If Network Digital Twin state cloning, scenario simulation (50% demand surge),
Baseline vs Predictive TE comparison, and state isolation proof.
"""

import pandas as pd
from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.routing.evaluator import RoutingEvaluator
from src.twin.models import WhatIfScenario
from src.twin.engine import WhatIfNetworkDigitalTwin, create_digital_twin


def run_phase5_demo(seed: int = 42):
    print("=" * 85)
    print(f" PHASE 5 DEMO: What-If Network Digital Twin Scenario Simulation (Seed: {seed})")
    print("=" * 85)

    # 1. Initialize Live Engine & Run Warmup Ticks
    live_engine = NetworkXEngine(seed=seed)
    traffic_gen = DeterministicTrafficGenerator(seed=seed)

    flows = traffic_gen.generate_fixed_flow_set()
    for f in flows:
        live_engine.add_flow(f)

    for _ in range(3):
        live_engine.step()

    # Capture live engine state prior to What-If scenario
    live_tick_before = live_engine.current_tick
    live_flows_before = len(live_engine.active_flows)
    live_bytes_before = sum(link.total_bytes for _, _, data in live_engine.graph.edges(data=True) for link in [data["object"]])

    print("\n[1] Live Network Simulation State Established:")
    print(f"    - Current Tick      : {live_tick_before}")
    print(f"    - Active Flow Count : {live_flows_before}")
    print(f"    - Total Link Bytes  : {live_bytes_before:,}")

    # 2. Train Phase 3/4 Model
    evaluator = RoutingEvaluator(seed=seed, alpha_penalty=10.0)
    model = evaluator.train_predictive_model(num_warmup_ticks=20)
    print("    - ML CongestionModel ready for Digital Twin inference.")

    # 3. Define What-If Scenario: "Increase H1 -> H3 traffic demand by 50%"
    scenario = WhatIfScenario(
        name="Surge_H1_H3_Plus50Pct",
        scenario_type="traffic_demand_increase",
        target_id="F1_H1_H3",
        parameter_name="demand_mbps",
        value=0.50  # +50% increase
    )

    print(f"\n[2] Defined What-If Scenario: '{scenario.name}'")
    print(f"    - Type            : {scenario.scenario_type}")
    print(f"    - Target Flow ID  : {scenario.target_id}")
    print(f"    - Parameter Change: {scenario.parameter_name} (+{int(scenario.value * 100)}%)")

    # 4. Instantiate Digital Twin Controller & Execute Scenario
    twin_controller = WhatIfNetworkDigitalTwin(seed=seed)

    print("\n[3] Executing What-If Scenario on Digital Twin (Baseline SPF)...")
    res_base = twin_controller.execute_scenario(
        live_engine=live_engine,
        scenario=scenario,
        routing_strategy="baseline",
        model=None,
        num_ticks=10
    )

    print("\n[4] Executing What-If Scenario on Digital Twin (ML-Assisted Predictive TE)...")
    res_pred = twin_controller.execute_scenario(
        live_engine=live_engine,
        scenario=scenario,
        routing_strategy="predictive",
        model=model,
        alpha_penalty=10.0,
        num_ticks=10
    )

    # 5. Display Comparative Results Tables
    print("\n[5] What-If Scenario Impact Comparison Table (Baseline SPF vs Predictive TE):")
    print("=" * 85)
    df_comp = pd.DataFrame([res_base.to_dict(), res_pred.to_dict()])
    print(df_comp.to_string(index=False))
    print("=" * 85)

    # 6. Explicit State Isolation Proof
    live_tick_after = live_engine.current_tick
    live_flows_after = len(live_engine.active_flows)
    live_bytes_after = sum(link.total_bytes for _, _, data in live_engine.graph.edges(data=True) for link in [data["object"]])

    print("\n[6] PROOF OF DIGITAL TWIN STATE ISOLATION:")
    print("-" * 85)
    print(f"  Live Engine State    | Before What-If | After What-If | Status")
    print("-" * 85)
    print(f"  Current Tick         | {live_tick_before:<14} | {live_tick_after:<13} | {'UNTOUCHED' if live_tick_before == live_tick_after else 'MUTATED!'}")
    print(f"  Active Flow Count    | {live_flows_before:<14} | {live_flows_after:<13} | {'UNTOUCHED' if live_flows_before == live_flows_after else 'MUTATED!'}")
    print(f"  Total Link Bytes     | {live_bytes_before:<14,} | {live_bytes_after:<13,} | {'UNTOUCHED' if live_bytes_before == live_bytes_after else 'MUTATED!'}")
    print("-" * 85)

    assert live_tick_before == live_tick_after
    assert live_flows_before == live_flows_after
    assert live_bytes_before == live_bytes_after

    print(">>> SUCCESS: Live network state remained 100% UNTOUCHED throughout What-If execution!\n")
    return res_base, res_pred


if __name__ == "__main__":
    run_phase5_demo()
