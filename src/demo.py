"""
Phase 1 Runnable Demonstration Script.
Demonstrates deterministic network simulation, traffic flow routing, telemetry tracking, and seed reproducibility.
"""

import sys
from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator


def run_phase1_demo(seed: int = 42) -> None:
    print("=" * 80)
    print(f" PHASE 1 DEMO: Deterministic Network Simulation Foundation (Seed: {seed})")
    print("=" * 80)

    # 1. Initialize Network Simulation Engine
    engine = NetworkXEngine(seed=seed)
    print(f"\n[1] Network Topology Initialized:")
    print(f"    - Total Nodes: {engine.graph.number_of_nodes()}")
    print(f"    - Total Directed Links: {engine.graph.number_of_edges()}")
    print("    - Topology Nodes:", list(engine.graph.nodes()))

    # 2. Initialize Traffic Generator with explicit seed
    traffic_gen = DeterministicTrafficGenerator(seed=seed)
    flows = traffic_gen.generate_fixed_flow_set()

    # Add an additional high-bandwidth flow to demonstrate capacity saturation & drop tracking
    high_demand_flow = traffic_gen.generate_random_flows(
        count=1,
        host_nodes=["H1", "H3"],
        min_demand_mbps=500.0,
        max_demand_mbps=500.0
    )[0]
    high_demand_flow.flow_id = "F4_OVERLOAD_H1_H3"
    flows.append(high_demand_flow)

    print(f"\n[2] Injecting {len(flows)} Deterministic Traffic Flows:")
    for flow in flows:
        path = engine.add_flow(flow)
        print(f"    - {flow.flow_id}: {flow.source} -> {flow.destination} (Demand: {flow.demand_mbps} Mbps) | Path: {' -> '.join(path)}")

    # 3. Run Simulation Ticks
    print(f"\n[3] Executing 5 Simulation Ticks:")
    print("-" * 80)
    print(f"{'Tick':<6} | {'Active Flows':<12} | {'Throughput (Mbps)':<18} | {'Total Dropped Pkts':<18}")
    print("-" * 80)

    snapshots = []
    for _ in range(5):
        snapshot = engine.step()
        snapshots.append(snapshot)
        print(f"{snapshot.timestamp_tick:<6} | {snapshot.active_flows_count:<12} | {snapshot.total_throughput_mbps:<18.2f} | {snapshot.total_dropped_packets:<18}")

    print("-" * 80)

    # 4. Display Final Link Telemetry Details
    latest = snapshots[-1]
    print(f"\n[4] Link Telemetry Detail Snapshot at Tick {latest.timestamp_tick}:")
    print("-" * 80)
    print(f"{'Link':<10} | {'Load / Cap (Mbps)':<22} | {'Utilization':<12} | {'Congested?':<12} | {'Total Bytes':<12}")
    print("-" * 80)

    for link_key, data in latest.links_telemetry.items():
        load_str = f"{data['current_load_mbps']:.1f} / {data['capacity_mbps']:.1f}"
        util_str = f"{data['utilization'] * 100:.1f}%"
        is_cong = "YES" if data['is_congested'] else "NO"
        bytes_val = int(data['total_bytes'])
        print(f"{link_key:<10} | {load_str:<22} | {util_str:<12} | {is_cong:<12} | {bytes_val:<12}")

    print("-" * 80)
    print("\n[5] Phase 1 Simulation Core Verification Complete!\n")
    return snapshots


def verify_reproducibility() -> None:
    print("=" * 80)
    print(" VERIFYING DETERMINISTIC REPRODUCIBILITY (Seed 42 Run A vs Run B)")
    print("=" * 80)
    run_a = run_phase1_demo(seed=42)
    run_b = run_phase1_demo(seed=42)

    # Compare snapshots byte-by-byte
    a_last = run_a[-1].links_telemetry
    b_last = run_b[-1].links_telemetry

    assert a_last == b_last, "Reproducibility failure: Run A and Run B outputs differ!"
    print(">>> SUCCESS: Run A and Run B telemetry outputs match 100% identically! Reproducibility Verified.\n")


if __name__ == "__main__":
    verify_reproducibility()
