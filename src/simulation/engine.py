"""
NetworkX Simulation Engine powering network state, packet forwarding, and telemetry counter tracking.
"""

from typing import Dict, List, Optional
import networkx as nx

from src.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_PACKET_SIZE_BYTES,
    CONGESTION_THRESHOLD_UTILIZATION
)
from src.simulation.models import Link, TrafficFlow, TelemetrySnapshot
from src.simulation.topology import build_custom_6node_topology


class NetworkXEngine:
    """
    Core Network Simulation Engine using NetworkX graph representation.
    """

    def __init__(self, seed: int = DEFAULT_RANDOM_SEED):
        self.seed = seed
        self.current_tick: int = 0
        self.graph: nx.DiGraph = build_custom_6node_topology()
        self.active_flows: Dict[str, TrafficFlow] = {}

    def reset(self, seed: Optional[int] = None) -> None:
        """Resets network graph, simulation tick counter, active flows, and telemetry counters."""
        if seed is not None:
            self.seed = seed
        self.current_tick = 0
        self.active_flows.clear()
        self.graph = build_custom_6node_topology()

    def compute_shortest_path(self, source: str, destination: str) -> List[str]:
        """Computes Dijkstra shortest path based on static link propagation delays."""
        try:
            path = nx.dijkstra_path(self.graph, source=source, target=destination, weight="weight")
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as err:
            raise ValueError(f"No path found between {source} and {destination}: {err}")

    def add_flow(self, flow: TrafficFlow) -> List[str]:
        """
        Registers a new flow demand, computes its forwarding path, and adds it to active flows.
        """
        path = self.compute_shortest_path(flow.source, flow.destination)
        flow.path = path
        flow.active = True
        self.active_flows[flow.flow_id] = flow
        return path

    def remove_flow(self, flow_id: str) -> bool:
        """Removes a flow by flow_id."""
        if flow_id in self.active_flows:
            del self.active_flows[flow_id]
            return True
        return False

    def step(self, tick_duration_sec: float = 1.0) -> TelemetrySnapshot:
        """
        Advances the simulation by one tick:
        1. Resets per-tick link load accumulators.
        2. Routes active flows along assigned paths and updates link loads.
        3. Updates byte counters, packet counters, and packet drop counts.
        4. Increments tick counter and returns TelemetrySnapshot.
        """
        self.current_tick += 1

        # Reset per-tick link loads and flow counts
        for _, _, data in self.graph.edges(data=True):
            link: Link = data["object"]
            link.current_load_mbps = 0.0
            link.active_flows_count = 0

        # Accumulate flow demands along assigned paths
        for flow in self.active_flows.values():
            if not flow.active:
                continue

            for i in range(len(flow.path) - 1):
                u, v = flow.path[i], flow.path[i + 1]
                if self.graph.has_edge(u, v):
                    data = self.graph[u][v]
                    link: Link = data["object"]
                    link.current_load_mbps += flow.demand_mbps
                    link.active_flows_count += 1

        # Update cumulative telemetry counters (bytes, packets, drops)
        total_throughput_mbps = 0.0
        total_dropped_packets = 0

        for _, _, data in self.graph.edges(data=True):
            link: Link = data["object"]
            total_throughput_mbps += min(link.current_load_mbps, link.capacity_mbps)

            # Calculate bytes transmitted in this tick
            carried_load_mbps = min(link.current_load_mbps, link.capacity_mbps)
            bytes_sent = int((carried_load_mbps * 1e6 / 8.0) * tick_duration_sec)
            packets_sent = bytes_sent // DEFAULT_PACKET_SIZE_BYTES

            link.total_bytes += bytes_sent
            link.total_packets += packets_sent

            # Calculate dropped packets if link load exceeds link capacity
            if link.current_load_mbps > link.capacity_mbps:
                excess_mbps = link.current_load_mbps - link.capacity_mbps
                dropped_bytes = int((excess_mbps * 1e6 / 8.0) * tick_duration_sec)
                dropped_pkts = dropped_bytes // DEFAULT_PACKET_SIZE_BYTES
                link.dropped_packets += dropped_pkts
                total_dropped_packets += dropped_pkts

        return self.get_telemetry_snapshot()

    def get_telemetry_snapshot(self) -> TelemetrySnapshot:
        """Returns a snapshot of current telemetry metrics across all links."""
        links_data: Dict[str, Dict[str, float]] = {}
        total_throughput_mbps = 0.0
        total_dropped = 0

        for u, v, data in self.graph.edges(data=True):
            link: Link = data["object"]
            key = f"{u}->{v}"
            links_data[key] = {
                "source": u,
                "target": v,
                "capacity_mbps": link.capacity_mbps,
                "delay_ms": link.delay_ms,
                "current_load_mbps": link.current_load_mbps,
                "utilization": round(link.utilization, 4),
                "is_congested": float(link.is_congested),
                "total_bytes": float(link.total_bytes),
                "total_packets": float(link.total_packets),
                "dropped_packets": float(link.dropped_packets),
                "active_flows_count": float(link.active_flows_count),
            }
            total_throughput_mbps += min(link.current_load_mbps, link.capacity_mbps)
            total_dropped += link.dropped_packets

        return TelemetrySnapshot(
            timestamp_tick=self.current_tick,
            active_flows_count=len(self.active_flows),
            total_throughput_mbps=round(total_throughput_mbps, 2),
            total_dropped_packets=total_dropped,
            links_telemetry=links_data
        )
