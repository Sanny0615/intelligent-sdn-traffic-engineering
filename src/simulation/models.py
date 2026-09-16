"""
Data models representing network entities, traffic flows, and telemetry snapshots.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Node:
    """Represents a network node (host or switch)."""
    id: str
    node_type: str  # 'host' or 'switch'
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.id


@dataclass
class Link:
    """Represents a directed link between two nodes in the network."""
    source: str
    target: str
    capacity_mbps: float
    delay_ms: float
    current_load_mbps: float = 0.0
    total_bytes: int = 0
    total_packets: int = 0
    dropped_packets: int = 0
    active_flows_count: int = 0

    @property
    def utilization(self) -> float:
        """Returns link utilization ratio in range [0.0, 1.0+]."""
        if self.capacity_mbps <= 0:
            return 0.0
        return self.current_load_mbps / self.capacity_mbps

    @property
    def is_congested(self) -> bool:
        """Returns True if utilization meets or exceeds 85% capacity."""
        return self.utilization >= 0.85


@dataclass
class TrafficFlow:
    """Represents a traffic flow demand between source and destination nodes."""
    flow_id: str
    source: str
    destination: str
    demand_mbps: float
    start_tick: int = 0
    duration_ticks: int = 10
    path: List[str] = field(default_factory=list)
    active: bool = True


@dataclass
class TelemetrySnapshot:
    """Represents network telemetry metrics captured at a specific simulation tick."""
    timestamp_tick: int
    active_flows_count: int
    total_throughput_mbps: float
    total_dropped_packets: int
    links_telemetry: Dict[str, Dict[str, float]] = field(default_factory=dict)
