"""
Data models representing What-If scenario definitions and simulation execution results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.routing.models import EvaluationMetrics


@dataclass
class WhatIfScenario:
    """
    Represents a hypothetical scenario configuration to test on the What-If Network Digital Twin.
    """
    name: str
    scenario_type: str  # 'traffic_demand_increase', 'add_flow', 'link_capacity_reduction', 'link_delay_increase'
    target_id: str      # flow_id or link_id (e.g. 'S1->S3')
    parameter_name: str
    value: Any


@dataclass
class WhatIfExecutionResult:
    """
    Comparative results returned after executing a What-If scenario on a cloned Digital Twin.
    """
    scenario_name: str
    routing_strategy: str
    num_simulated_ticks: int
    original_metrics: EvaluationMetrics
    twin_metrics: EvaluationMetrics
    impact_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns side-by-side comparative dictionary of original vs twin metrics."""
        orig = self.original_metrics.to_dict()
        twin = self.twin_metrics.to_dict()
        return {
            "Scenario Name": self.scenario_name,
            "Routing Strategy": self.routing_strategy,
            "Original Throughput (Mbps)": orig["Total Throughput (Mbps)"],
            "Twin Throughput (Mbps)": twin["Total Throughput (Mbps)"],
            "Throughput Delta (Mbps)": round(twin["Total Throughput (Mbps)"] - orig["Total Throughput (Mbps)"], 2),
            "Original Dropped Pkts": orig["Total Dropped Packets"],
            "Twin Dropped Pkts": twin["Total Dropped Packets"],
            "Dropped Pkts Delta": twin["Total Dropped Packets"] - orig["Total Dropped Packets"],
            "Original Avg Utilization (%)": orig["Avg Utilization (%)"],
            "Twin Avg Utilization (%)": twin["Avg Utilization (%)"],
            "Original Avg Delay (ms)": orig["Avg Path Delay (ms)"],
            "Twin Avg Delay (ms)": twin["Avg Path Delay (ms)"]
        }
