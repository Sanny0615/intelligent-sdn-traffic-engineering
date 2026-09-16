"""
Data models representing routing candidate path evaluations, explainable routing decisions, and evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class CandidatePathEvaluation:
    """Evaluation breakdown for a single candidate path."""
    path: List[str]
    total_delay_ms: float
    predicted_congestion_penalty: float
    total_predictive_cost: float
    link_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Detailed explainable routing decision structure."""
    flow_id: str
    source: str
    destination: str
    demand_mbps: float
    baseline_path: List[str]
    selected_path: List[str]
    candidate_evaluations: List[CandidatePathEvaluation]
    reason: str
    is_rerouted: bool = False


@dataclass
class EvaluationMetrics:
    """Summary of measured simulation performance metrics for a routing strategy."""
    algorithm_name: str
    total_ticks: int
    average_utilization: float
    max_utilization: float
    total_throughput_mbps: float
    total_dropped_packets: int
    average_path_delay_ms: float
    congested_ticks: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics to dictionary."""
        return {
            "Algorithm": self.algorithm_name,
            "Total Ticks": self.total_ticks,
            "Avg Utilization (%)": round(self.average_utilization * 100, 2),
            "Max Utilization (%)": round(self.max_utilization * 100, 2),
            "Total Throughput (Mbps)": round(self.total_throughput_mbps, 2),
            "Total Dropped Packets": self.total_dropped_packets,
            "Avg Path Delay (ms)": round(self.average_path_delay_ms, 2),
            "Congested Ticks": self.congested_ticks
        }
