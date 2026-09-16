"""
Pydantic Request and Response Schemas for the FastAPI service boundary.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


# -----------------------------------------------------------------------------
# Health Schemas
# -----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    simulation: str = "active"


# -----------------------------------------------------------------------------
# Topology Schemas
# -----------------------------------------------------------------------------
class NodeSchema(BaseModel):
    id: str
    node_type: str
    name: str


class LinkSchema(BaseModel):
    source: str
    target: str
    capacity_mbps: float
    delay_ms: float


class TopologyResponse(BaseModel):
    status: str = "success"
    nodes_count: int
    edges_count: int
    nodes: List[NodeSchema]
    links: List[LinkSchema]


# -----------------------------------------------------------------------------
# Telemetry Schemas
# -----------------------------------------------------------------------------
class LinkTelemetrySchema(BaseModel):
    link_id: str
    source: str
    target: str
    capacity_mbps: float
    current_load_mbps: float
    utilization: float
    is_congested: int
    total_bytes: float
    dropped_packets: float


class TelemetryResponse(BaseModel):
    timestamp_tick: int
    active_flows_count: int
    total_throughput_mbps: float
    total_dropped_packets: int
    links: List[LinkTelemetrySchema]


# -----------------------------------------------------------------------------
# Prediction Schemas
# -----------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    link_id: Optional[str] = "S1->S3"
    utilization: float = Field(..., ge=0.0, description="Current link utilization ratio")
    prev_utilization: float = Field(0.0, ge=0.0)
    delta_utilization: float = Field(0.0)
    current_load_mbps: float = Field(..., ge=0.0)
    capacity_mbps: float = Field(1000.0, gt=0.0)
    delay_ms: float = Field(2.0, gt=0.0)
    drop_rate: float = Field(0.0, ge=0.0)
    active_flows_count: float = Field(1.0, ge=0.0)


class PredictionResponse(BaseModel):
    link_id: Optional[str]
    predicted_congestion_class: int
    congestion_probability: float
    prediction_horizon: str = "t+1"


# -----------------------------------------------------------------------------
# Routing Schemas
# -----------------------------------------------------------------------------
class RouteRequest(BaseModel):
    source: str
    destination: str
    demand_mbps: float = Field(..., gt=0.0, description="Traffic flow demand in Mbps")
    strategy: str = Field("predictive", description="Routing strategy: 'baseline' or 'predictive'")
    alpha_penalty: float = Field(10.0, ge=0.0)
    flow_id: Optional[str] = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        s = v.lower()
        if s not in ["baseline", "predictive"]:
            raise ValueError("Routing strategy must be either 'baseline' or 'predictive'")
        return s


class CandidateEvaluationSchema(BaseModel):
    path: List[str]
    total_delay_ms: float
    predicted_congestion_penalty: float
    total_predictive_cost: float


class RouteResponse(BaseModel):
    flow_id: str
    source: str
    destination: str
    demand_mbps: float
    strategy: str
    baseline_path: List[str]
    selected_path: List[str]
    is_rerouted: bool
    reason: str
    candidate_evaluations: List[CandidateEvaluationSchema]


# -----------------------------------------------------------------------------
# What-If Scenario Schemas
# -----------------------------------------------------------------------------
class WhatIfRequest(BaseModel):
    scenario_name: str = Field(..., min_length=1)
    scenario_type: str = Field(..., description="Scenario type: 'traffic_demand_increase', 'add_flow', 'link_capacity_reduction', 'link_delay_increase'")
    target_id: str = Field(..., min_length=1, description="Target flow_id or link_id (e.g., 'S1->S3')")
    parameter_name: str = Field(..., min_length=1)
    value: Any
    routing_strategy: str = Field("predictive", description="'baseline' or 'predictive'")
    alpha_penalty: float = Field(10.0, ge=0.0)
    num_ticks: int = Field(10, gt=0, le=100)

    @field_validator("scenario_type")
    @classmethod
    def validate_scenario_type(cls, v: str) -> str:
        valid_types = [
            "traffic_demand_increase",
            "add_flow",
            "link_capacity_reduction",
            "link_delay_increase"
        ]
        s = v.lower()
        if s not in valid_types:
            raise ValueError(f"Invalid scenario_type '{v}'. Must be one of {valid_types}")
        return s


class WhatIfResponse(BaseModel):
    scenario_name: str
    routing_strategy: str
    num_simulated_ticks: int
    original_throughput_mbps: float
    twin_throughput_mbps: float
    throughput_delta_mbps: float
    original_dropped_packets: int
    twin_dropped_packets: int
    dropped_packets_delta: int
    original_avg_utilization: float
    twin_avg_utilization: float
    original_avg_delay_ms: float
    twin_avg_delay_ms: float


# -----------------------------------------------------------------------------
# AI RAG Explanation Schemas
# -----------------------------------------------------------------------------
class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question regarding SDN, telemetry, ML, or routing decisions.")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Optional structured simulation or decision facts.")


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    provider: str = "Local"

