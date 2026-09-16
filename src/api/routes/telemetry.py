"""
Telemetry route handler.
"""

from typing import List
from fastapi import APIRouter, Depends
from src.api.schemas import TelemetryResponse, LinkTelemetrySchema
from src.api.services import get_service_manager, APIServiceManager

router = APIRouter(tags=["Telemetry"])


@router.get("/telemetry", response_model=TelemetryResponse, summary="Retrieve Live Network Telemetry Snapshot")
def get_telemetry(services: APIServiceManager = Depends(get_service_manager)):
    """Returns structured current link telemetry metrics (current load, utilization, drops, congestion status)."""
    snapshot = services.engine.get_telemetry_snapshot()

    link_schemas: List[LinkTelemetrySchema] = []
    for link_key, data in snapshot.links_telemetry.items():
        link_schemas.append(LinkTelemetrySchema(
            link_id=link_key,
            source=data["source"],
            target=data["target"],
            capacity_mbps=float(data["capacity_mbps"]),
            current_load_mbps=float(data["current_load_mbps"]),
            utilization=float(data["utilization"]),
            is_congested=int(data["is_congested"]),
            total_bytes=float(data["total_bytes"]),
            dropped_packets=float(data["dropped_packets"])
        ))

    return TelemetryResponse(
        timestamp_tick=snapshot.timestamp_tick,
        active_flows_count=snapshot.active_flows_count,
        total_throughput_mbps=snapshot.total_throughput_mbps,
        total_dropped_packets=snapshot.total_dropped_packets,
        links=link_schemas
    )
