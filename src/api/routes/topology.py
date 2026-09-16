"""
Topology route handler.
"""

from typing import List
from fastapi import APIRouter, Depends
from src.api.schemas import TopologyResponse, NodeSchema, LinkSchema
from src.api.services import get_service_manager, APIServiceManager

router = APIRouter(tags=["Topology"])


@router.get("/topology", response_model=TopologyResponse, summary="Retrieve Network Topology Structure")
def get_topology(services: APIServiceManager = Depends(get_service_manager)):
    """Returns structured details about active network nodes, links, capacities, and propagation delays."""
    graph = services.engine.graph

    nodes: List[NodeSchema] = []
    for node_id, data in graph.nodes(data=True):
        nodes.append(NodeSchema(
            id=node_id,
            node_type=data.get("node_type", "unknown"),
            name=data.get("name", node_id)
        ))

    links: List[LinkSchema] = []
    for u, v, data in graph.edges(data=True):
        links.append(LinkSchema(
            source=u,
            target=v,
            capacity_mbps=float(data.get("capacity", 1000.0)),
            delay_ms=float(data.get("delay", 2.0))
        ))

    return TopologyResponse(
        status="success",
        nodes_count=graph.number_of_nodes(),
        edges_count=graph.number_of_edges(),
        nodes=nodes,
        links=links
    )
