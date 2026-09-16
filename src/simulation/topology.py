"""
Topology abstraction and small deterministic 6-node custom topology builder.
"""

from typing import Tuple, List, Dict
import networkx as nx

from src.config import DEFAULT_BANDWIDTH_MBPS, DEFAULT_LINK_DELAY_MS
from src.simulation.models import Node, Link


def build_custom_6node_topology(
    bandwidth_mbps: float = DEFAULT_BANDWIDTH_MBPS,
    delay_ms: float = DEFAULT_LINK_DELAY_MS
) -> nx.DiGraph:
    """
    Constructs a small deterministic 6-node directed network graph.
    
    Topology Structure:
      Hosts: H1, H2, H3
      Switches: S1 (Edge 1), S2 (Edge 2), S3 (Core)
      
    Connections (Bidirectional directed pairs):
      - H1 <--> S1 (Access Link 1)
      - H2 <--> S1 (Access Link 2)
      - S1 <--> S2 (Inter-switch Edge Link)
      - S1 <--> S3 (Primary Core Link)
      - S2 <--> S3 (Secondary Core Link)
      - S3 <--> H3 (Destination Access Link)
      
    This topology provides multi-path choices between S1 and S3 (direct S1->S3 vs S1->S2->S3),
    ideal for traffic engineering, path selection, and congestion management testing.
    """
    graph = nx.DiGraph()

    # Define Nodes
    nodes = [
        Node(id="H1", node_type="host", name="Host 1"),
        Node(id="H2", node_type="host", name="Host 2"),
        Node(id="S1", node_type="switch", name="Switch 1 (Edge)"),
        Node(id="S2", node_type="switch", name="Switch 2 (Edge)"),
        Node(id="S3", node_type="switch", name="Switch 3 (Core)"),
        Node(id="H3", node_type="host", name="Host 3"),
    ]

    for node in nodes:
        graph.add_node(node.id, object=node, node_type=node.node_type, name=node.name)

    # Define Bidirectional Edges with Link properties
    edge_pairs: List[Tuple[str, str, float, float]] = [
        ("H1", "S1", bandwidth_mbps, delay_ms),
        ("S1", "H1", bandwidth_mbps, delay_ms),
        ("H2", "S1", bandwidth_mbps, delay_ms),
        ("S1", "H2", bandwidth_mbps, delay_ms),
        ("S1", "S2", bandwidth_mbps, delay_ms * 1.5),
        ("S2", "S1", bandwidth_mbps, delay_ms * 1.5),
        ("S1", "S3", bandwidth_mbps, delay_ms),
        ("S3", "S1", bandwidth_mbps, delay_ms),
        ("S2", "S3", bandwidth_mbps, delay_ms * 1.5),
        ("S3", "S2", bandwidth_mbps, delay_ms * 1.5),
        ("S3", "H3", bandwidth_mbps, delay_ms),
        ("H3", "S3", bandwidth_mbps, delay_ms),
    ]

    for src, dst, cap, dly in edge_pairs:
        link_obj = Link(source=src, target=dst, capacity_mbps=cap, delay_ms=dly)
        graph.add_edge(
            src,
            dst,
            capacity=cap,
            delay=dly,
            weight=dly,  # Default routing weight = propagation delay
            object=link_obj
        )

    return graph
