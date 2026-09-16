"""
Baseline router implementing conventional Dijkstra Shortest Path First (SPF) based on static link delays.
"""

from typing import List
import networkx as nx


class BaselineRouter:
    """
    Conventional Shortest Path First (SPF) router using static link propagation delay as edge weight.
    """

    def __init__(self, weight_attribute: str = "delay"):
        self.weight_attribute = weight_attribute

    def compute_path(self, graph: nx.DiGraph, source: str, destination: str) -> List[str]:
        """
        Computes shortest path from source to destination using static Dijkstra.
        """
        try:
            path = nx.dijkstra_path(graph, source=source, target=destination, weight=self.weight_attribute)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound) as err:
            raise ValueError(f"No baseline path found between {source} and {destination}: {err}")
