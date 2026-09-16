"""
What-If Scenario application module.
Applies hypothetical traffic or link state overrides to a cloned NetworkXEngine instance.
"""

from typing import Tuple
from src.simulation.engine import NetworkXEngine
from src.simulation.models import TrafficFlow, Link
from src.twin.models import WhatIfScenario


def parse_link_id(link_id: str) -> Tuple[str, str]:
    """Parses a link ID string like 'S1->S3' into source and target node IDs ('S1', 'S3')."""
    if "->" in link_id:
        parts = link_id.split("->")
        return parts[0].strip(), parts[1].strip()
    raise ValueError(f"Invalid link_id format '{link_id}'. Expected format 'source->target'.")


def apply_scenario(engine: NetworkXEngine, scenario: WhatIfScenario) -> None:
    """
    Applies scenario override directly to the provided NetworkXEngine instance.
    """
    stype = scenario.scenario_type.lower()

    if stype == "traffic_demand_increase":
        # Target ID can be a specific flow_id or "ALL"
        factor = float(scenario.value)  # e.g., 0.50 for +50%
        if scenario.target_id.upper() == "ALL":
            for flow in engine.active_flows.values():
                flow.demand_mbps = round(flow.demand_mbps * (1.0 + factor), 2)
        elif scenario.target_id in engine.active_flows:
            flow = engine.active_flows[scenario.target_id]
            flow.demand_mbps = round(flow.demand_mbps * (1.0 + factor), 2)
        else:
            # If target_id not active yet, scale any matching starting flow
            for flow in engine.active_flows.values():
                if scenario.target_id in flow.flow_id:
                    flow.demand_mbps = round(flow.demand_mbps * (1.0 + factor), 2)

    elif stype == "add_flow":
        if isinstance(scenario.value, TrafficFlow):
            engine.add_flow(scenario.value)
        elif isinstance(scenario.value, dict):
            flow = TrafficFlow(
                flow_id=scenario.value.get("flow_id", f"whatif_flow_{scenario.target_id}"),
                source=scenario.value["source"],
                destination=scenario.value["destination"],
                demand_mbps=float(scenario.value["demand_mbps"]),
                start_tick=engine.current_tick,
                duration_ticks=int(scenario.value.get("duration_ticks", 10))
            )
            engine.add_flow(flow)

    elif stype == "link_capacity_reduction":
        u, v = parse_link_id(scenario.target_id)
        if engine.graph.has_edge(u, v):
            new_capacity = float(scenario.value)
            engine.graph[u][v]["capacity"] = new_capacity
            link_obj: Link = engine.graph[u][v]["object"]
            link_obj.capacity_mbps = new_capacity

    elif stype == "link_delay_increase":
        u, v = parse_link_id(scenario.target_id)
        if engine.graph.has_edge(u, v):
            new_delay = float(scenario.value)
            engine.graph[u][v]["delay"] = new_delay
            engine.graph[u][v]["weight"] = new_delay
            link_obj: Link = engine.graph[u][v]["object"]
            link_obj.delay_ms = new_delay

    else:
        raise ValueError(f"Unsupported scenario type: '{scenario.scenario_type}'")
