"""
Deterministic Traffic Generator supporting repeatable synthetic traffic flow demands.
"""

import random
from typing import List
from src.simulation.models import TrafficFlow


class DeterministicTrafficGenerator:
    """
    Generates deterministic synthetic traffic flows using an explicit random seed.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(self.seed)

    def reset_seed(self, seed: int = 42) -> None:
        """Reset the random number generator seed for exact reproducibility."""
        self.seed = seed
        self.rng = random.Random(self.seed)

    def generate_fixed_flow_set(self) -> List[TrafficFlow]:
        """
        Generates a predefined set of traffic flows for testing and debugging.
        Flow 1: H1 -> H3 (Demand: 600 Mbps)
        Flow 2: H2 -> H3 (Demand: 500 Mbps)
        Flow 3: H3 -> H1 (Demand: 200 Mbps)
        """
        return [
            TrafficFlow(
                flow_id="F1_H1_H3",
                source="H1",
                destination="H3",
                demand_mbps=600.0,
                start_tick=0,
                duration_ticks=10
            ),
            TrafficFlow(
                flow_id="F2_H2_H3",
                source="H2",
                destination="H3",
                demand_mbps=500.0,
                start_tick=1,
                duration_ticks=10
            ),
            TrafficFlow(
                flow_id="F3_H3_H1",
                source="H3",
                destination="H1",
                demand_mbps=200.0,
                start_tick=2,
                duration_ticks=10
            ),
        ]

    def generate_random_flows(
        self,
        count: int = 5,
        host_nodes: List[str] = None,
        min_demand_mbps: float = 100.0,
        max_demand_mbps: float = 800.0,
        start_tick: int = 0,
        duration_ticks: int = 10
    ) -> List[TrafficFlow]:
        """
        Generates N random traffic flows deterministically based on self.seed.
        """
        if host_nodes is None:
            host_nodes = ["H1", "H2", "H3"]

        flows: List[TrafficFlow] = []
        for i in range(count):
            src, dst = self.rng.sample(host_nodes, 2)
            demand = round(self.rng.uniform(min_demand_mbps, max_demand_mbps), 2)
            flow = TrafficFlow(
                flow_id=f"flow_{i+1}_{src}_{dst}",
                source=src,
                destination=dst,
                demand_mbps=demand,
                start_tick=start_tick,
                duration_ticks=duration_ticks
            )
            flows.append(flow)
        return flows

    def generate_dynamic_time_series_flows(self, num_ticks: int = 30) -> List[TrafficFlow]:
        """
        Generates a dynamic multi-tick flow schedule with varying load phases (light, moderate, bursty).
        Ensures realistic transitions between congested and non-congested link states over time.
        """
        hosts = ["H1", "H2", "H3"]
        flows: List[TrafficFlow] = []
        flow_counter = 1

        for tick in range(num_ticks):
            # Phase 1: Ticks 0-5 -> Light traffic (1-2 flows)
            if tick < 6:
                num_flows = self.rng.randint(1, 2)
                min_d, max_d = 100.0, 300.0
            # Phase 2: Ticks 6-15 -> Moderate traffic (2-3 flows)
            elif tick < 16:
                num_flows = self.rng.randint(2, 3)
                min_d, max_d = 300.0, 600.0
            # Phase 3: Ticks 16-24 -> Bursty/Heavy traffic (3-4 flows, high demand)
            elif tick < 25:
                num_flows = self.rng.randint(3, 4)
                min_d, max_d = 600.0, 1100.0
            # Phase 4: Ticks 25+ -> Light cooling traffic
            else:
                num_flows = self.rng.randint(1, 2)
                min_d, max_d = 150.0, 350.0

            for _ in range(num_flows):
                src, dst = self.rng.sample(hosts, 2)
                demand = round(self.rng.uniform(min_d, max_d), 2)
                duration = self.rng.randint(3, 8)
                flow = TrafficFlow(
                    flow_id=f"dyn_f{flow_counter}_{src}_{dst}_t{tick}",
                    source=src,
                    destination=dst,
                    demand_mbps=demand,
                    start_tick=tick,
                    duration_ticks=duration
                )
                flows.append(flow)
                flow_counter += 1

        return flows

