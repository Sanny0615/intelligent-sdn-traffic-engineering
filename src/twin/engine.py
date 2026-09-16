"""
What-If Network Digital Twin module.
Provides in-memory deep-cloning of live simulation state and isolated What-If scenario execution.
"""

import copy
from typing import Optional, List, Dict, Any
import numpy as np

from src.simulation.engine import NetworkXEngine
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor
from src.ml.model import CongestionModel
from src.routing.baseline import BaselineRouter
from src.routing.predictive import PredictiveTERouter
from src.routing.models import EvaluationMetrics
from src.twin.models import WhatIfScenario, WhatIfExecutionResult
from src.twin.scenarios import apply_scenario


def create_digital_twin(live_engine: NetworkXEngine) -> NetworkXEngine:
    """
    Creates an independent in-memory deep copy of the current live NetworkXEngine.
    Modifying the returned twin engine leaves live_engine 100% unchanged.
    """
    twin_engine = NetworkXEngine(seed=live_engine.seed)
    twin_engine.current_tick = live_engine.current_tick
    twin_engine.graph = copy.deepcopy(live_engine.graph)
    twin_engine.active_flows = copy.deepcopy(live_engine.active_flows)
    return twin_engine


class WhatIfNetworkDigitalTwin:
    """
    What-If Network Digital Twin controller.
    Runs hypothetical scenarios on isolated cloned twin instances without altering live network state.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def execute_scenario(
        self,
        live_engine: NetworkXEngine,
        scenario: WhatIfScenario,
        routing_strategy: str = "predictive",
        model: Optional[CongestionModel] = None,
        alpha_penalty: float = 10.0,
        num_ticks: int = 10
    ) -> WhatIfExecutionResult:
        """
        Executes a What-If scenario on a cloned Digital Twin:
        1. Deep-clones live_engine -> twin_engine.
        2. Applies scenario modification to twin_engine.
        3. Executes simulation on twin_engine using requested routing strategy (baseline vs predictive).
        4. Calculates metrics and returns WhatIfExecutionResult.
        5. Live engine state is guaranteed to remain 100% untouched.
        """
        # Record baseline metrics from live engine prior to scenario
        snapshot_orig = live_engine.get_telemetry_snapshot()
        orig_utils = [d["utilization"] for d in snapshot_orig.links_telemetry.values()]
        original_metrics = EvaluationMetrics(
            algorithm_name=f"Original Live State ({routing_strategy})",
            total_ticks=num_ticks,
            average_utilization=float(np.mean(orig_utils)) if orig_utils else 0.0,
            max_utilization=float(np.max(orig_utils)) if orig_utils else 0.0,
            total_throughput_mbps=snapshot_orig.total_throughput_mbps,
            total_dropped_packets=snapshot_orig.total_dropped_packets,
            average_path_delay_ms=6.0,
            congested_ticks=1 if any(u >= 0.85 for u in orig_utils) else 0
        )

        # 1. Create twin
        twin_engine = create_digital_twin(live_engine)

        # 2. Apply scenario to twin
        apply_scenario(twin_engine, scenario)

        # 3. Setup routers
        baseline_router = BaselineRouter()
        predictive_router = PredictiveTERouter(model=model, alpha_penalty=alpha_penalty) if model else None

        collector_twin = TelemetryCollector()
        extractor_twin = FeatureExtractor()

        twin_utils: List[float] = []
        twin_max_utils: List[float] = []
        twin_throughputs: List[float] = []
        twin_delays: List[float] = []
        twin_dropped_pkts = 0
        twin_congested_ticks = 0

        # 4. Simulate scenario ticks on twin
        for tick_step in range(num_ticks):
            current_features = extractor_twin.to_dataframe(collector_twin.get_history())

            # Apply routing strategy to twin flows
            for fid, active_flow in list(twin_engine.active_flows.items()):
                if routing_strategy == "predictive" and predictive_router and not current_features.empty:
                    decision = predictive_router.compute_predictive_path(
                        graph=twin_engine.graph,
                        source=active_flow.source,
                        destination=active_flow.destination,
                        current_features_df=current_features,
                        flow_id=active_flow.flow_id,
                        demand_mbps=active_flow.demand_mbps
                    )
                    active_flow.path = decision.selected_path
                else:
                    path = baseline_router.compute_path(twin_engine.graph, active_flow.source, active_flow.destination)
                    active_flow.path = path

            # Step twin simulation & collect snapshot
            snap_twin = twin_engine.step()
            collector_twin.collect(snap_twin)

            # Record metrics
            utils = [d["utilization"] for d in snap_twin.links_telemetry.values()]
            twin_utils.append(float(np.mean(utils)))
            twin_max_utils.append(float(np.max(utils)))
            twin_throughputs.append(snap_twin.total_throughput_mbps)
            twin_dropped_pkts += snap_twin.total_dropped_packets

            tick_delays = []
            for fl in twin_engine.active_flows.values():
                path_delay = sum(twin_engine.graph[fl.path[i]][fl.path[i+1]]["delay"] for i in range(len(fl.path)-1))
                tick_delays.append(path_delay)
            twin_delays.append(float(np.mean(tick_delays)) if tick_delays else 6.0)

            if any(u >= 0.85 for u in utils):
                twin_congested_ticks += 1

        twin_metrics = EvaluationMetrics(
            algorithm_name=f"What-If Twin ({scenario.name} - {routing_strategy})",
            total_ticks=num_ticks,
            average_utilization=float(np.mean(twin_utils)),
            max_utilization=float(np.max(twin_max_utils)),
            total_throughput_mbps=float(np.mean(twin_throughputs)),
            total_dropped_packets=twin_dropped_pkts,
            average_path_delay_ms=float(np.mean(twin_delays)),
            congested_ticks=twin_congested_ticks
        )

        impact_summary = {
            "throughput_delta_mbps": round(twin_metrics.total_throughput_mbps - original_metrics.total_throughput_mbps, 2),
            "dropped_packets_delta": twin_metrics.total_dropped_packets - original_metrics.total_dropped_packets,
            "avg_utilization_delta": round(twin_metrics.average_utilization - original_metrics.average_utilization, 4),
            "congested_ticks_delta": twin_metrics.congested_ticks - original_metrics.congested_ticks
        }

        return WhatIfExecutionResult(
            scenario_name=scenario.name,
            routing_strategy=routing_strategy,
            num_simulated_ticks=num_ticks,
            original_metrics=original_metrics,
            twin_metrics=twin_metrics,
            impact_summary=impact_summary
        )
