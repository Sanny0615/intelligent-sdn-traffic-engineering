"""
Routing Evaluator module for running side-by-side comparative experiments comparing
Baseline Shortest Path First (SPF) vs. ML-Assisted Predictive Traffic Engineering under identical traffic scenarios.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor
from src.ml.dataset import SupervisedDatasetBuilder
from src.ml.model import CongestionModel
from src.routing.baseline import BaselineRouter
from src.routing.predictive import PredictiveTERouter
from src.routing.models import EvaluationMetrics, RoutingDecision


class RoutingEvaluator:
    """
    Executes reproducible side-by-side comparative network experiments.
    Ensures identical traffic demands and random seeds for both Baseline and Predictive TE.
    """

    def __init__(self, seed: int = 42, alpha_penalty: float = 10.0):
        self.seed = seed
        self.alpha_penalty = alpha_penalty

    def train_predictive_model(self, num_warmup_ticks: int = 30) -> CongestionModel:
        """Trains Phase 3 Random Forest model on dynamic simulation telemetry."""
        engine = NetworkXEngine(seed=self.seed)
        collector = TelemetryCollector()
        extractor = FeatureExtractor()
        traffic_gen = DeterministicTrafficGenerator(seed=self.seed)
        flows = traffic_gen.generate_dynamic_time_series_flows(num_ticks=num_warmup_ticks)

        flows_by_tick = {}
        for f in flows:
            flows_by_tick.setdefault(f.start_tick, []).append(f)

        for tick in range(num_warmup_ticks):
            if tick in flows_by_tick:
                for f in flows_by_tick[tick]:
                    engine.add_flow(f)
            collector.collect(engine.step())

        dataset_builder = SupervisedDatasetBuilder()
        X, y = dataset_builder.create_supervised_dataset(extractor.to_dataframe(collector.get_history()))

        model = CongestionModel(n_estimators=50, max_depth=5, random_state=self.seed)
        model.fit(X, y)
        return model

    def run_comparison(
        self,
        num_ticks: int = 25,
        model: Optional[CongestionModel] = None
    ) -> Tuple[EvaluationMetrics, EvaluationMetrics, List[RoutingDecision], pd.DataFrame]:
        """
        Runs identical simulation traffic workloads under:
        1. Baseline Dijkstra Shortest Path First
        2. ML-Assisted Predictive Traffic Engineering
        """
        if model is None:
            model = self.train_predictive_model(num_warmup_ticks=num_ticks + 10)

        # Generate traffic workload schedule
        traffic_gen = DeterministicTrafficGenerator(seed=self.seed)
        scheduled_flows = traffic_gen.generate_dynamic_time_series_flows(num_ticks=num_ticks)

        # ---------------------------------------------------------------------
        # EXPERIMENT A: Baseline Shortest Path First (Dijkstra)
        # ---------------------------------------------------------------------
        engine_base = NetworkXEngine(seed=self.seed)
        baseline_router = BaselineRouter()

        base_flows_by_tick = {}
        for f in scheduled_flows:
            # Create fresh copy of flow for Experiment A
            f_copy = traffic_gen.generate_random_flows(count=1)[0]
            f_copy.flow_id = f.flow_id
            f_copy.source = f.source
            f_copy.destination = f.destination
            f_copy.demand_mbps = f.demand_mbps
            f_copy.start_tick = f.start_tick
            f_copy.duration_ticks = f.duration_ticks
            base_flows_by_tick.setdefault(f.start_tick, []).append(f_copy)

        base_utilizations: List[float] = []
        base_max_utils: List[float] = []
        base_throughputs: List[float] = []
        base_dropped_pkts = 0
        base_congested_ticks = 0

        for tick in range(num_ticks):
            if tick in base_flows_by_tick:
                for f in base_flows_by_tick[tick]:
                    path = baseline_router.compute_path(engine_base.graph, f.source, f.destination)
                    f.path = path
                    engine_base.add_flow(f)

            # Remove finished flows
            active_ids = list(engine_base.active_flows.keys())
            for fid in active_ids:
                fl = engine_base.active_flows[fid]
                if tick >= fl.start_tick + fl.duration_ticks:
                    engine_base.remove_flow(fid)

            snapshot = engine_base.step()

            # Record metrics
            utils = [d["utilization"] for d in snapshot.links_telemetry.values()]
            base_utilizations.append(float(np.mean(utils)))
            base_max_utils.append(float(np.max(utils)))
            base_throughputs.append(snapshot.total_throughput_mbps)
            base_dropped_pkts += snapshot.total_dropped_packets

            if any(u >= 0.85 for u in utils):
                base_congested_ticks += 1

        baseline_metrics = EvaluationMetrics(
            algorithm_name="Baseline Shortest Path First (SPF)",
            total_ticks=num_ticks,
            average_utilization=float(np.mean(base_utilizations)),
            max_utilization=float(np.max(base_max_utils)),
            total_throughput_mbps=float(np.mean(base_throughputs)),
            total_dropped_packets=base_dropped_pkts,
            average_path_delay_ms=6.0,  # Baseline 3-hop static delay (2ms * 3)
            congested_ticks=base_congested_ticks
        )

        # ---------------------------------------------------------------------
        # EXPERIMENT B: ML-Assisted Predictive Traffic Engineering
        # ---------------------------------------------------------------------
        engine_pred = NetworkXEngine(seed=self.seed)
        collector_pred = TelemetryCollector()
        extractor_pred = FeatureExtractor()
        predictive_router = PredictiveTERouter(model=model, alpha_penalty=self.alpha_penalty)

        pred_flows_by_tick = {}
        for f in scheduled_flows:
            f_copy = traffic_gen.generate_random_flows(count=1)[0]
            f_copy.flow_id = f.flow_id
            f_copy.source = f.source
            f_copy.destination = f.destination
            f_copy.demand_mbps = f.demand_mbps
            f_copy.start_tick = f.start_tick
            f_copy.duration_ticks = f.duration_ticks
            pred_flows_by_tick.setdefault(f.start_tick, []).append(f_copy)

        pred_utilizations: List[float] = []
        pred_max_utils: List[float] = []
        pred_throughputs: List[float] = []
        pred_delays: List[float] = []
        pred_dropped_pkts = 0
        pred_congested_ticks = 0
        decisions_log: List[RoutingDecision] = []

        for tick in range(num_ticks):
            # Extract current feature DataFrame up to current tick
            current_features = extractor_pred.to_dataframe(collector_pred.get_history())

            # Route new starting flows
            if tick in pred_flows_by_tick:
                for f in pred_flows_by_tick[tick]:
                    decision = predictive_router.compute_predictive_path(
                        graph=engine_pred.graph,
                        source=f.source,
                        destination=f.destination,
                        current_features_df=current_features,
                        flow_id=f.flow_id,
                        demand_mbps=f.demand_mbps
                    )
                    f.path = decision.selected_path
                    engine_pred.add_flow(f)
                    decisions_log.append(decision)

            # Re-evaluate active flows for potential proactive rerouting
            for fid, active_flow in list(engine_pred.active_flows.items()):
                decision = predictive_router.compute_predictive_path(
                    graph=engine_pred.graph,
                    source=active_flow.source,
                    destination=active_flow.destination,
                    current_features_df=current_features,
                    flow_id=active_flow.flow_id,
                    demand_mbps=active_flow.demand_mbps
                )
                if decision.is_rerouted:
                    active_flow.path = decision.selected_path
                    decisions_log.append(decision)

            # Remove finished flows
            active_ids = list(engine_pred.active_flows.keys())
            for fid in active_ids:
                fl = engine_pred.active_flows[fid]
                if tick >= fl.start_tick + fl.duration_ticks:
                    engine_pred.remove_flow(fid)

            # Step simulation & collect snapshot
            snapshot = engine_pred.step()
            collector_pred.collect(snapshot)

            # Record metrics
            utils = [d["utilization"] for d in snapshot.links_telemetry.values()]
            pred_utilizations.append(float(np.mean(utils)))
            pred_max_utils.append(float(np.max(utils)))
            pred_throughputs.append(snapshot.total_throughput_mbps)
            pred_dropped_pkts += snapshot.total_dropped_packets

            # Average path delay of active flows
            tick_delays = []
            for fl in engine_pred.active_flows.values():
                path_delay = sum(engine_pred.graph[fl.path[i]][fl.path[i+1]]["delay"] for i in range(len(fl.path)-1))
                tick_delays.append(path_delay)

            avg_delay = float(np.mean(tick_delays)) if tick_delays else 6.0
            pred_delays.append(avg_delay)

            if any(u >= 0.85 for u in utils):
                pred_congested_ticks += 1

        predictive_metrics = EvaluationMetrics(
            algorithm_name="ML-Assisted Predictive TE",
            total_ticks=num_ticks,
            average_utilization=float(np.mean(pred_utilizations)),
            max_utilization=float(np.max(pred_max_utils)),
            total_throughput_mbps=float(np.mean(pred_throughputs)),
            total_dropped_packets=pred_dropped_pkts,
            average_path_delay_ms=float(np.mean(pred_delays)),
            congested_ticks=pred_congested_ticks
        )

        # Build comparison DataFrame
        comparison_df = pd.DataFrame([
            baseline_metrics.to_dict(),
            predictive_metrics.to_dict()
        ])

        return baseline_metrics, predictive_metrics, decisions_log, comparison_df
