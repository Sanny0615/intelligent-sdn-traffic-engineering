"""
ML-Assisted Predictive Traffic Engineering router component.
Computes dynamic predictive path costs combining static link delays and Phase 3 ML predicted congestion probabilities.
"""

from typing import List, Dict, Any, Optional
import networkx as nx
import pandas as pd

from src.ml.dataset import ML_FEATURE_COLUMNS
from src.ml.model import CongestionModel
from src.routing.baseline import BaselineRouter
from src.routing.models import CandidatePathEvaluation, RoutingDecision


class PredictiveTERouter:
    """
    ML-Assisted Predictive Traffic Engineering Router.
    Calculates dynamic link costs:
        link_cost = delay_ms + alpha * P(future_congestion_t+1)
    and selects candidate path minimizing total predictive path cost.
    """

    def __init__(self, model: CongestionModel, alpha_penalty: float = 10.0):
        self.model = model
        self.alpha_penalty = alpha_penalty
        self.baseline_router = BaselineRouter()

    def get_candidate_paths(
        self,
        graph: nx.DiGraph,
        source: str,
        destination: str,
        max_candidates: int = 5
    ) -> List[List[str]]:
        """
        Generates candidate simple paths between source and destination sorted by base propagation delay.
        """
        try:
            paths = list(nx.all_simple_paths(graph, source=source, target=destination))
            # Sort paths by total static propagation delay
            paths.sort(key=lambda p: sum(graph[p[i]][p[i+1]]["delay"] for i in range(len(p)-1)))
            return paths[:max_candidates]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def compute_predictive_path(
        self,
        graph: nx.DiGraph,
        source: str,
        destination: str,
        current_features_df: pd.DataFrame,
        flow_id: str = "FLOW_UNSPECIFIED",
        demand_mbps: float = 0.0
    ) -> RoutingDecision:
        """
        Calculates predictive path cost for all candidate paths and selects optimal path.
        Uses ONLY information available at current tick t (no future data leakage).
        """
        # 1. Baseline path computation
        baseline_path = self.baseline_router.compute_path(graph, source, destination)

        # 2. Extract current tick link predictions
        link_predictions: Dict[str, Dict[str, Any]] = {}
        for u, v, data in graph.edges(data=True):
            link_key = f"{u}->{v}"
            link_delay = float(data.get("delay", 2.0))
            link_capacity = float(data.get("capacity", 1000.0))

            # Query current features for link_key
            link_rows = current_features_df[current_features_df["link_id"] == link_key]
            prob_congested = 0.0

            if not link_rows.empty and self.model.is_fitted:
                # Use latest tick row for this link
                latest_row = link_rows.iloc[-1]
                feature_dict = {col: float(latest_row[col]) for col in ML_FEATURE_COLUMNS if col in latest_row}
                input_df = pd.DataFrame([feature_dict])

                probas = self.model.predict_proba(input_df)[0]
                pos_idx = list(self.model.model.classes_).index(1) if 1 in self.model.model.classes_ else 0
                prob_congested = float(probas[pos_idx]) if pos_idx < len(probas) else 0.0

            pred_penalty = self.alpha_penalty * prob_congested
            predictive_cost = link_delay + pred_penalty

            link_predictions[link_key] = {
                "link_id": link_key,
                "delay_ms": link_delay,
                "capacity_mbps": link_capacity,
                "predicted_congestion_prob": round(prob_congested, 4),
                "predicted_penalty": round(pred_penalty, 4),
                "predictive_cost": round(predictive_cost, 4)
            }

        # 3. Generate candidate paths
        candidate_paths = self.get_candidate_paths(graph, source, destination)
        if not candidate_paths:
            candidate_paths = [baseline_path]

        # 4. Evaluate each candidate path
        evaluations: List[CandidatePathEvaluation] = []
        for path in candidate_paths:
            path_delay = 0.0
            path_penalty = 0.0
            link_details: List[Dict[str, Any]] = []

            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                link_key = f"{u}->{v}"
                pred_info = link_predictions.get(link_key, {
                    "delay_ms": 2.0, "predicted_congestion_prob": 0.0, "predicted_penalty": 0.0, "predictive_cost": 2.0
                })

                path_delay += pred_info["delay_ms"]
                path_penalty += pred_info["predicted_penalty"]
                link_details.append(pred_info)

            total_cost = path_delay + path_penalty
            evaluations.append(CandidatePathEvaluation(
                path=path,
                total_delay_ms=round(path_delay, 2),
                predicted_congestion_penalty=round(path_penalty, 2),
                total_predictive_cost=round(total_cost, 2),
                link_details=link_details
            ))

        # 5. Select path with lowest total predictive cost
        best_eval = min(evaluations, key=lambda ev: ev.total_predictive_cost)
        selected_path = best_eval.path
        is_rerouted = (selected_path != baseline_path)

        # 6. Formulate detailed decision explanation
        if is_rerouted:
            reason = (
                f"Rerouted flow {flow_id} via {'->'.join(selected_path)} (Cost: {best_eval.total_predictive_cost:.2f}) "
                f"to bypass predicted congestion on baseline path {'->'.join(baseline_path)}."
            )
        else:
            reason = (
                f"Selected primary shortest path {'->'.join(selected_path)} (Cost: {best_eval.total_predictive_cost:.2f}). "
                f"No significant congestion risk predicted on baseline path."
            )

        return RoutingDecision(
            flow_id=flow_id,
            source=source,
            destination=destination,
            demand_mbps=demand_mbps,
            baseline_path=baseline_path,
            selected_path=selected_path,
            candidate_evaluations=evaluations,
            reason=reason,
            is_rerouted=is_rerouted
        )
