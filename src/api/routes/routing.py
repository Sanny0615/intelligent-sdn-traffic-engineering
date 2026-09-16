"""
Routing route handler.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import RouteRequest, RouteResponse, CandidateEvaluationSchema
from src.api.services import get_service_manager, APIServiceManager
from src.routing.baseline import BaselineRouter
from src.routing.predictive import PredictiveTERouter

router = APIRouter(tags=["Routing"])


@router.post("/routing/optimize", response_model=RouteResponse, summary="Optimize Traffic Engineering Path Selection")
def optimize_routing(
    req: RouteRequest,
    services: APIServiceManager = Depends(get_service_manager)
):
    """
    Computes optimal traffic path for a flow demand using Baseline SPF or ML-Assisted Predictive TE.
    Returns selected path, baseline path, candidate path evaluations, reroute indicator, and decision reason.
    """
    graph = services.engine.graph
    valid_nodes = set(graph.nodes())

    if req.source not in valid_nodes:
        raise HTTPException(status_code=400, detail=f"Source node '{req.source}' does not exist in topology.")
    if req.destination not in valid_nodes:
        raise HTTPException(status_code=400, detail=f"Destination node '{req.destination}' does not exist in topology.")

    flow_id = req.flow_id or f"flow_{req.source}_{req.destination}"

    current_features = services.extractor.to_dataframe(services.collector.get_history())

    if req.strategy.lower() == "baseline":
        baseline_router = BaselineRouter()
        base_path = baseline_router.compute_path(graph, req.source, req.destination)
        return RouteResponse(
            flow_id=flow_id,
            source=req.source,
            destination=req.destination,
            demand_mbps=req.demand_mbps,
            strategy="baseline",
            baseline_path=base_path,
            selected_path=base_path,
            is_rerouted=False,
            reason=f"Selected standard shortest path {' -> '.join(base_path)} via Baseline SPF.",
            candidate_evaluations=[CandidateEvaluationSchema(
                path=base_path,
                total_delay_ms=6.0,
                predicted_congestion_penalty=0.0,
                total_predictive_cost=6.0
            )]
        )

    # Predictive TE strategy
    predictive_router = PredictiveTERouter(model=services.ml_model, alpha_penalty=req.alpha_penalty)
    decision = predictive_router.compute_predictive_path(
        graph=graph,
        source=req.source,
        destination=req.destination,
        current_features_df=current_features,
        flow_id=flow_id,
        demand_mbps=req.demand_mbps
    )

    cand_schemas: List[CandidateEvaluationSchema] = []
    for cand in decision.candidate_evaluations:
        cand_schemas.append(CandidateEvaluationSchema(
            path=cand.path,
            total_delay_ms=cand.total_delay_ms,
            predicted_congestion_penalty=cand.predicted_congestion_penalty,
            total_predictive_cost=cand.total_predictive_cost
        ))

    return RouteResponse(
        flow_id=decision.flow_id,
        source=decision.source,
        destination=decision.destination,
        demand_mbps=decision.demand_mbps,
        strategy="predictive",
        baseline_path=decision.baseline_path,
        selected_path=decision.selected_path,
        is_rerouted=decision.is_rerouted,
        reason=decision.reason,
        candidate_evaluations=cand_schemas
    )
