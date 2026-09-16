"""
What-If Digital Twin Scenario route handler.
"""

from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import WhatIfRequest, WhatIfResponse
from src.api.services import get_service_manager, APIServiceManager
from src.twin.models import WhatIfScenario
from src.twin.engine import WhatIfNetworkDigitalTwin

router = APIRouter(tags=["What-If Digital Twin"])


@router.post("/simulation/what-if", response_model=WhatIfResponse, summary="Execute Hypothetical What-If Scenario on Digital Twin")
def execute_what_if(
    req: WhatIfRequest,
    services: APIServiceManager = Depends(get_service_manager)
):
    """
    Executes a hypothetical What-If scenario on an isolated in-memory Digital Twin clone.
    Returns comparative metric summaries (throughput delta, packet drop delta, utilization delta)
    while guaranteeing that live network simulation state remains 100% untouched.
    """
    scenario = WhatIfScenario(
        name=req.scenario_name,
        scenario_type=req.scenario_type,
        target_id=req.target_id,
        parameter_name=req.parameter_name,
        value=req.value
    )

    twin_controller = WhatIfNetworkDigitalTwin(seed=services.seed)

    try:
        result = twin_controller.execute_scenario(
            live_engine=services.engine,
            scenario=scenario,
            routing_strategy=req.routing_strategy,
            model=services.ml_model,
            alpha_penalty=req.alpha_penalty,
            num_ticks=req.num_ticks
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Scenario execution failed: {str(err)}")

    res_dict = result.to_dict()

    return WhatIfResponse(
        scenario_name=result.scenario_name,
        routing_strategy=result.routing_strategy,
        num_simulated_ticks=result.num_simulated_ticks,
        original_throughput_mbps=res_dict["Original Throughput (Mbps)"],
        twin_throughput_mbps=res_dict["Twin Throughput (Mbps)"],
        throughput_delta_mbps=res_dict["Throughput Delta (Mbps)"],
        original_dropped_packets=res_dict["Original Dropped Pkts"],
        twin_dropped_packets=res_dict["Twin Dropped Pkts"],
        dropped_packets_delta=res_dict["Dropped Pkts Delta"],
        original_avg_utilization=res_dict["Original Avg Utilization (%)"],
        twin_avg_utilization=res_dict["Twin Avg Utilization (%)"],
        original_avg_delay_ms=res_dict["Original Avg Delay (ms)"],
        twin_avg_delay_ms=res_dict["Twin Avg Delay (ms)"]
    )
