"""
Phase 4 Runnable Demonstration Script.
Demonstrates Baseline Shortest Path vs ML-Assisted Predictive TE comparison under identical traffic inputs,
detailed routing decision explainability breakdown, and actual simulation performance metrics.
"""

from src.routing.evaluator import RoutingEvaluator


def run_phase4_demo(seed: int = 42, alpha_penalty: float = 10.0):
    print("=" * 85)
    print(f" PHASE 4 DEMO: ML-Assisted Predictive Traffic Engineering (Seed: {seed}, Alpha: {alpha_penalty})")
    print("=" * 85)

    # 1. Initialize Routing Evaluator
    evaluator = RoutingEvaluator(seed=seed, alpha_penalty=alpha_penalty)

    print("\n[1] Training Phase 3 Random Forest Model on Telemetry...")
    model = evaluator.train_predictive_model(num_warmup_ticks=30)
    print("    - CongestionModel ready for inference.")

    # 2. Run Comparative Experiments (Identical Traffic Workload & Random Seed)
    print("\n[2] Executing 25-Tick Comparative Simulation (Baseline SPF vs Predictive TE)...")
    base_metrics, pred_metrics, decisions_log, comp_df = evaluator.run_comparison(
        num_ticks=25, model=model
    )
    print("    - Simulation execution complete.")

    # 3. Display Detailed Routing Decision Explanation Sample
    print("\n[3] Sample Detailed Routing Decision & Explainability Breakdown:")
    print("=" * 85)

    # Find a rerouted decision if available, else first decision
    rerouted_decisions = [d for d in decisions_log if d.is_rerouted]
    sample_decision = rerouted_decisions[0] if rerouted_decisions else decisions_log[0]

    print(f"  Flow ID          : {sample_decision.flow_id}")
    print(f"  Route            : {sample_decision.source} -> {sample_decision.destination} (Demand: {sample_decision.demand_mbps} Mbps)")
    print(f"  Baseline Path    : {' -> '.join(sample_decision.baseline_path)}")
    print(f"  Selected Path    : {' -> '.join(sample_decision.selected_path)}")
    print(f"  Action           : {'[REROUTED]' if sample_decision.is_rerouted else '[DEFAULT SHORT-PATH]'}")
    print(f"  Reason           : {sample_decision.reason}")

    print("\n  Candidate Paths Predictive Cost Evaluation Breakdown:")
    print("-" * 85)
    print(f"  {'Candidate Path':<28} | {'Base Delay':<12} | {'Pred Penalty':<14} | {'Total Cost':<12}")
    print("-" * 85)
    for eval_cand in sample_decision.candidate_evaluations:
        path_str = " -> ".join(eval_cand.path)
        print(f"  {path_str:<28} | {eval_cand.total_delay_ms:<12.2f} | {eval_cand.predicted_congestion_penalty:<14.2f} | {eval_cand.total_predictive_cost:<12.2f}")
    print("-" * 85)

    # 4. Display Side-by-Side Measured Performance Metrics Table
    print("\n[4] Empirical Network Performance Comparison Table (Actual Simulation Measurements Only):")
    print("=" * 85)
    print(comp_df.to_string(index=False))
    print("=" * 85)

    print("\n[5] Phase 4 ML-Assisted Predictive Traffic Engineering Pipeline Complete!\n")
    return base_metrics, pred_metrics, sample_decision


if __name__ == "__main__":
    run_phase4_demo()
