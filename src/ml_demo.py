"""
Phase 3 Runnable Demonstration Script.
Demonstrates dynamic simulation telemetry collection, supervised dataset building (t -> t+1 target),
Random Forest training, empirical test set evaluation, feature importances, and live prediction.
"""

from sklearn.model_selection import train_test_split

from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor
from src.ml.dataset import SupervisedDatasetBuilder, ML_FEATURE_COLUMNS
from src.ml.model import CongestionModel
from src.ml.predictor import predict_link_congestion


def run_phase3_demo(seed: int = 42):
    print("=" * 80)
    print(f" PHASE 3 DEMO: ML-Based Future Congestion Prediction (Seed: {seed})")
    print("=" * 80)

    # 1. Initialize Engine, Collector, Extractor, Builder
    engine = NetworkXEngine(seed=seed)
    collector = TelemetryCollector()
    extractor = FeatureExtractor()
    dataset_builder = SupervisedDatasetBuilder()

    # 2. Inject Dynamic Traffic Workload Schedule over 30 Ticks
    traffic_gen = DeterministicTrafficGenerator(seed=seed)
    dynamic_flows = traffic_gen.generate_dynamic_time_series_flows(num_ticks=30)

    print(f"\n[1] Running Dynamic Simulation over 30 Ticks with {len(dynamic_flows)} Scheduled Flows...")

    # Group flows by start tick and inject as simulation progresses
    flows_by_start_tick = {}
    for flow in dynamic_flows:
        flows_by_start_tick.setdefault(flow.start_tick, []).append(flow)

    for tick in range(30):
        # Add new flows starting at this tick
        if tick in flows_by_start_tick:
            for flow in flows_by_start_tick[tick]:
                engine.add_flow(flow)

        # Remove finished flows
        active_ids = list(engine.active_flows.keys())
        for fid in active_ids:
            f = engine.active_flows[fid]
            if tick >= f.start_tick + f.duration_ticks:
                engine.remove_flow(fid)

        # Step simulation & collect snapshot
        snapshot = engine.step()
        collector.collect(snapshot)

    history = collector.get_history()
    print(f"    - Collected {len(history)} Telemetry Snapshots across 12 directed links.")

    # 3. Extract Feature DataFrame & Build Supervised Dataset
    print("\n[2] Extracting Feature DataFrame and Building Supervised Dataset (t -> t+1)...")
    raw_df = extractor.to_dataframe(history)
    X, y = dataset_builder.create_supervised_dataset(raw_df)

    class_counts = dict(y.value_counts().to_dict())
    print(f"    - Raw Telemetry Rows: {len(raw_df)}")
    print(f"    - Supervised Dataset Rows: {len(X)} (dropped last tick per link)")
    print(f"    - Class Distribution (0=Normal, 1=Congested): {class_counts}")

    if len(class_counts) < 2:
        print(">>> WARNING: Dataset contains only one target class! Cannot perform stratified split.")
        return

    # 4. Perform Reproducible Stratified Train / Test Split (random_state=42)
    print("\n[3] Splitting Dataset into Train (75%) and Test (25%) Sets (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    print(f"    - Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # 5. Train Random Forest Classifier
    print("\n[4] Training sklearn RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)...")
    model = CongestionModel(n_estimators=50, max_depth=5, random_state=seed)
    model.fit(X_train, y_train)
    print("    - Model training complete.")

    # 6. Evaluate Model on Test Set (Actual Empirical Metrics Only)
    print("\n[5] Evaluating Model on Unseen Test Set (Empirical Metrics Only):")
    eval_results = model.evaluate(X_test, y_test)
    print("-" * 60)
    print(f"  Accuracy                : {eval_results['accuracy']}")
    print(f"  Precision               : {eval_results['precision']}")
    print(f"  Recall                  : {eval_results['recall']}")
    print(f"  F1 Score                : {eval_results['f1_score']}")
    print(f"  ROC-AUC Score           : {eval_results['roc_auc']}")
    print(f"  Confusion Matrix        : [[TN={eval_results['confusion_matrix'][0][0]}, FP={eval_results['confusion_matrix'][0][1]}], [FN={eval_results['confusion_matrix'][1][0]}, TP={eval_results['confusion_matrix'][1][1]}]]")
    print(f"  Test Class Distribution : {eval_results['test_class_distribution']}")
    print("-" * 60)

    # 7. Extract Feature Importances
    print("\n[6] Random Forest Feature Importances:")
    importances = model.get_feature_importances(ML_FEATURE_COLUMNS)
    print("-" * 60)
    for feat, imp in importances.items():
        bar = "#" * int(imp * 40)
        print(f"  {feat:<22} : {imp:.4f} | {bar}")
    print("-" * 60)

    # 8. Demonstrate Live Prediction API Interface
    print("\n[7] Demonstrating Single-Link Prediction API Call:")
    sample_features = {
        "utilization": 0.88,
        "prev_utilization": 0.65,
        "delta_utilization": 0.23,
        "current_load_mbps": 880.0,
        "capacity_mbps": 1000.0,
        "delay_ms": 2.0,
        "drop_rate": 0.0,
        "active_flows_count": 3.0
    }
    prediction = predict_link_congestion(model, sample_features)
    print("-" * 60)
    print(f"  Sample Input Link Features: {sample_features}")
    print(f"  Predicted Congestion Class : {prediction['predicted_congestion_class']} (1 = Congested at t+1)")
    print(f"  Congestion Probability     : {prediction['congestion_probability'] * 100:.1f}%")
    print("-" * 60)

    print("\n[8] Phase 3 ML Future Congestion Prediction Pipeline Complete!\n")
    return eval_results, importances


if __name__ == "__main__":
    run_phase3_demo()
