"""
Singleton service manager providing access to underlying simulation engine, telemetry, and pre-trained ML model instances.
"""

from typing import Optional
from src.simulation.engine import NetworkXEngine
from src.simulation.traffic import DeterministicTrafficGenerator
from src.telemetry.collector import TelemetryCollector
from src.telemetry.features import FeatureExtractor
from src.ml.dataset import SupervisedDatasetBuilder
from src.ml.model import CongestionModel
from src.routing.evaluator import RoutingEvaluator


class APIServiceManager:
    """
    Singleton service manager wrapping project components for FastAPI routes.
    """

    _instance: Optional["APIServiceManager"] = None

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.engine = NetworkXEngine(seed=self.seed)
        self.collector = TelemetryCollector()
        self.extractor = FeatureExtractor()
        self.dataset_builder = SupervisedDatasetBuilder()
        self.traffic_gen = DeterministicTrafficGenerator(seed=self.seed)

        # Warmup engine with flows
        flows = self.traffic_gen.generate_fixed_flow_set()
        for f in flows:
            self.engine.add_flow(f)
        for _ in range(5):
            snapshot = self.engine.step()
            self.collector.collect(snapshot)

        # Train ML Model on startup
        evaluator = RoutingEvaluator(seed=self.seed, alpha_penalty=10.0)
        self.ml_model = evaluator.train_predictive_model(num_warmup_ticks=25)

    @classmethod
    def get_instance(cls, seed: int = 42) -> "APIServiceManager":
        """Returns global singleton instance."""
        if cls._instance is None:
            cls._instance = APIServiceManager(seed=seed)
        return cls._instance


def get_service_manager() -> APIServiceManager:
    """Dependency helper function for route handlers."""
    return APIServiceManager.get_instance()
