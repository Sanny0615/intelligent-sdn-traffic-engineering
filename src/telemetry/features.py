"""
Feature extraction module converting raw network telemetry snapshots into structured ML-ready feature datasets.
"""

from typing import List, Dict, Any
import pandas as pd

from src.config import CONGESTION_THRESHOLD_UTILIZATION
from src.simulation.models import TelemetrySnapshot


class FeatureExtractor:
    """
    Extracts structured time-series feature rows and Pandas DataFrames from network telemetry history.
    """

    FEATURE_COLUMNS: List[str] = [
        "tick",
        "link_id",
        "source",
        "target",
        "capacity_mbps",
        "delay_ms",
        "current_load_mbps",
        "utilization",
        "prev_utilization",
        "delta_utilization",
        "total_bytes",
        "total_packets",
        "dropped_packets",
        "drop_rate",
        "active_flows_count",
        "is_congested"
    ]

    def __init__(self):
        pass

    def extract_features(self, history: List[TelemetrySnapshot]) -> List[Dict[str, Any]]:
        """
        Extracts feature dictionaries from a list of chronological TelemetrySnapshots.
        Computes time-series lag features (prev_utilization, delta_utilization) across consecutive ticks.
        """
        feature_rows: List[Dict[str, Any]] = []
        prev_utilization_map: Dict[str, float] = {}

        for snapshot in history:
            tick = snapshot.timestamp_tick
            for link_key, link_data in snapshot.links_telemetry.items():
                current_util = float(link_data["utilization"])
                prev_util = prev_utilization_map.get(link_key, 0.0)
                delta_util = round(current_util - prev_util, 4)

                total_pkts = float(link_data["total_packets"])
                dropped_pkts = float(link_data["dropped_packets"])
                drop_rate = round(dropped_pkts / total_pkts, 4) if total_pkts > 0 else 0.0

                is_congested = 1 if current_util >= CONGESTION_THRESHOLD_UTILIZATION else 0

                row = {
                    "tick": tick,
                    "link_id": link_key,
                    "source": link_data["source"],
                    "target": link_data["target"],
                    "capacity_mbps": float(link_data["capacity_mbps"]),
                    "delay_ms": float(link_data.get("delay_ms", 2.0)),
                    "current_load_mbps": float(link_data["current_load_mbps"]),
                    "utilization": current_util,
                    "prev_utilization": round(prev_util, 4),
                    "delta_utilization": delta_util,
                    "total_bytes": float(link_data["total_bytes"]),
                    "total_packets": total_pkts,
                    "dropped_packets": dropped_pkts,
                    "drop_rate": drop_rate,
                    "active_flows_count": float(link_data["active_flows_count"]),
                    "is_congested": is_congested
                }
                feature_rows.append(row)

                # Update previous utilization for the next tick
                prev_utilization_map[link_key] = current_util

        return feature_rows

    def to_dataframe(self, history: List[TelemetrySnapshot]) -> pd.DataFrame:
        """
        Converts telemetry history into a structured Pandas DataFrame.
        """
        rows = self.extract_features(history)
        if not rows:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)
        df = pd.DataFrame(rows)
        return df[self.FEATURE_COLUMNS]
