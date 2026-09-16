"""
Supervised dataset builder module.
Converts Phase 2 telemetry feature DataFrames into supervised (X, y) datasets for predicting t+1 link congestion.
"""

from typing import Tuple, List
import pandas as pd


ML_FEATURE_COLUMNS: List[str] = [
    "utilization",
    "prev_utilization",
    "delta_utilization",
    "current_load_mbps",
    "capacity_mbps",
    "delay_ms",
    "drop_rate",
    "active_flows_count",
]

TARGET_COLUMN: str = "future_congestion"


class SupervisedDatasetBuilder:
    """
    Constructs supervised (X, y) datasets from time-series telemetry DataFrames.
    Features come from tick t, and target comes from tick t+1 (future_congestion).
    """

    def __init__(self, feature_columns: List[str] = None):
        if feature_columns is None:
            self.feature_columns = list(ML_FEATURE_COLUMNS)
        else:
            self.feature_columns = list(feature_columns)

    def create_supervised_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Processes telemetry DataFrame:
        1. Groups by link_id and shifts 'is_congested' by -1 to create 'future_congestion' (tick t+1 target).
        2. Drops final tick per link where t+1 target is unknown (NaN).
        3. Extracts feature matrix X (tick t features) and target vector y (tick t+1 congestion).
        
        Prevents data leakage:
        - Identifiers (tick, link_id, source, target) are excluded from X.
        - Current/future target column is excluded from X.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot build supervised dataset.")

        required_cols = set(self.feature_columns + ["tick", "link_id", "is_congested"])
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

        # Sort chronologically per link
        df_sorted = df.sort_values(["link_id", "tick"]).copy()

        # Shift is_congested by -1 per link to form t+1 target
        df_sorted[TARGET_COLUMN] = df_sorted.groupby("link_id")["is_congested"].shift(-1)

        # Drop last tick for each link where future target is NaN
        df_clean = df_sorted.dropna(subset=[TARGET_COLUMN]).copy()

        # Extract features X and target y
        X = df_clean[self.feature_columns].copy()
        y = df_clean[TARGET_COLUMN].astype(int).copy()

        return X, y
