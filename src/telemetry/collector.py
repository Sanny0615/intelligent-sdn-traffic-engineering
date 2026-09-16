"""
Telemetry collector responsible for buffering time-series network snapshots across simulation ticks.
"""

from typing import List, Optional
from src.simulation.models import TelemetrySnapshot


class TelemetryCollector:
    """
    Buffers historical network telemetry snapshots across simulation ticks.
    """

    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.history: List[TelemetrySnapshot] = []

    def collect(self, snapshot: TelemetrySnapshot) -> None:
        """Appends a new simulation tick snapshot to historical buffer."""
        self.history.append(snapshot)
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

    def get_history(self) -> List[TelemetrySnapshot]:
        """Returns the list of all collected snapshots in chronological order."""
        return list(self.history)

    def get_latest_snapshot(self) -> Optional[TelemetrySnapshot]:
        """Returns the most recent snapshot, or None if history is empty."""
        if not self.history:
            return None
        return self.history[-1]

    def reset(self) -> None:
        """Clears all historical snapshots."""
        self.history.clear()
