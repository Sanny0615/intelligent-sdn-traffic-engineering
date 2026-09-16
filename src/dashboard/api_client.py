"""
API Client Helper for Streamlit Dashboard.
Handles HTTP communication with FastAPI backend endpoints with graceful exception handling.
"""

from typing import Dict, Any
import httpx


class DashboardAPIClient:
    """
    HTTP API Client communicating with FastAPI backend service.
    """

    def __init__(self, base_url: str = "http://localhost:8000/api/v1", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception as err:
            return {"error": True, "message": f"Connection error to '{url}': {str(err)}"}

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as err:
            try:
                detail = err.response.json().get("detail", str(err))
            except Exception:
                detail = str(err)
            return {"error": True, "message": f"HTTP {err.response.status_code}: {detail}"}
        except Exception as err:
            return {"error": True, "message": f"Connection error to '{url}': {str(err)}"}

    def get_health(self) -> Dict[str, Any]:
        """Fetch health status."""
        return self._get("/health")

    def get_topology(self) -> Dict[str, Any]:
        """Fetch network topology."""
        return self._get("/topology")

    def get_telemetry(self) -> Dict[str, Any]:
        """Fetch current telemetry snapshot."""
        return self._get("/telemetry")

    def predict_congestion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Query ML congestion prediction endpoint."""
        return self._post("/predict/congestion", payload)

    def optimize_routing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Query TE routing optimization endpoint."""
        return self._post("/routing/optimize", payload)

    def execute_what_if(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Query Digital Twin What-If scenario endpoint."""
        return self._post("/simulation/what-if", payload)

    def explain_ai(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Query AI RAG explanation endpoint."""
        return self._post("/ai/explain", payload)

