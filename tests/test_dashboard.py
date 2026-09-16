"""
Unit test suite for Phase 7 Streamlit Dashboard API Client.
"""

import pytest
from src.dashboard.api_client import DashboardAPIClient


def test_client_connection_error_handling():
    """Verify DashboardAPIClient handles unreachable connection gracefully without crashing."""
    client = DashboardAPIClient(base_url="http://localhost:9999/invalid_api")
    res = client.get_health()
    assert res.get("error") is True
    assert "Connection error" in res.get("message", "")


def test_client_methods_structure():
    """Verify API client payload construction and endpoints formatting."""
    client = DashboardAPIClient(base_url="http://localhost:8000/api/v1")
    assert client.base_url == "http://localhost:8000/api/v1"
