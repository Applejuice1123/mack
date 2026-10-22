import pytest

from mack import system


@pytest.fixture(autouse=True)
def quiet_mac(monkeypatch):
    """No test may depend on this laptop's real battery, heat, memory or idle time:
    at 20% battery every runner (correctly) pauses. Tests that need a condition set it."""
    monkeypatch.setattr(system, "power", lambda: system.Power(False, 100))
    monkeypatch.setattr(system, "thermal_state", lambda: system.THERMAL_NOMINAL)
    monkeypatch.setattr(system, "input_idle_seconds", lambda: 0.0)
    monkeypatch.setattr(system, "memory_pressure", lambda: system.MEMORY_NORMAL)
    monkeypatch.setattr(system, "available_memory_gb", lambda: 16.0)
