"""Shared pytest configuration and fixtures for Liquid Vision tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_image():
    """Generate a sample test image."""
    return np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)


@pytest.fixture(scope="session") 
def sample_model_config():
    """Sample LNN model configuration."""
    return {
        "input_resolution": (160, 120),
        "ode_solver": "FIXED_POINT_RK4",
        "timestep_adaptive": True,
        "max_inference_time_ms": 20.0,
        "memory_limit_kb": 256
    }


@pytest.fixture
def performance_thresholds():
    """Performance test thresholds."""
    return {
        "max_inference_time_ms": 25.0,
        "max_memory_usage_kb": 512,
        "min_accuracy_percent": 85.0
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmarks")
    config.addinivalue_line("markers", "hardware: Hardware-in-loop tests")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on location."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)