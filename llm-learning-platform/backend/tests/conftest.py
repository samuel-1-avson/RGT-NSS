"""
Pytest configuration and fixtures.
"""

import pytest
import asyncio
import numpy as np
from fastapi.testclient import TestClient
from typing import Generator, AsyncGenerator
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.core.state_manager import MemoryStateManager, set_state_manager
from app.models.gpt import GPTConfig, MicroGPT
from app.models.pytorch_gpt import PyTorchGPTConfig, PyTorchGPT


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> Generator:
    """Create a test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
async def state_manager():
    """Create a fresh memory state manager."""
    manager = MemoryStateManager()
    set_state_manager(manager)
    yield manager
    # Cleanup
    set_state_manager(None)


@pytest.fixture
def gpt_config() -> GPTConfig:
    """Create a small GPT config for testing."""
    return GPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        dropout=0.0  # Disable dropout for deterministic tests
    )


@pytest.fixture
def pytorch_gpt_config() -> PyTorchGPTConfig:
    """Create a small PyTorch GPT config for testing."""
    return PyTorchGPTConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        dropout=0.0
    )


@pytest.fixture
def microgpt(gpt_config) -> MicroGPT:
    """Create a MicroGPT model for testing."""
    return MicroGPT(gpt_config)


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return "Hello, world! This is a test sentence."


@pytest.fixture
def sample_tokens() -> list:
    """Sample token IDs for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def auth_headers() -> dict:
    """Authentication headers for testing."""
    return {
        "Authorization": "Bearer llm_dev_admin_key_change_in_production",
        "X-API-Key": "llm_dev_admin_key_change_in_production"
    }


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    # Reset state manager
    set_state_manager(MemoryStateManager())
    yield
    # Cleanup after test
    set_state_manager(None)
