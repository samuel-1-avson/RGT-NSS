"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_text():
    return "The quick brown fox jumps over the lazy dog. The dog barked loudly."


@pytest.fixture
def sample_corpus():
    return (
        "The quick brown fox jumps over the lazy dog. "
        "The lazy dog sleeps all day long. "
        "A fox is a quick animal that hunts prey. "
        "Dogs and foxes are both mammals that live in many habitats."
    )


@pytest.fixture
def training_data():
    """Small training data array for trainer tests."""
    np.random.seed(42)
    return np.random.randint(0, 256, size=(100, 16))
