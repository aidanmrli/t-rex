"""Archival tests for obsolete twist-model entry points."""

import pytest

from trex.models.twist_model import TwistModel, TwistModelConfig


def test_twist_model_config_raises_archival_error():
    """Archived TwistModelConfig should fail fast at construction."""
    with pytest.raises(RuntimeError, match="archived"):
        TwistModelConfig(model_name_or_path="dummy")


def test_twist_model_constructor_raises_archival_error():
    """Archived TwistModel should fail fast at construction."""
    with pytest.raises(RuntimeError, match="archived"):
        TwistModel(model_name_or_path="dummy")
