"""Archival tests for obsolete TSMC baseline/value-head entry points."""

import argparse
import sys
from types import ModuleType

import pytest

sys.modules.setdefault("jsonlines", ModuleType("jsonlines"))

import trex.models as models
from trex.baselines.tsmc_baseline import TSMCBaseline, load_config
from trex.baselines.tsmc_config import TSMCConfig


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(config=None)


def test_tsmc_config_instantiation_raises_archival_error():
    """TSMC config is archived and should not be instantiated."""
    with pytest.raises(RuntimeError, match="archived"):
        TSMCConfig()


def test_load_config_raises_archival_error():
    """TSMC baseline config loader should fail fast for archived baseline."""
    with pytest.raises(RuntimeError, match="archived"):
        load_config(_base_args())


def test_tsmc_baseline_construction_raises_archival_error():
    """TSMC baseline runner is archived and should not construct."""
    with pytest.raises(RuntimeError, match="archived"):
        TSMCBaseline(config=None)


def test_models_package_no_longer_exports_twist_or_value_head_symbols():
    """`trex.models` should stop exporting obsolete twist/value-head symbols."""
    assert not hasattr(models, "TwistModel")
    assert not hasattr(models, "ValueHead")
    assert not hasattr(models, "LinearValueHead")
    assert not hasattr(models, "MLPValueHead")
    assert not hasattr(models, "AttentionPooledValueHead")
