"""Unit tests for TSMC baseline config loading."""

import argparse
import sys
from types import ModuleType

import pytest

sys.modules.setdefault("jsonlines", ModuleType("jsonlines"))

from trex.baselines.tsmc_baseline import load_config


def _base_args() -> argparse.Namespace:
    return argparse.Namespace(
        config=None,
        output_dir=None,
        dataset_path=None,
        generator_model_path=None,
        value_model_path=None,
        value_head_path=None,
        value_head_type=None,
        twist_space=None,
        twist_mode=None,
        n_particles=None,
        max_smc_iterations=None,
        resampling_unit=None,
        resample_every_tokens=None,
        temperature=None,
    )


def test_load_config_revalidates_cli_overrides():
    """Invalid CLI overrides should fail validation."""
    args = _base_args()
    args.twist_space = "invalid"

    with pytest.raises(ValueError, match="twist_space"):
        load_config(args)
