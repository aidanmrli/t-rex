"""Archival tests for obsolete twisted-SMC entry points."""

import pytest

import trex.smc as smc
from trex.smc import twisted_smc


def test_twisted_symbols_not_exported_from_smc_package():
    """`trex.smc` should not expose twisted-SMC symbols after the March 2026 pivot."""
    assert not hasattr(smc, "compute_twisted_weights")
    assert not hasattr(smc, "TwistedSMCConfig")
    assert not hasattr(smc, "TwistedSMC")


def test_twisted_module_functions_raise_archival_error():
    """Direct use of archived twisted-SMC helpers should fail fast."""
    with pytest.raises(RuntimeError, match="archived"):
        twisted_smc.compute_twisted_weights(None, None)


def test_twisted_module_classes_raise_archival_error():
    """Direct construction of archived twisted-SMC classes should fail fast."""
    with pytest.raises(RuntimeError, match="archived"):
        twisted_smc.TwistedSMCConfig()

    with pytest.raises(RuntimeError, match="archived"):
        twisted_smc.TwistedSMC(config=None)
