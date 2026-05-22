"""Model specification and simulation backends."""

from galvani.model.adex import AdExResult, simulate_adex
from galvani.model.hh import HHResult, simulate_hh
from galvani.model.lif import LIFResult, simulate_lif
from galvani.model.rate import Result, Stimulus, simulate
from galvani.model.spec import ModelSpec

__all__ = [
    "AdExResult",
    "HHResult",
    "LIFResult",
    "ModelSpec",
    "Result",
    "Stimulus",
    "simulate",
    "simulate_adex",
    "simulate_hh",
    "simulate_lif",
]
