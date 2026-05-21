"""Model specification and simulation backends."""

from galvani.model.lif import LIFResult, simulate_lif
from galvani.model.rate import Result, Stimulus, simulate
from galvani.model.spec import ModelSpec

__all__ = ["LIFResult", "ModelSpec", "Result", "Stimulus", "simulate", "simulate_lif"]
