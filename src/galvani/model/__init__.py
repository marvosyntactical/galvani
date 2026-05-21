"""Model specification and simulation backends."""

from galvani.model.rate import Result, Stimulus, simulate
from galvani.model.spec import ModelSpec

__all__ = ["ModelSpec", "Result", "Stimulus", "simulate"]
