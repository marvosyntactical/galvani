"""Galvani: connectome subgraph to executable rate-model pipeline."""

from importlib.metadata import PackageNotFoundError, version

from galvani.connectome.base import Neuron, Subgraph
from galvani.connectome.hemibrain import HD_RING_NT, HemibrainConnectome
from galvani.model.rate import Result, simulate
from galvani.model.spec import ModelSpec
from galvani.parameterize import ParameterizerOptions, default_parameterizer

try:
    __version__ = version("galvani")
except PackageNotFoundError:  # pragma: no cover - editable, pre-install
    __version__ = "0.0.0+local"

__all__ = [
    "HD_RING_NT",
    "HemibrainConnectome",
    "ModelSpec",
    "Neuron",
    "ParameterizerOptions",
    "Result",
    "Subgraph",
    "__version__",
    "default_parameterizer",
    "simulate",
]
