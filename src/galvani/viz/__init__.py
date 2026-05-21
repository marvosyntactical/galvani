"""Visualization payload exporters.

The pattern: run a simulation in Python, then `write_payload(...)` produces a
static JSON file that the web demo (`examples/web_demo/`) loads on demand.
"""

from galvani.viz.payload import build_hd_ring_payload, build_payload, write_payload

__all__ = ["build_hd_ring_payload", "build_payload", "write_payload"]
