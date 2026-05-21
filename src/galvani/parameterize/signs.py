"""Neurotransmitter -> synaptic sign mapping for *Drosophila*.

The convention in fly is different from mammalian: glutamate is largely
inhibitory via GluClalpha receptors. Acetylcholine is the dominant excitatory
NT. Monoamines (octopamine, serotonin, dopamine) are modulatory and treated as
0 in the recurrent dynamics for v1.

References:
  - Liu & Wilson (2013) for GluClalpha-mediated glutamatergic inhibition in
    fly.
  - Eckstein et al. (2024) for the predicted NT labels in hemibrain
    (~85% accuracy).

This is exposed as a callable so users can override it cleanly per circuit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

Sign = int  # -1, 0, or +1

FLY_NT_SIGN: Final[dict[str, Sign]] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "octopamine": 0,
    "serotonin": 0,
    "dopamine": 0,
}
"""Default fly NT → sign. Unknown NTs default to 0 (modulatory / ignored)."""


def fly_default(nt: str | None) -> Sign:
    """Map a predicted neurotransmitter to a synaptic sign (-1, 0, +1).

    Unknown or missing NTs return 0 (no contribution to recurrent dynamics).
    Users override by passing their own `nt_to_sign` to `default_parameterizer`.
    """
    if nt is None:
        return 0
    return FLY_NT_SIGN.get(nt.lower(), 0)


NTToSign = Callable[[str | None], Sign]
