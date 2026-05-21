"""Re-pull connectome queries from neuPrint and write them under tests/fixtures/.

Run after rotating your neuPrint token or after a hemibrain dataset update:

    NEUPRINT_TOKEN=... uv run python scripts/refresh_fixtures.py

The committed fixtures back the loader's unit tests so CI never needs a token.
"""

from __future__ import annotations

from pathlib import Path

from galvani.circuits.mushroom_body import APL_BODY_ID, MB_KC_TYPES, MB_NT
from galvani.connectome.cache import ParquetCache
from galvani.connectome.hemibrain import (
    DEFAULT_DATASET,
    HD_RING_NT,
    HemibrainConnectome,
)

HD_RING_TYPES = ("EPG", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7")


def refresh_hd_ring(cache: ParquetCache) -> None:
    """HD-ring fixtures (Phase 5)."""
    conn = HemibrainConnectome(cache=cache, nt_by_type=HD_RING_NT)
    neurons = conn.query(type=list(HD_RING_TYPES))
    sg = conn.subgraph(neurons)
    print(
        f"  HD ring:       neurons={len(neurons):4d}  syn_rows={len(sg.counts):5d}  total_syn={int(sg.counts.sum()):7d}"
    )


def refresh_mushroom_body(cache: ParquetCache) -> None:
    """Mushroom body fixtures (Phase 7)."""
    conn = HemibrainConnectome(cache=cache, nt_by_type=MB_NT)
    kcs = conn.query(type=list(MB_KC_TYPES))
    apl = conn.query(ids=[APL_BODY_ID])
    neurons = tuple(kcs) + tuple(apl)
    sg = conn.subgraph(neurons)
    print(
        f"  Mushroom body: neurons={len(neurons):4d}  syn_rows={len(sg.counts):5d}  total_syn={int(sg.counts.sum()):7d}"
    )


def main() -> None:
    fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    cache = ParquetCache(fixtures)
    print(f"Refreshing fixtures for {DEFAULT_DATASET} into {fixtures}")
    refresh_hd_ring(cache)
    refresh_mushroom_body(cache)

    total = sum(p.stat().st_size for p in fixtures.rglob("*.parquet"))
    print(
        f"\nTotal fixtures on disk: {total / 1024 / 1024:.2f} MiB across "
        f"{len(list(fixtures.rglob('*.parquet')))} parquet files"
    )


if __name__ == "__main__":
    main()
