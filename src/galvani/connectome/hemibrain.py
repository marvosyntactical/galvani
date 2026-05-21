"""hemibrain connectome backend via neuPrint.

Wraps `neuprint-python` and maps its DataFrame conventions onto the typed
`Neuron` / `Subgraph` objects in [base.py]. Designed so the rest of galvani
never sees a neuPrint-specific column name.

Pinned to `hemibrain:v1.2.1` by default; the loader never silently follows
"latest" -- reproducibility is the whole point of the infrastructure layer.

What this backend does:
  - Query neurons by cell type or body id.
  - Fetch the pre x post synapse table for a set of neurons.
  - Cache both as parquet under `~/.cache/galvani/` (see [cache.py]).
  - Stamp NTs onto neurons from a caller-supplied lookup table.

What it does NOT do (intentionally):
  - It does not invent NT predictions. hemibrain:v1.2.1 does not carry
    Eckstein et al. 2024 NTs in the neuPrint graph (the paper postdates the
    dataset). For HD-ring use, pass `nt_by_type=HD_RING_NT` (a published
    lookup; see Turner-Evans 2020). Proper Eckstein-NT integration is a
    v1.5 item.
  - It does not load skeletons in v1. The `Neuron.skeleton_path` field stays
    None; visualization-by-skeleton is a Phase 6 concern.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from galvani.connectome.base import Hemisphere, Neuron, Subgraph
from galvani.connectome.cache import ParquetCache

if TYPE_CHECKING:
    from neuprint import Client

DEFAULT_DATASET = "hemibrain:v1.2.1"
DEFAULT_SERVER = "neuprint.janelia.org"

HD_RING_NT: dict[str, str] = {
    # Published canonical NTs for the HD-ring core types. Sources:
    #   - Turner-Evans et al. (2020), Neuron 108: cholinergic EPG / PEN / PEG;
    #     Delta7 is glutamatergic (inhibitory via GluClalpha in fly).
    #   - Pasch et al. (2024), Hulse et al. (2021) for the EL classification.
    # The neuPrint graph for hemibrain:v1.2.1 does not carry these NTs; we
    # pass them in explicitly via `nt_by_type=HD_RING_NT`.
    "EPG": "acetylcholine",
    "PEN_a(PEN1)": "acetylcholine",
    "PEN_b(PEN2)": "acetylcholine",
    "PEG": "acetylcholine",
    "Delta7": "glutamate",
    "EL": "acetylcholine",
}
"""Canonical NTs for the HD-ring core cell types in hemibrain.

Pass this dict to `HemibrainConnectome(nt_by_type=HD_RING_NT)` to stamp the
right NTs onto queried neurons. Required for the parameterizer's sign
assignment to do anything useful -- without it, every neuron's NT stays None
and all weights collapse to zero.
"""

_HEMI_RX = re.compile(r"_([LR])\d*$")


def _parse_hemisphere(instance: str | None) -> Hemisphere | None:
    """Parse 'L' or 'R' off the trailing token of a hemibrain `instance`
    label, e.g. 'EPG(PB08)_L3' -> 'L', 'Delta7(PB15)_L6R3_L' -> 'L'."""
    if instance is None:
        return None
    match = _HEMI_RX.search(instance)
    if match is None:
        return None
    side = match.group(1)
    return cast(Hemisphere, side)


def _soma_position(loc: Any) -> tuple[float, float, float] | None:
    """neuPrint's `somaLocation` is a 3-list or None. Defensive cast."""
    if loc is None:
        return None
    try:
        x, y, z = loc[0], loc[1], loc[2]
    except (TypeError, IndexError):
        return None
    return float(x), float(y), float(z)


class HemibrainConnectome:
    """Connectome backend backed by Janelia's hemibrain release on neuPrint.

    Cheap to instantiate. Queries are pulled on demand and cached as parquet.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        dataset_version: str = DEFAULT_DATASET,
        server: str = DEFAULT_SERVER,
        client: Client | None = None,
        cache: ParquetCache | None = None,
        nt_by_type: dict[str, str] | None = None,
    ) -> None:
        """Create a hemibrain connectome client.

        Args:
            token: neuPrint auth token. If None, falls back to
                `NEUPRINT_TOKEN` env var. The library does not read
                `NEUPRINT_APPLICATION_CREDENTIALS`; that variable is for
                neuprint-python's own implicit auth path which we sidestep so
                galvani's env contract is one canonical variable.
            dataset_version: pinned dataset (e.g. 'hemibrain:v1.2.1').
            server: neuPrint host.
            client: pre-built `neuprint.Client`; mostly for testing.
            cache: parquet cache; defaults to user_cache_dir/galvani/.
            nt_by_type: maps cell-type string to a lowercased NT name. When
                set, neurons whose type appears here get `nt` stamped on at
                query time. See `HD_RING_NT` for a published HD-ring lookup.
        """
        self.dataset_version = dataset_version
        self._server = server
        self._cache = cache if cache is not None else ParquetCache()
        self._nt_by_type = dict(nt_by_type) if nt_by_type else {}
        self._client = client
        self._token = token if token is not None else os.environ.get("NEUPRINT_TOKEN")

    # -- client lazy init ----------------------------------------------------

    def _ensure_client(self) -> Client:
        if self._client is not None:
            return self._client
        if self._token is None:
            raise RuntimeError(
                "No neuPrint token. Set NEUPRINT_TOKEN or pass token=... to "
                "HemibrainConnectome(...)."
            )
        # Imported lazily so library users without the [neuprint] extra can
        # still import galvani.
        from neuprint import Client as NeuprintClient

        self._client = NeuprintClient(self._server, dataset=self.dataset_version, token=self._token)
        return self._client

    # -- caching helpers -----------------------------------------------------

    def _ds_key(self) -> str:
        # 'hemibrain:v1.2.1' -> 'hemibrain.v1_2_1'. The cache splits keys on
        # '.' to make directories, so the version dots must become underscores.
        return self.dataset_version.replace(".", "_").replace(":", ".").replace("-", "_")

    def _neuron_cache_key(self, type_label: str) -> str:
        # type names may contain '(' and ')'; keep them but disallow '.' and '/'.
        safe = type_label.replace(".", "_").replace("/", "_")
        return f"{self._ds_key()}.neurons.by_type.{safe}"

    def _adj_cache_key(self, body_ids: tuple[int, ...]) -> str:
        # Order-independent key (sorted), hashed for length stability.
        import hashlib

        digest = hashlib.sha1(",".join(str(i) for i in sorted(body_ids)).encode()).hexdigest()[:16]
        return f"{self._ds_key()}.adjacencies.{digest}"

    def _by_id_cache_key(self, body_ids: tuple[int, ...]) -> str:
        # Order-independent key for an ids-only neuron query. Mirrors the
        # adjacency hash scheme; needed so circuits like the mushroom body
        # (where the global inhibitor APL has no `type` field in
        # hemibrain:v1.2.1 and must be fetched by id) can read from the
        # committed fixtures without a live client.
        import hashlib

        digest = hashlib.sha1(",".join(str(i) for i in sorted(body_ids)).encode()).hexdigest()[:16]
        return f"{self._ds_key()}.neurons.by_id.{digest}"

    def _skeleton_cache_key(self, body_id: int) -> str:
        return f"{self._ds_key()}.skeletons.{body_id}"

    # -- queries -------------------------------------------------------------

    def _fetch_neurons_df(self, *, type_label: str) -> pd.DataFrame:
        key = self._neuron_cache_key(type_label)
        if self._cache.has(key):
            return self._cache.load(key)

        client = self._ensure_client()
        # Local import keeps the [neuprint] extra optional at import time.
        from neuprint import NeuronCriteria, fetch_neurons

        df, _roi_df = fetch_neurons(NeuronCriteria(type=type_label), client=client)
        df = cast(pd.DataFrame, df)
        self._cache.store(key, df)
        return df

    def _fetch_adjacencies_df(self, *, body_ids: tuple[int, ...]) -> pd.DataFrame:
        key = self._adj_cache_key(body_ids)
        if self._cache.has(key):
            return self._cache.load(key)

        client = self._ensure_client()
        from neuprint import fetch_adjacencies

        ids = list(body_ids)
        _, conn_df = fetch_adjacencies(sources=ids, targets=ids, client=client)
        # `conn_df` is per-ROI; collapse to one row per (pre, post) by summing
        # the `weight` column. The ROI breakdown is useful for analysis but
        # not for the parameterizer.
        if conn_df is None or conn_df.empty:
            collapsed = pd.DataFrame(
                {
                    "bodyId_pre": pd.Series(dtype=np.int64),
                    "bodyId_post": pd.Series(dtype=np.int64),
                    "weight": pd.Series(dtype=np.int64),
                }
            )
        else:
            collapsed = (
                conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
                .sum()
                .astype({"bodyId_pre": np.int64, "bodyId_post": np.int64, "weight": np.int64})
            )
        self._cache.store(key, collapsed)
        return collapsed

    def query(
        self,
        *,
        type: str | Iterable[str] | None = None,
        ids: Iterable[int] | None = None,
    ) -> tuple[Neuron, ...]:
        """Return neurons matching the query. One of `type` or `ids` is required.

        `type` may be a single string or an iterable of strings (multiple
        cell types). When `ids` is given, results are filtered to that id set
        after the type-based fetch.
        """
        if type is None and ids is None:
            raise ValueError("query() requires one of `type` or `ids`.")

        type_labels: list[str]
        if type is None:
            type_labels = []
        elif isinstance(type, str):
            type_labels = [type]
        else:
            type_labels = list(type)

        if not type_labels:
            # ids-only path. Cached by sorted-id hash so circuits without
            # `type` (e.g. APL in the mushroom body) can still be served
            # from the committed fixtures.
            id_list = list(ids) if ids is not None else []
            key = self._by_id_cache_key(tuple(id_list))
            if self._cache.has(key):
                df = self._cache.load(key)
            else:
                client = self._ensure_client()
                from neuprint import NeuronCriteria, fetch_neurons

                df, _ = fetch_neurons(NeuronCriteria(bodyId=id_list), client=client)
                df = cast(pd.DataFrame, df)
                self._cache.store(key, df)
        else:
            frames = [self._fetch_neurons_df(type_label=t) for t in type_labels]
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if ids is not None:
                wanted = set(int(i) for i in ids)
                df = df[df["bodyId"].astype(np.int64).isin(wanted)]

        return tuple(self._row_to_neuron(row) for _, row in df.iterrows())

    def _row_to_neuron(self, row: pd.Series) -> Neuron:
        cell_type = str(row.get("type", "")) if row.get("type") is not None else ""
        nt = self._nt_by_type.get(cell_type)
        return Neuron(
            id=int(row["bodyId"]),
            cell_type=cell_type,
            hemisphere=_parse_hemisphere(row.get("instance")),
            nt=nt,
            soma_position=_soma_position(row.get("somaLocation")),
            skeleton_path=None,
        )

    def fetch_skeleton(self, body_id: int) -> pd.DataFrame:
        """Return the SWC skeleton for one neuron as a DataFrame.

        Columns follow the neuPrint convention: `rowId, x, y, z, radius,
        link`, where `link` is the parent `rowId` (-1 for the root). All
        coordinates are in nanometers.

        Results are cached per-neuron under the parquet cache so the demo
        and notebook scripts only pay the network cost once.
        """
        key = self._skeleton_cache_key(body_id)
        if self._cache.has(key):
            return self._cache.load(key)

        client = self._ensure_client()
        df = client.fetch_skeleton(body_id, format="pandas")
        df = cast(pd.DataFrame, df)
        self._cache.store(key, df)
        return df

    def subgraph(self, neurons: Iterable[Neuron]) -> Subgraph:
        """Pull the synapse table among `neurons` and return a `Subgraph`."""
        neuron_tuple = tuple(neurons)
        if not neuron_tuple:
            return Subgraph(
                neurons=(),
                pre_ids=np.array([], dtype=np.int64),
                post_ids=np.array([], dtype=np.int64),
                counts=np.array([], dtype=np.int32),
                nt_pre=(),
                dataset_version=self.dataset_version,
            )

        body_ids = tuple(int(n.id) for n in neuron_tuple)
        adj = self._fetch_adjacencies_df(body_ids=body_ids)

        nt_by_id = {n.id: n.nt for n in neuron_tuple}
        # Restrict to the requested set (the live fetch may include synapses
        # to/from neurons outside the query if a cache key collides).
        wanted = set(body_ids)
        adj = adj[adj["bodyId_pre"].isin(wanted) & adj["bodyId_post"].isin(wanted)]

        pre_ids = adj["bodyId_pre"].to_numpy(dtype=np.int64)
        post_ids = adj["bodyId_post"].to_numpy(dtype=np.int64)
        counts = adj["weight"].to_numpy(dtype=np.int32)
        nt_pre = tuple(nt_by_id.get(int(i)) for i in pre_ids)

        return Subgraph(
            neurons=neuron_tuple,
            pre_ids=pre_ids,
            post_ids=post_ids,
            counts=counts,
            nt_pre=nt_pre,
            dataset_version=self.dataset_version,
        )


__all__ = ["HD_RING_NT", "HemibrainConnectome"]
