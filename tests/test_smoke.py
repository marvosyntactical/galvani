"""Phase 0 smoke test: package imports and version is present."""

import galvani


def test_version_attribute_exists() -> None:
    assert isinstance(galvani.__version__, str)
    assert len(galvani.__version__) > 0
