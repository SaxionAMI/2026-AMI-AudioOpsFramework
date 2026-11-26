"""Audio Ops Framework — Dagster orchestration, task contract and shared core.

The package is import-light: nothing heavy (torch, dagster) is imported here.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("audio-ops-framework")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "1.2.0"

__all__ = ["__version__"]
