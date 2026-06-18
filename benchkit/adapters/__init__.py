"""Adapter registry. Add reference compressors here in M3."""
from __future__ import annotations

from ..config import RunEntry
from .base import Adapter
from .fzgm import FzgmAdapter

_BUILDERS = {
    "fzgm": lambda entry: FzgmAdapter(variant=entry.variant, cli_path=entry.cli_path),
}


def build_adapter(entry: RunEntry) -> Adapter:
    try:
        return _BUILDERS[entry.compressor](entry)
    except KeyError:
        raise ValueError(
            f"no adapter for compressor '{entry.compressor}' "
            f"(have: {sorted(_BUILDERS)}). Reference adapters land in M3.")
