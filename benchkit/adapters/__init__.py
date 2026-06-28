"""Adapter registry."""
from __future__ import annotations

from ..config import RunEntry
from .base import Adapter
from .cusz_ref import CuszAdapter
from .fzgm import FzgmAdapter

_BUILDERS = {
    "fzgm": lambda entry: FzgmAdapter(variant=entry.variant, cli_path=entry.cli_path),
    "cusz": lambda entry: CuszAdapter(variant=entry.variant, cli_path=entry.cli_path),
}


def build_adapter(entry: RunEntry) -> Adapter:
    try:
        return _BUILDERS[entry.compressor](entry)
    except KeyError:
        raise ValueError(
            f"no adapter for compressor '{entry.compressor}' "
            f"(have: {sorted(_BUILDERS)})")
