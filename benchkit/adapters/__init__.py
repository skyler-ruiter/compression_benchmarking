"""Adapter registry."""
from __future__ import annotations

from ..config import RunEntry
from .base import Adapter
from .cusz_ref import CuszAdapter
from .cuszhi import CuszhiAdapter
from .cuszp import CuszpAdapter
from .fzgm import FzgmAdapter
from .fzgpu import FzgpuAdapter
from .mans import MansAdapter
from .pfpl import PfplAdapter

_BUILDERS = {
    "fzgm":   lambda entry: FzgmAdapter(variant=entry.variant, cli_path=entry.cli_path),
    "cusz":   lambda entry: CuszAdapter(variant=entry.variant, cli_path=entry.cli_path),
    "cuszhi": lambda entry: CuszhiAdapter(variant=entry.variant, cli_path=entry.cli_path),
    "cuszp2": lambda entry: CuszpAdapter(version=2, variant=entry.variant, cli_path=entry.cli_path),
    "cuszp3": lambda entry: CuszpAdapter(version=3, variant=entry.variant, cli_path=entry.cli_path),
    "pfpl":   lambda entry: PfplAdapter(variant=entry.variant, cli_path=entry.cli_path),
    "fzgpu":  lambda entry: FzgpuAdapter(variant=entry.variant, cli_path=entry.cli_path),
    "mans":   lambda entry: MansAdapter(variant=entry.variant, cli_path=entry.cli_path),
}


def build_adapter(entry: RunEntry) -> Adapter:
    try:
        return _BUILDERS[entry.compressor](entry)
    except KeyError:
        raise ValueError(
            f"no adapter for compressor '{entry.compressor}' "
            f"(have: {sorted(_BUILDERS)})")
