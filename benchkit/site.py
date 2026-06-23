"""Site configuration — the per-machine paths that must NOT be hardcoded.

Resolves the few environment-specific values (the fzgmod-cli binary, the results root)
so the same configs run unchanged on the local desktop and on an HPC cluster. Precedence:
explicit CLI arg > environment variable > configs/site.local.yaml (gitignored) > default.

Dataset paths are resolved separately in datasets.py via ${ENV} expansion in the manifest,
so a portable manifest can point `root: ${BENCHKIT_DATA_ROOT}/...` per site.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_SITE_LOCAL = REPO_ROOT / "configs" / "site.local.yaml"


@dataclass
class Site:
    fzgmod_cli: str | None
    results_root: Path

    @classmethod
    def load(cls, results_root: str | None = None) -> "Site":
        local: dict = {}
        if _SITE_LOCAL.exists():
            local = yaml.safe_load(_SITE_LOCAL.read_text()) or {}

        cli = (os.environ.get("FZGMOD_CLI")
               or local.get("fzgmod_cli"))

        root = (results_root
                or os.environ.get("BENCHKIT_RESULTS_ROOT")
                or local.get("results_root")
                or str(REPO_ROOT / "results"))

        return cls(fzgmod_cli=cli, results_root=Path(root).expanduser())

    def export_env(self) -> None:
        """Publish resolved values to the environment so adapters pick them up."""
        if self.fzgmod_cli:
            os.environ.setdefault("FZGMOD_CLI", self.fzgmod_cli)
