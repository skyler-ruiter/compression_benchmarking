"""Load and validate experiment + dataset configs.

Stdlib dataclasses (no pydantic dependency) with light, explicit validation so a
malformed config fails with a clear message rather than a deep traceback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---- dataset manifest -------------------------------------------------------


@dataclass
class FieldSpec:
    dataset: str
    field: str
    dtype: str            # "f32" | "f64"
    dim_order: str        # "fast-to-slow"
    dims: list[int]
    path: Path            # absolute path to the raw binary

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n

    @property
    def element_size(self) -> int:
        return 8 if self.dtype in ("f64", "i64") else 4

    @property
    def original_bytes(self) -> int:
        return self.num_elements * self.element_size

    @property
    def dim_arg(self) -> str:
        """FZGM -l argument: fast x mid x slow."""
        return "x".join(str(d) for d in self.dims)


class DatasetCatalog:
    """Resolves (dataset, field) -> FieldSpec from configs/datasets.yaml."""

    def __init__(self, datasets: dict[str, Any]):
        self._raw = datasets

    @classmethod
    def load(cls, path: str | Path) -> "DatasetCatalog":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: dataset manifest must be a mapping of dataset->spec")
        return cls(raw)

    def fields(self, dataset: str) -> list[str]:
        self._require(dataset)
        return list(self._raw[dataset].get("fields", {}).keys())

    def resolve(self, dataset: str, field_name: str) -> FieldSpec:
        self._require(dataset)
        ds = self._raw[dataset]
        fields = ds.get("fields", {})
        if field_name not in fields:
            raise KeyError(f"field '{field_name}' not found in dataset '{dataset}' "
                           f"(have: {sorted(fields)})")
        fspec = fields[field_name]
        root = Path(ds.get("root", "")).expanduser()
        rel = Path(fspec["path"])
        path = rel if rel.is_absolute() else (root / rel)
        dims = list(fspec["dims"])
        if not dims or any(int(d) <= 0 for d in dims):
            raise ValueError(f"{dataset}/{field_name}: dims must be positive ints, got {dims}")
        spec = FieldSpec(
            dataset=dataset,
            field=field_name,
            dtype=ds.get("dtype", "f32"),
            dim_order=ds.get("dim_order", "fast-to-slow"),
            dims=[int(d) for d in dims],
            path=path.resolve(),
        )
        if not spec.path.exists():
            raise FileNotFoundError(f"{dataset}/{field_name}: data file not found: {spec.path}")
        actual = spec.path.stat().st_size
        if actual < spec.original_bytes:
            raise ValueError(
                f"{dataset}/{field_name}: file {spec.path} is {actual} bytes but dims "
                f"{dims} x {spec.element_size}B = {spec.original_bytes} expected")
        return spec

    def _require(self, dataset: str) -> None:
        if dataset not in self._raw:
            raise KeyError(f"dataset '{dataset}' not in manifest (have: {sorted(self._raw)})")


# ---- experiment config ------------------------------------------------------


# Canonical, tool-agnostic error modes. Adapters translate these into each tool's
# native flag/mode and declare the eb basis (DESIGN.md §5.4). `rel_range` is the
# cross-tool comparable: every EBLC supports it (as "REL" for ABS/REL tools, "NOA" for
# ABS/NOA/REL tools like FZGM/PFPL). `from_toml` means "use the bound the pipeline
# already declares" — for shipping a hand-tuned config as-is (no sweep).
CANONICAL_MODES = {"abs", "rel_range", "rel_maxabs", "from_toml"}


@dataclass
class RunEntry:
    compressor: str
    variant: str          # "reference" | "fzgm"
    pipeline: str         # path to a .toml pipeline, or a --stages chain for quick tests
    cli_path: str | None = None   # optional per-entry binary override

    @property
    def is_toml(self) -> bool:
        return self.pipeline.strip().endswith(".toml")


@dataclass
class ExperimentConfig:
    name: str
    datasets: list[str]
    fields: Any                    # "all" | {dataset: [field, ...]}
    error_mode: str
    error_bounds: list[float]
    repetitions: int
    warmup_reps: int
    lock_clocks: bool
    retain_decompressed: bool
    runs: list[RunEntry]
    pairings: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        try:
            err = raw["error"]
            runs = [RunEntry(compressor=r["compressor"],
                             variant=r.get("variant", r["compressor"]),
                             pipeline=r.get("pipeline", "default"),
                             cli_path=r.get("cli_path"))
                    for r in raw["runs"]]
        except KeyError as e:
            raise ValueError(f"{path}: missing required key {e}") from e
        mode = err.get("mode", "rel_range")
        if mode not in CANONICAL_MODES:
            raise ValueError(f"{path}: error.mode '{mode}' not in {sorted(CANONICAL_MODES)}")
        bounds = [None] if mode == "from_toml" else [float(b) for b in err["bounds"]]
        cfg = cls(
            name=raw.get("name", Path(path).stem),
            datasets=list(raw["datasets"]),
            fields=raw.get("fields", "all"),
            error_mode=mode,
            error_bounds=bounds,
            repetitions=int(raw.get("repetitions", 5)),
            warmup_reps=int(raw.get("warmup_reps", 3)),
            lock_clocks=bool(raw.get("lock_clocks", False)),
            # The decompressed output is ~original-sized; off by default to respect the
            # local disk budget. Its checksum is recorded regardless, and c.fzm is kept
            # so it can be regenerated. Toggle on for experiments that need the array.
            retain_decompressed=bool(raw.get("retain_decompressed", False)),
            runs=runs,
            pairings=list(raw.get("pairings", [])),
            raw=raw,
        )
        if not cfg.datasets:
            raise ValueError(f"{path}: 'datasets' is empty")
        if not cfg.runs:
            raise ValueError(f"{path}: 'runs' is empty")
        return cfg

    def fields_for(self, dataset: str, catalog: DatasetCatalog) -> list[str]:
        if self.fields == "all" or self.fields is None:
            return catalog.fields(dataset)
        if isinstance(self.fields, dict):
            return list(self.fields.get(dataset, catalog.fields(dataset)))
        if isinstance(self.fields, list):
            return list(self.fields)
        raise ValueError(f"unsupported 'fields' value: {self.fields!r}")
