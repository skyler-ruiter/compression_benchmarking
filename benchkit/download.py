"""SDRBench dataset downloader.

Public API:
    download_dataset(key, data_dir, tarball_dir)  -- download + extract one dataset
    ALL_DATASETS                                   -- list of all known Dataset objects
    DATASET_KEYS                                   -- sorted list of valid key strings
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


BASE_URL = (
    "https://g-8d6b0.fd635.8443.data.globus.org"
    "/ds131.2/Data-Reduction-Repo/raw-data"
)


@dataclass(frozen=True)
class Dataset:
    key: str
    tarball: str
    url_suffix: str
    dest_dir: str
    name: str
    dims: str
    dtype: str
    num_fields: int
    description: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.url_suffix}"


ALL_DATASETS: list[Dataset] = [
    Dataset(
        key="CESM",
        tarball="SDRBENCH-CESM-ATM-1800x3600.tar.gz",
        url_suffix="CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz",
        dest_dir="CESM_1800x3600",
        name="CESM-ATM 2D", dims="1800x3600", dtype="f32", num_fields=79,
        description="CESM atmosphere model, 79 2-D surface fields",
    ),
    Dataset(
        key="CESMATM",
        tarball="SDRBENCH-CESM-ATM-26x1800x3600.tar.gz",
        url_suffix="CESM-ATM/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz",
        dest_dir="CESMATM_26x1800x3600",
        name="CESM-ATM 3D", dims="26x1800x3600", dtype="f32", num_fields=33,
        description="CESM atmosphere model, 33 3-D fields (26 pressure levels)",
    ),
    Dataset(
        key="EXAALT",
        tarball="SDRBENCH-EXAALT-2869440.tar.gz",
        url_suffix="EXAALT/SDRBENCH-EXAALT-2869440.tar.gz",
        dest_dir="EXAALT_2869440",
        name="EXAALT", dims="2869440", dtype="f32", num_fields=6,
        description="Molecular dynamics particle positions/velocities (vx/vy/vz/xx/yy/zz)",
    ),
    Dataset(
        key="HURR",
        tarball="SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz",
        url_suffix="Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz",
        dest_dir="HURR_100x500x500",
        name="Hurricane Isabel", dims="100x500x500", dtype="f32", num_fields=13,
        description="Hurricane Isabel simulation (NCAR WRF), 13 atmospheric fields",
    ),
    Dataset(
        key="HACC",
        tarball="EXASKY-HACC-data-medium-size.tar.gz",
        url_suffix="EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz",
        dest_dir="HACCM_280953867",
        name="HACC medium", dims="280953867", dtype="f32", num_fields=6,
        description="N-body cosmology (HACC), 280M particles, positions/velocities",
    ),
    Dataset(
        key="NYX",
        tarball="SDRBENCH-EXASKY-NYX-512x512x512.tar.gz",
        url_suffix="EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz",
        dest_dir="NYX_512x512x512",
        name="NYX", dims="512x512x512", dtype="f32", num_fields=6,
        description="AMR cosmology simulation (NYX), 6 fields",
    ),
    Dataset(
        key="MIRANDA",
        tarball="SDRBENCH-Miranda-256x384x384.tar.gz",
        url_suffix="Miranda/SDRBENCH-Miranda-256x384x384.tar.gz",
        dest_dir="MIRANDA_256x384x384",
        name="Miranda", dims="256x384x384", dtype="f64", num_fields=7,
        description="Turbulence simulation (Miranda), 7 fields, double precision",
    ),
    Dataset(
        key="QMCPACK",
        tarball="SDRBENCH-QMCPack.tar.gz",
        url_suffix="QMCPack/SDRBENCH-QMCPack.tar.gz",
        dest_dir="QMCPACK",
        name="QMCPACK", dims="69x69x115x288", dtype="f32", num_fields=288,
        description="Quantum Monte Carlo orbitals, 288 orbitals",
    ),
]

DATASET_KEYS: list[str] = [d.key for d in ALL_DATASETS]
_BY_KEY: dict[str, Dataset] = {d.key: d for d in ALL_DATASETS}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_metadata(ds: Dataset, dest: Path) -> None:
    (dest / "metadata.yaml").write_text(
        f"dataset:     {ds.name}\n"
        f"key:         {ds.key}\n"
        f"dims:        {ds.dims}\n"
        f"dtype:       {ds.dtype}\n"
        f"num_fields:  {ds.num_fields}\n"
        f"description: {ds.description}\n"
        f"source:      SDRBench (https://sdrbench.github.io)\n"
        f"url:         {ds.url}\n"
    )


def _fetch(url: str, tarball_dir: Path, filename: str) -> Path:
    tarpath = tarball_dir / filename
    if tarpath.exists():
        print(f"[cached]   {filename}")
        return tarpath
    print(f"[download] {filename}")
    subprocess.run(["wget", "-c", "-P", str(tarball_dir), url], check=True)
    return tarpath


def _extract(tarpath: Path, dest: Path) -> None:
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".tmp_") as tmpdir:
        tmp = Path(tmpdir)
        print(f"[extract]  {tarpath.name} -> {dest.name}/")
        with tarfile.open(tarpath) as tf:
            # filter='data' (3.12+) strips absolute paths and .. components
            if sys.version_info >= (3, 12):
                tf.extractall(tmp, filter="data")
            else:
                tf.extractall(tmp)  # noqa: S202
        entries = list(tmp.iterdir())
        dest.mkdir(parents=True, exist_ok=True)
        # if the tarball has a single top-level dir, hoist its contents
        if len(entries) == 1 and entries[0].is_dir():
            for item in entries[0].iterdir():
                shutil.move(str(item), dest)
        else:
            for item in entries:
                shutil.move(str(item), dest)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dataset(key: str, data_dir: Path, tarball_dir: Path) -> None:
    """Download, extract, and write metadata for one SDRBench dataset."""
    ds = _BY_KEY[key]
    dest = data_dir / ds.dest_dir

    if (dest / "metadata.yaml").exists() and any(dest.iterdir()):
        print(f"[skip]     {ds.dest_dir} — already populated")
        return

    tarpath = _fetch(ds.url, tarball_dir, ds.tarball)
    _extract(tarpath, dest)
    _write_metadata(ds, dest)

    n = sum(1 for f in dest.rglob("*") if f.is_file() and f.name != "metadata.yaml")
    print(f"[done]     {ds.dest_dir}/ ({n} data files)")
