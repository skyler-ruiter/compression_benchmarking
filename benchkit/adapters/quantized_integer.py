"""Shared float <-> unsigned-integer wrapper for lossless integer codecs.

The quantizer is uniform midtread with an offset at the field minimum:

    q = round((x - min(x)) / (abs_eb / 4))
    x_hat = min(x) + q * (abs_eb / 4)

The quarter-bound bin width makes ideal quantization error at most one eighth of
the requested bound, leaving headroom for floating-point reconstruction at very
tight f32 bounds. The integer codec must reproduce q
bit-exactly. A small JSON header makes the resulting adapter artifact
self-describing; its bytes count toward the reported compressed size.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from .base import AdapterError, RunSpec

MAGIC = b"BKQINT1\0"
_LEN = struct.Struct("<Q")
_CHUNK_ELEMS = 8 * 1024 * 1024


def make_metadata(spec: RunSpec, codec: str, pipeline: str) -> dict:
    dtype = {"f32": np.float32, "f64": np.float64}.get(spec.field.dtype)
    if dtype is None:
        raise AdapterError(f"{codec}: quantized-float wrapper supports f32/f64, got {spec.field.dtype}")
    src = np.memmap(spec.field.path, dtype=dtype, mode="r",
                    shape=(spec.field.num_elements,))
    vmin, vmax = float(src.min()), float(src.max())
    eb = float(spec.error_bound)
    if spec.error_mode == "abs":
        abs_eb, basis = eb, "abs"
    elif spec.error_mode == "rel_range":
        abs_eb, basis = eb * (vmax - vmin), "range"
    elif spec.error_mode == "rel_maxabs":
        abs_eb, basis = eb * max(abs(vmin), abs(vmax)), "maxabs"
    else:
        raise AdapterError(f"{codec}: unsupported error mode '{spec.error_mode}'")
    if abs_eb < 0 or not np.isfinite(abs_eb):
        raise AdapterError(f"{codec}: invalid effective absolute bound {abs_eb}")

    constant = vmax == vmin
    if abs_eb == 0 and not constant:
        raise AdapterError(f"{codec}: zero effective bound on a non-constant field")
    # Reconstructing an f32 field from a double offset/step can itself round by
    # an ulp. A quarter-bound bin limits ideal quantization error to eb/8 and
    # leaves enough headroom for that conversion at the requested tight bounds.
    step = 1.0 if constant else abs_eb / 4.0
    qmax = 0 if constant else int(np.rint((vmax - vmin) / step))
    if qmax <= np.iinfo(np.uint16).max:
        int_dtype = "u16"
    elif qmax <= np.iinfo(np.uint32).max:
        int_dtype = "u32"
    else:
        raise AdapterError(
            f"{codec}: {qmax} quantization levels exceed uint32 at abs_eb={abs_eb:g}")

    return {
        "format": "benchkit-quantized-integer-v1",
        "codec": codec,
        "pipeline": pipeline,
        "original_dtype": spec.field.dtype,
        "integer_dtype": int_dtype,
        "dims_fast_to_slow": list(spec.field.dims),
        "num_elements": spec.field.num_elements,
        "offset": vmin,
        "step": step,
        "constant": constant,
        "requested_error_mode": spec.error_mode,
        "requested_error_bound": eb,
        "absolute_error_bound": abs_eb,
        "error_basis": basis,
    }


def metadata_sha256(meta: dict) -> str:
    raw = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_metadata(meta: dict, path: Path) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def quantize(input_path: Path, output_path: Path, meta: dict) -> float:
    """Write q and return wall milliseconds for quantization plus q-file write."""
    src_dtype = {"f32": np.float32, "f64": np.float64}[meta["original_dtype"]]
    q_dtype = {"u16": np.uint16, "u32": np.uint32}[meta["integer_dtype"]]
    n = int(meta["num_elements"])
    src = np.memmap(input_path, dtype=src_dtype, mode="r", shape=(n,))
    dst = np.memmap(output_path, dtype=q_dtype, mode="w+", shape=(n,))
    t0 = time.perf_counter()
    if meta["constant"]:
        dst[:] = 0
    else:
        off, step = float(meta["offset"]), float(meta["step"])
        for lo in range(0, n, _CHUNK_ELEMS):
            hi = min(n, lo + _CHUNK_ELEMS)
            dst[lo:hi] = np.rint((src[lo:hi] - off) / step).astype(q_dtype)
    dst.flush()
    elapsed = (time.perf_counter() - t0) * 1000.0
    del dst, src
    return elapsed


def dequantize(input_path: Path, output_path: Path, meta: dict) -> float:
    """Write reconstructed floats and return wall ms incl output-file write."""
    dst_dtype = {"f32": np.float32, "f64": np.float64}[meta["original_dtype"]]
    q_dtype = {"u16": np.uint16, "u32": np.uint32}[meta["integer_dtype"]]
    n = int(meta["num_elements"])
    src = np.memmap(input_path, dtype=q_dtype, mode="r", shape=(n,))
    dst = np.memmap(output_path, dtype=dst_dtype, mode="w+", shape=(n,))
    t0 = time.perf_counter()
    off, step = float(meta["offset"]), float(meta["step"])
    for lo in range(0, n, _CHUNK_ELEMS):
        hi = min(n, lo + _CHUNK_ELEMS)
        dst[lo:hi] = (off + src[lo:hi].astype(np.float64) * step).astype(dst_dtype)
    dst.flush()
    elapsed = (time.perf_counter() - t0) * 1000.0
    del dst, src
    return elapsed


def pack(payload: Path, artifact: Path, meta: dict) -> None:
    header = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    with open(artifact, "wb") as out, open(payload, "rb") as src:
        out.write(MAGIC)
        out.write(_LEN.pack(len(header)))
        out.write(header)
        shutil.copyfileobj(src, out, length=16 * 1024 * 1024)


def unpack(artifact: Path, payload: Path) -> dict:
    with open(artifact, "rb") as src:
        if src.read(len(MAGIC)) != MAGIC:
            raise AdapterError(f"{artifact}: not a benchkit quantized-integer container")
        raw_len = src.read(_LEN.size)
        if len(raw_len) != _LEN.size:
            raise AdapterError(f"{artifact}: truncated quantized-integer header")
        nheader = _LEN.unpack(raw_len)[0]
        if nheader > 1024 * 1024:
            raise AdapterError(f"{artifact}: unreasonable header size {nheader}")
        meta = json.loads(src.read(nheader))
        with open(payload, "wb") as out:
            shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    return meta
