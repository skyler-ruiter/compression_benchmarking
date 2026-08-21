#!/usr/bin/env python3
"""Create a lightweight, reproducible survey of the SDRBench S3D snapshot."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np


FIELDS = ("CH4", "O2", "CO", "CO2", "H2O", "N2", "TEMP", "PRES", "U", "V", "W")
SHAPE = (500, 500, 500)  # z, y, x: dataset metadata says x is fastest
SCALES = {"TEMP": 120.0, "PRES": 1.41837e5, "U": 347.2, "V": 347.2, "W": 347.2}
UNITS = {"TEMP": "K", "PRES": "Pa", "U": "m/s", "V": "m/s", "W": "m/s"}


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PNG input must be HxWx3")
    rows = b"".join(b"\0" + rgb[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
                     + png_chunk(b"IDAT", zlib.compress(rows, 6)) + png_chunk(b"IEND", b""))


def normalize(values: np.ndarray, signed: bool = False) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float64)
    if signed:
        limit = float(np.quantile(np.abs(finite), 0.99)) or 1.0
        return np.clip(values / (2.0 * limit) + 0.5, 0.0, 1.0)
    lo, hi = np.quantile(finite, [0.01, 0.99])
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def colorize(values: np.ndarray, signed: bool = False) -> np.ndarray:
    t = normalize(values, signed)
    if signed:
        r = np.where(t < 0.5, 2 * t, 1.0)
        b = np.where(t < 0.5, 1.0, 2 * (1 - t))
        g = 1.0 - 1.45 * np.abs(t - 0.5)
    else:
        r = np.clip(1.8 * t - 0.35, 0, 1)
        g = np.clip(2.1 * t - 0.9, 0, 1)
        b = np.clip(1.5 - 2.2 * np.abs(t - 0.32), 0, 1)
    return (np.stack((r, g, b), axis=-1) * 255).astype(np.uint8)


def add_border(image: np.ndarray, width: int = 2) -> np.ndarray:
    return np.pad(image, ((width, width), (width, width), (0, 0)), constant_values=245)


def midplanes(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z, y, x = (size // 2 for size in array.shape)
    return np.asarray(array[z, :, :]), np.asarray(array[:, y, :]), np.asarray(array[:, :, x])


def sample_flat(array: np.ndarray, count: int) -> np.ndarray:
    total = array.size
    indices = np.linspace(0, total - 1, min(count, total), dtype=np.int64)
    return np.asarray(array.reshape(-1)[indices], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("BENCHKIT_DATA_ROOT"))
    parser.add_argument("--sample-count", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("artifacts"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.data_root is None:
        local = Path("/media/volume/Compression_Data/sdrbench_data")
        if local.exists():
            args.data_root = local
        else:
            parser.error("pass --data-root or set BENCHKIT_DATA_ROOT")
    source = args.data_root / "S3D_500x500x500" / "vars_500x500x500"
    missing = [name for name in FIELDS if not (source / f"{name}.d64").is_file()]
    if missing:
        parser.error(f"missing S3D fields under {source}: {', '.join(missing)}")
    if args.output.exists():
        if not args.force:
            parser.error(f"{args.output} exists; pass --force to replace it")
        shutil.rmtree(args.output)
    figures = args.output / "figures"
    figures.mkdir(parents=True)

    samples: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    panels: list[np.ndarray] = []
    temp_plane: np.ndarray | None = None
    for name in FIELDS:
        path = source / f"{name}.d64"
        array = np.memmap(path, dtype="<f8", mode="r", shape=SHAPE)
        sample = sample_flat(array, args.sample_count)
        samples.append(sample)
        planes = midplanes(array)
        if name == "TEMP":
            temp_plane = planes[0].copy()
        signed = name in {"U", "V", "W"}
        panels.append(np.concatenate([add_border(colorize(p, signed)) for p in planes], axis=1))
        gradients = np.hypot(*np.gradient(planes[0]))
        scale = SCALES.get(name, 1.0)
        qs = np.quantile(sample, [0, .01, .05, .5, .95, .99, 1]) * scale
        rows.append({
            "field": name, "unit": UNITS.get(name, "mass fraction"),
            "scale": scale, "sample_count": sample.size,
            "min": qs[0], "p01": qs[1], "p05": qs[2], "median": qs[3],
            "p95": qs[4], "p99": qs[5], "max": qs[6],
            "mean": np.mean(sample) * scale, "std": np.std(sample) * scale,
            "mid_xy_gradient_mean_normalized": np.mean(gradients) / (np.ptp(planes[0]) or 1.0),
            "mid_xy_gradient_p99_normalized": np.quantile(gradients, .99) / (np.ptp(planes[0]) or 1.0),
        })
        del array

    write_png(figures / "s3d_orthogonal_slices.png", np.concatenate(panels, axis=0))
    matrix = np.corrcoef(np.vstack(samples))
    with (args.output / "sample_correlations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", *FIELDS])
        writer.writerows(([name, *matrix[i]] for i, name in enumerate(FIELDS)))
    correlation_rgb = colorize(matrix, signed=True)
    correlation_rgb = np.repeat(np.repeat(correlation_rgb, 36, axis=0), 36, axis=1)
    write_png(figures / "correlation_matrix.png", correlation_rgb)

    assert temp_plane is not None
    temp_gradient = np.hypot(*np.gradient(temp_plane))
    cutoff = np.quantile(temp_gradient, .90)
    proxy = np.where(temp_gradient >= cutoff, 255, 0).astype(np.uint8)
    proxy_rgb = np.repeat(proxy[..., None], 3, axis=2)
    write_png(figures / "temp_gradient_proxy.png", np.concatenate([
        add_border(colorize(temp_plane)), add_border(colorize(temp_gradient)), add_border(proxy_rgb)
    ], axis=1))

    with (args.output / "field_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "field_summary.md").open("w") as handle:
        handle.write("# S3D sampled field summary\n\n")
        handle.write(f"Source: `{source}`  \nSamples per field: {len(samples[0]):,}  \n")
        handle.write("Values below are converted to physical units where metadata supplies a scale.\n\n")
        handle.write("| field | unit | min | median | max | mean | std | XY gradient p99 / range |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write("| {field} | {unit} | {min:.5g} | {median:.5g} | {max:.5g} | "
                         "{mean:.5g} | {std:.5g} | {mid_xy_gradient_p99_normalized:.5g} |\n".format(**row))
        handle.write("\nThe gradient proxy selects the top 10% of temperature-gradient pixels on the "
                     "central XY plane. It is diagnostic only, not a validated flame mask.\n")
    print(f"wrote S3D survey to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
