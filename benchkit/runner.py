"""The core loop: expand the run matrix and produce result rows.

For each cell (run-entry x dataset-field x error-bound):
  1. compress()   -> compressed blob (+ authoritative byte counts)
  2. decompress() -> decompressed array (for harness-owned quality)
  3. benchmark()  -> warm device-time arrays (in-process repeat)
  4. metrics      -> CR, bitrate, PSNR, NRMSE, eb-check, throughput (harness-computed)
  5. store.append -> one JSONL row
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import metrics
from .adapters import build_adapter
from .adapters.base import RunSpec
from .config import DatasetCatalog, ExperimentConfig
from .gpu import GpuSampler
from .provenance import capture_session
from .store import ResultStore, sha256_file


def cell_key(entry, dataset: str, field: str, mode: str, eb) -> str:
    """Deterministic identity of one cell — used for shard assignment and resume-skip."""
    ebtxt = "toml" if eb is None else f"{eb:g}"
    return f"{entry.compressor}|{entry.variant}|{entry.pipeline}|{dataset}|{field}|{mode}|{ebtxt}"


def _iter_cells(cfg: ExperimentConfig, catalog: DatasetCatalog):
    # Deterministic order so `--shard k/N` partitions the same way across array tasks.
    for entry in cfg.runs:
        for dataset in cfg.datasets:
            for fname in cfg.fields_for(dataset, catalog):
                fspec = catalog.resolve(dataset, fname)
                for eb in cfg.error_bounds:
                    yield entry, fspec, eb


def run_experiment(cfg: ExperimentConfig, catalog: DatasetCatalog,
                   results_root: Path, repo_root: Path,
                   session_id: str | None = None,
                   shard: tuple[int, int] | None = None) -> ResultStore:
    # Build adapters once; collect their provenance for the session manifest.
    adapters = {}
    adapter_prov = {}
    for entry in cfg.runs:
        key = f"{entry.compressor}:{entry.variant}"
        if key not in adapters:
            ad = build_adapter(entry)
            if not ad.is_available():
                raise RuntimeError(f"adapter '{key}' not available: {ad.provenance()}")
            adapters[key] = ad
            adapter_prov[key] = ad.provenance()

    manifest = capture_session(cfg.raw, repo_root, adapter_prov,
                               session_id=session_id, shard=shard)
    store = ResultStore(results_root, manifest["session_id"], shard=shard)
    store.write_provenance(manifest)

    # Resume: cells already completed OK in this session dir (any shard) are skipped.
    done = store.completed_keys()
    cells = list(enumerate(_iter_cells(cfg, catalog)))
    if shard is not None:
        k, n = shard
        cells = [(i, c) for (i, c) in cells if i % n == k]
    shard_txt = "" if shard is None else f" shard {shard[0]}/{shard[1]}"
    n_skip = sum(1 for _, (e, f, eb) in cells
                 if cell_key(e, f.dataset, f.field, cfg.error_mode, eb) in done)
    print(f"[session] {store.session_id}{shard_txt}  ->  {store.dir}")
    print(f"[plan] {len(cells)} cells this task, {n_skip} already done (skipped)")

    n_runs = cfg.warmup_reps + cfg.repetitions
    for idx, (entry, fspec, eb) in cells:
        run_id = f"{cfg.name}-{idx:04d}"
        key = cell_key(entry, fspec.dataset, fspec.field, cfg.error_mode, eb)
        if key in done:
            continue
        adapter = adapters[f"{entry.compressor}:{entry.variant}"]
        spec = RunSpec(field=fspec, error_mode=cfg.error_mode, error_bound=eb,
                       pipeline=entry.pipeline, variant=entry.variant)
        wd = store.workdir(run_id)
        ebtxt = "toml" if eb is None else f"{eb:g}"
        label = (f"{entry.compressor}:{entry.variant} [{Path(entry.pipeline).name}] "
                 f"{fspec.dataset}/{fspec.field} eb={ebtxt}")
        try:
            prep = adapter.prepare(spec, wd)
            comp = adapter.compress(spec, prep, wd)
            dec = adapter.decompress(spec, comp.compressed_path, wd)
            # Sample GPU clocks/throttle reasons concurrently with the timed benchmark.
            with GpuSampler() as samp:
                bench = adapter.benchmark(spec, prep, n_runs, wd)
            gpu = samp.summary()

            size = metrics.compute_size(comp.original_bytes, comp.compressed_bytes,
                                        fspec.num_elements)
            qual = metrics.compute_quality(fspec.path, dec.decompressed_path,
                                           fspec.dtype, fspec.num_elements,
                                           prep.eb, prep.basis)
            # Checksum the decompressed output, then (by default) delete it — it is
            # ~original-sized and regenerable from c.fzm. Keeps the repo under budget.
            dsha = sha256_file(dec.decompressed_path)
            if not cfg.retain_decompressed:
                dec.decompressed_path.unlink(missing_ok=True)
            tcv = cfg.timing_cv_threshold
            ct = metrics.summarize_timing(bench.compress_device_ms_all,
                                          size.original_bytes, cfg.warmup_reps, tcv)
            dt = metrics.summarize_timing(bench.decompress_device_ms_all,
                                          size.original_bytes, cfg.warmup_reps, tcv)
            thermal = bool(gpu.get("throttled_thermal"))
            reliable = ct.stable and dt.stable and not thermal
            row = _row(run_id, store.session_id, entry, fspec, cfg, prep,
                       size, qual, ct, dt, bench)
            row["cell_key"] = key
            row["decompressed_sha256"] = dsha
            row["decompressed_retained"] = cfg.retain_decompressed
            row["gpu_sampling"] = gpu
            row["timing_reliable"] = reliable
            store.append(row)
            flag = "" if reliable else (
                " !THERMAL-THROTTLE" if thermal else
                f" !UNSTABLE(cv c={ct.cv:.2f} d={dt.cv:.2f})")
            print(f"  [{idx}] OK   {label}  CR={size.cr:.2f} "
                  f"PSNR={qual.psnr:.2f}dB cT={ct.throughput_gbs:.1f} "
                  f"dT={dt.throughput_gbs:.1f}GB/s eb_ok={qual.eb_satisfied}{flag}")
        except Exception as e:  # noqa: BLE001 - one bad cell shouldn't kill the matrix
            store.append({"run_id": run_id, "session_id": store.session_id,
                          "cell_key": key,
                          "compressor": entry.compressor, "variant": entry.variant,
                          "pipeline": entry.pipeline, "dataset": fspec.dataset,
                          "field": fspec.field, "error_bound": eb,
                          "status": "fail", "error_message": str(e)})
            print(f"  [{idx}] FAIL {label}  -> {e}")

    rows = store.load_rows()
    unreliable = [r for r in rows if r.get("status") == "ok"
                  and r.get("timing_reliable") is False]
    if unreliable:
        print(f"\n[timing] {len(unreliable)} cell(s) flagged unreliable "
              f"(high variance / thermal throttle) — see timing_reliable in rows. "
              f"On unlocked GPUs prefer the *_device_ms_min columns.")
    return store


def _row(run_id, session_id, entry, f, cfg, prep, size, qual, ct, dt, bench) -> dict:
    return {
        "run_id": run_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "compressor": entry.compressor,
        "variant": entry.variant,
        "pipeline": entry.pipeline,
        "pipeline_ref": prep.pipeline_ref,
        "pipeline_sha256": prep.pipeline_sha256,
        "native_mode": prep.native_mode,
        "dataset": f.dataset,
        "field": f.field,
        "dtype": f.dtype,
        "dims": f.dims,
        "dim_order": f.dim_order,
        "num_elements": f.num_elements,
        "original_bytes": size.original_bytes,
        "error_mode": cfg.error_mode,
        "error_bound": prep.eb,
        "rel_basis": qual.rel_basis,
        "eb_abs_effective": qual.eb_abs_effective,
        "err_over_bound": qual.err_over_bound,
        "compressed_bytes": size.compressed_bytes,
        "cr": size.cr,
        "bitrate_bits_per_elem": size.bitrate_bits_per_elem,
        "compress_device_ms_median": ct.median_ms,
        "compress_device_ms_min": ct.min_ms,
        "decompress_device_ms_median": dt.median_ms,
        "decompress_device_ms_min": dt.min_ms,
        "compress_throughput_gbs": ct.throughput_gbs,
        "decompress_throughput_gbs": dt.throughput_gbs,
        "throughput_unit": "GB/s_decimal",
        "timing_reps": ct.n,
        "compress_cv": ct.cv,
        "decompress_cv": dt.cv,
        "compress_rel_spread": ct.rel_spread,
        "decompress_rel_spread": dt.rel_spread,
        "compress_stable": ct.stable,
        "decompress_stable": dt.stable,
        "psnr": qual.psnr,
        "nrmse": qual.nrmse,
        "max_abs_err": qual.max_abs_err,
        "max_rel_err": qual.max_rel_err,
        "eb_satisfied": qual.eb_satisfied,
        # cross-check: tool's own PSNR vs harness-computed (should agree closely)
        "native_psnr": (bench.native_quality or {}).get("psnr_db"),
        "stages": bench.stages,
        "status": "ok",
        "error_message": None,
    }
