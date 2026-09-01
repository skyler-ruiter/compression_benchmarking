"""The core loop: expand the run matrix and produce result rows.

For each cell (run-entry x dataset-field x error-bound):
  1. compress()   -> compressed blob (+ authoritative byte counts)
  2. decompress() -> decompressed array (for harness-owned quality)
  3. benchmark()  -> warm device-time arrays (in-process repeat)
  4. metrics      -> CR, bitrate, PSNR, NRMSE, eb-check, throughput (harness-computed)
  5. store.append -> one JSONL row
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import os
import yaml

from . import metrics
from .adapters import build_adapter
from .adapters.base import RunSpec
from .config import DatasetCatalog, ExperimentConfig
from .gpu import GpuSampler
from .identity import (execution_id, logical_cell_id, make_execution_context,
                       make_logical_cell, normalize_pipeline_ref, sha256_prefix,
                       tool_identity)
from .pipelines import PipelineToml
from .provenance import assign_provenance_id, capture_git_state, capture_session
from .store import ResultStore, sha256_file


_DIAGNOSTIC_SUFFIXES = frozenset((".log", ".json", ".toml"))
# Variables that change which FZGM kernels execute without changing the pipeline
# TOML or the binary, so they must participate in exact resume identity.
# FZ_SPECIALIZE is the current name for the finalize-time specialization policy
# (kernel fusion + single-pass decoupled-lookback + ...); FZ_FUSION is its
# still-honored deprecated alias. Both are recorded so an off/auto A/B keyed on
# either one produces distinct execution_ids.
_FZGM_BEHAVIOR_ENV = ("FZ_SPECIALIZE", "FZ_FUSION", "FZ_FUSION_NVRTC")


def _cleanup_cell_artifacts(workdir: Path) -> None:
    """Delete regenerable cell data while preserving diagnostic metadata.

    Adapters may create benchmark-only compressed or reconstructed files in addition
    to the canonical paths returned by compress()/decompress().  The runner cannot
    enumerate those names, so a no-retention run sweeps every non-diagnostic regular
    file after the result row (or failure row) has been recorded.
    """
    for leftover in workdir.iterdir():
        if leftover.is_file() and leftover.suffix not in _DIAGNOSTIC_SUFFIXES:
            leftover.unlink(missing_ok=True)


def cell_key(entry, dataset: str, field: str, mode: str, eb) -> str:
    """Legacy transition alias; new resume/merge logic uses H2 identities."""
    ebtxt = "toml" if eb is None else f"{eb:g}"
    return f"{entry.compressor}|{entry.variant}|{entry.pipeline}|{dataset}|{field}|{mode}|{ebtxt}"


def _adapter_key(entry) -> tuple:
    # Build declarations are part of the executable provenance. Two otherwise equal
    # entries must not accidentally share the first entry's declaration.
    return (entry.compressor, entry.variant, entry.cli_path, entry.tool_version,
            entry.tool_source_commit, entry.tool_build_flags, entry.tool_patch_sha256)


def _pipeline_resolution_inputs(entry, repo_root: Path) -> dict:
    requested = normalize_pipeline_ref(entry.pipeline, repo_root)
    candidate = Path(entry.pipeline)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    source_sha = sha256_file(candidate) if candidate.is_file() else None
    return {"requested": requested, "source_sha256": source_sha}


def _iter_cells(cfg: ExperimentConfig, catalog: DatasetCatalog):
    # Deterministic order so `--shard k/N` partitions the same way across array tasks.
    for entry in cfg.runs:
        for dataset in cfg.datasets:
            if not entry.applies_to(dataset):
                continue
            for fname in cfg.fields_for(dataset, catalog):
                fspec = catalog.resolve(dataset, fname)
                for eb in cfg.error_bounds:
                    yield entry, fspec, eb


def verify_dataset_inputs(cells, allow_unverified: bool = False):
    """Hash all distinct selected inputs and enforce their manifest declarations."""
    dataset_digests: dict[tuple[Path, int], str] = {}
    dataset_records: dict[tuple[str, str], dict] = {}
    for _, (_, fspec, _) in cells:
        digest_key = (fspec.path, fspec.original_bytes)
        # Do not use setdefault(key, sha256_prefix(...)): Python evaluates the
        # default eagerly, so a matrix with many pipelines would rehash the same
        # multi-GB field once per cell even though the cached value wins.
        if digest_key not in dataset_digests:
            dataset_digests[digest_key] = sha256_prefix(*digest_key)
        observed = dataset_digests[digest_key]
        if fspec.expected_sha256 is not None and observed != fspec.expected_sha256:
            raise RuntimeError(f"dataset checksum mismatch for {fspec.dataset}/{fspec.field}: "
                               f"expected {fspec.expected_sha256}, observed {observed}")
        if fspec.expected_sha256 is None and not allow_unverified:
            raise RuntimeError(
                f"dataset {fspec.dataset}/{fspec.field} has no declared sha256; add it "
                "to the dataset manifest or explicitly use --allow-unverified-datasets "
                "(the session will be marked non-publication-grade)")
        dataset_records[(fspec.dataset, fspec.field)] = {
            "dataset": fspec.dataset, "field": fspec.field,
            "path": str(fspec.path), "bytes_consumed": fspec.original_bytes,
            "expected_sha256": fspec.expected_sha256, "observed_sha256": observed,
            "verified": fspec.expected_sha256 == observed,
        }
    return dataset_digests, dataset_records


def run_experiment(cfg: ExperimentConfig, catalog: DatasetCatalog,
                   results_root: Path, repo_root: Path,
                   session_id: str | None = None,
                   shard: tuple[int, int] | None = None,
                   only_stale: set[str] | None = None,
                   allow_unverified_datasets: bool = False,
                   site_manifest: dict | None = None) -> ResultStore:
    cells = list(enumerate(_iter_cells(cfg, catalog)))
    if shard is not None:
        k, n = shard
        cells = [(i, c) for (i, c) in cells if i % n == k]

    # Integrity is a preflight gate: no compressor is invoked until every input this
    # shard may consume has an observed digest and, normally, a matching declared one.
    dataset_digests, dataset_records = verify_dataset_inputs(
        cells, allow_unverified_datasets)
    # Build adapters once; collect their provenance for the session manifest.
    adapters = {}
    adapter_prov = {}
    adapter_tools = {}
    for entry in cfg.runs:
        key = _adapter_key(entry)
        if key not in adapters:
            ad = build_adapter(entry)
            if not ad.is_available():
                raise RuntimeError(f"adapter '{key}' not available: {ad.provenance()}")
            adapters[key] = ad
            provenance = ad.provenance()
            declared = {
                "version": entry.tool_version or provenance.get("version"),
                "source_commit": (entry.tool_source_commit or
                                  provenance.get("source_commit") or
                                  provenance.get("commit")),
                "build_flags": entry.tool_build_flags or provenance.get("build_flags"),
                "local_patch_sha256": (entry.tool_patch_sha256 or
                                       provenance.get("local_patch_sha256")),
            }
            provenance = {**provenance,
                          "declared_build_provenance": declared,
                          "build_provenance_complete": all(declared.values())}
            adapter_tools[key] = tool_identity(provenance)
            label = f"{entry.compressor}:{entry.variant}"
            if label in adapter_prov and adapter_prov[label] != provenance:
                label += f"@{len(adapter_prov)}"
            adapter_prov[label] = provenance

    integrity = {
        "policy": "explicit_opt_out" if allow_unverified_datasets else "required",
        "all_verified": all(r["verified"] for r in dataset_records.values()),
        "publication_grade": (not allow_unverified_datasets and
                              all(r["verified"] for r in dataset_records.values()) and
                              all(p["build_provenance_complete"]
                                  for p in adapter_prov.values())),
        "datasets": list(dataset_records.values()),
    }
    manifest = capture_session(cfg.raw, repo_root, adapter_prov, session_id=session_id,
                               shard=shard, dataset_integrity=integrity)
    store = ResultStore(results_root, manifest["session_id"], shard=shard)
    experiment_text = cfg.source_text or yaml.safe_dump(cfg.raw, sort_keys=False)
    datasets_text = catalog.source_text or yaml.safe_dump(catalog._raw, sort_keys=False)
    artifacts = {
        "experiment": store.archive_bytes("experiment", experiment_text.encode(), ".yaml"),
        "datasets": store.archive_bytes("datasets", datasets_text.encode(), ".yaml"),
        "site": store.archive_json("site", site_manifest or {}),
        "resolved_datasets": store.archive_json("resolved-datasets", integrity),
    }
    if catalog.checksum_text is not None:
        artifacts["dataset_checksums"] = store.archive_bytes(
            "dataset-checksums", catalog.checksum_text.encode(), ".yaml")
    git_state, patch = capture_git_state(repo_root)
    artifacts["harness_patch"] = store.archive_bytes("harness", patch, ".patch")
    manifest["input_artifacts"] = artifacts
    manifest["harness"]["git"] = git_state
    manifest["provenance_id"] = assign_provenance_id(manifest)
    store.write_provenance(manifest)

    # Resume is exact-execution keyed. A legacy cell_key cannot prove that the dataset,
    # rendered config, binary, graph request, or harness state still matches.
    done = store.completed_execution_ids()
    declared_bounds: dict[str, float] = {}

    harness_identity = dict(manifest["harness"])
    # Current clocks are volatile observations on unlocked HPC GPUs. Binding resume to
    # them would rerun an otherwise identical session whenever DVFS sampled a different
    # instant. Model/driver/memory/ECC/max-clock and the software stack are stable
    # execution inputs; per-cell clock/throttle observations remain result diagnostics.
    stable_gpu = {k: v for k, v in manifest["gpu"].items()
                  if k not in {"sm_clock", "mem_clock"}}
    environment_identity = {
        "gpu": stable_gpu,
        "host": {k: v for k, v in manifest["host"].items() if k != "node"},
        "software": manifest["software"],
        # These variables change which FZGM kernels execute without changing the
        # TOML or binary. They must participate in exact resume identity.
        "fzgm_behavior": {key: os.environ[key] for key in _FZGM_BEHAVIOR_ENV
                          if key in os.environ},
    }

    def identities(entry, fspec, eb):
        pipeline = normalize_pipeline_ref(entry.pipeline, repo_root)
        logical_eb = eb
        if cfg.error_mode == "from_toml":
            if pipeline not in declared_bounds:
                declared_bounds[pipeline] = PipelineToml.load(entry.pipeline).declared_eb_mode()[0]
            logical_eb = declared_bounds[pipeline]
        logical = make_logical_cell(
            compressor=entry.compressor, variant=entry.variant, pipeline=pipeline,
            dataset=fspec.dataset, field=fspec.field, error_mode=cfg.error_mode,
            error_bound=logical_eb)
        logical_id = logical_cell_id(logical)
        digest_key = (fspec.path, fspec.original_bytes)
        if digest_key not in dataset_digests:
            dataset_digests[digest_key] = sha256_prefix(*digest_key)
        run_entry = asdict(entry)
        # Dataset scoping controls matrix membership, not the behavior of this cell.
        run_entry.pop("only_datasets", None)
        run_entry.pop("skip_datasets", None)
        run_entry["pipeline"] = pipeline
        context = make_execution_context(
            logical_id=logical_id,
            run_parameters={
                "run_entry": run_entry,
                "repetitions": cfg.repetitions,
                "warmup_reps": cfg.warmup_reps,
                "timing_cv_threshold": cfg.timing_cv_threshold,
                "lock_clocks": cfg.lock_clocks,
            },
            resolved_config={
                **_pipeline_resolution_inputs(entry, repo_root),
                "error_mode": cfg.error_mode,
                "error_bound": None if logical_eb is None else float(logical_eb),
                "dtype": fspec.dtype,
                "dims": list(fspec.dims),
                "dim_order": fspec.dim_order,
            },
            dataset={
                "sha256": dataset_digests[digest_key],
                "bytes": fspec.original_bytes,
            },
            tool=adapter_tools[_adapter_key(entry)], harness=harness_identity,
            environment=environment_identity)
        return logical, logical_id, context, execution_id(context)

    if only_stale is not None:
        # Re-measure a named subset. Two things have to happen together: keep only the
        # stale cells, AND drop them from `done` so resume does not immediately skip
        # the very rows we are here to replace. The new rows are APPENDED — nothing is
        # edited in place — so the old measurement survives in runs.jsonl and `merge`
        # is what picks the newer one (see cmd_merge). That keeps "this stage got 1.8x
        # faster" answerable from the file instead of destroying the evidence.
        before = len(cells)
        cells = [(i, c) for (i, c) in cells
                 if identities(c[0], c[1], c[2])[1] in only_stale]
        # Explicit invalidation is authority to append another attempt even when the
        # exact execution inputs have not changed (for example, a noise re-measurement
        # selected with --stage rather than --against-build).
        done -= {identities(c[0], c[1], c[2])[3] for _, c in cells}
        print(f"[stale] {len(cells)} of {before} cells selected for re-measurement",
              flush=True)
        if not cells:
            print("[stale] nothing to do — the stale set does not intersect this "
                  "experiment (wrong experiment file, or wrong shard?)", flush=True)

    shard_txt = "" if shard is None else f" shard {shard[0]}/{shard[1]}"
    n_skip = sum(1 for _, (e, f, eb) in cells
                 if identities(e, f, eb)[3] in done)
    print(f"[session] {store.session_id}{shard_txt}  ->  {store.dir}", flush=True)
    print(f"[plan] {len(cells)} cells this task, {n_skip} already done (skipped)", flush=True)

    n_runs = cfg.warmup_reps + cfg.repetitions
    for idx, (entry, fspec, eb) in cells:
        key = cell_key(entry, fspec.dataset, fspec.field, cfg.error_mode, eb)
        logical, logical_id, execution_context, exec_id = identities(entry, fspec, eb)
        if exec_id in done:
            continue
        run_id = f"{cfg.name}-{idx:04d}-{uuid4().hex[:12]}"
        adapter = adapters[_adapter_key(entry)]
        spec = RunSpec(field=fspec, error_mode=cfg.error_mode, error_bound=eb,
                       pipeline=entry.pipeline, variant=entry.variant, graph=entry.graph)
        wd = store.workdir(run_id)
        rendered_pipeline = None
        # Display only. The legacy alias retains its historical "toml" spelling;
        # H2 resume and merge use the canonical identities above.
        ebtxt = f"{eb:g}" if eb is not None else (
            "lossless" if cfg.error_mode == "lossless" else "toml")
        label = (f"{entry.compressor}:{entry.variant} [{Path(entry.pipeline).name}] "
                 f"{fspec.dataset}/{fspec.field} eb={ebtxt}")
        try:
            prep = adapter.prepare(spec, wd)
            rendered_pipeline = store.archive_rendered_pipeline(prep.pipeline_path)
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
            # Same treatment for the compressed artifact itself, once nothing else in
            # this cell needs it on disk (compress/decompress/benchmark have all run by
            # this point). Its size is already captured in `size` independent of this.
            csha = sha256_file(comp.compressed_path)
            if not cfg.retain_compressed:
                comp.compressed_path.unlink(missing_ok=True)
            tcv = cfg.timing_cv_threshold
            ct = metrics.summarize_timing(bench.compress_device_ms_all,
                                          size.original_bytes, cfg.warmup_reps, tcv)
            dt = metrics.summarize_timing(bench.decompress_device_ms_all,
                                          size.original_bytes, cfg.warmup_reps, tcv)
            # Host wall time around the same calls, when the tool reports it. See
            # BenchmarkResult.compress_host_ms_all and DESIGN.md D33 — a device-only
            # number is only comparable across tools if you know what each excludes.
            cht = metrics.summarize_timing(bench.compress_host_ms_all,
                                           size.original_bytes, cfg.warmup_reps, tcv) \
                if bench.compress_host_ms_all else None
            dht = metrics.summarize_timing(bench.decompress_host_ms_all,
                                           size.original_bytes, cfg.warmup_reps, tcv) \
                if bench.decompress_host_ms_all else None
            thermal = bool(gpu.get("throttled_thermal"))
            reliable = ct.stable and dt.stable and not thermal
            row = _row(run_id, store.session_id, entry, fspec, cfg, prep,
                       size, qual, ct, dt, bench, cht, dht)
            row["cell_key"] = key
            row.update({
                "identity_schema_version": 1,
                "logical_cell_id": logical_id,
                "execution_id": exec_id,
                "logical_cell": logical,
                "execution_context": execution_context,
                "dataset_sha256": execution_context["dataset"]["sha256"],
                "provenance_id": manifest["provenance_id"],
                "rendered_pipeline_artifact": rendered_pipeline,
            })
            row["decompressed_sha256"] = dsha
            row["decompressed_retained"] = cfg.retain_decompressed
            row["compressed_sha256"] = csha
            row["compressed_retained"] = cfg.retain_compressed
            row["gpu_sampling"] = gpu
            row["timing_reliable"] = reliable
            store.append(row)
            flag = "" if reliable else (
                " !THERMAL-THROTTLE" if thermal else
                f" !UNSTABLE(cv c={ct.cv:.2f} d={dt.cv:.2f})")
            if bench.graph_requested:
                gtxt = ("on" if bench.graph_active else
                        "off(fallback)" if bench.graph_active is False else "?")
                flag += f" graph={gtxt}"
            print(f"  [{idx}] OK   {label}  CR={size.cr:.2f} "
                  f"PSNR={qual.psnr:.2f}dB cT={ct.throughput_gbs:.1f} "
                  f"dT={dt.throughput_gbs:.1f}GB/s eb_ok={qual.eb_satisfied}{flag}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - one bad cell shouldn't kill the matrix
            msg = str(e)
            phase = next((p for p in ("compress", "decompress", "benchmark", "prepare")
                          if f"{p} failed" in msg), "unknown")
            store.append({"run_id": run_id, "session_id": store.session_id,
                          "cell_key": key,
                          "identity_schema_version": 1,
                          "logical_cell_id": logical_id,
                          "execution_id": exec_id,
                          "logical_cell": logical,
                          "execution_context": execution_context,
                          "dataset_sha256": execution_context["dataset"]["sha256"],
                          "provenance_id": manifest["provenance_id"],
                          "rendered_pipeline_artifact": rendered_pipeline,
                          "compressor": entry.compressor, "variant": entry.variant,
                          "pipeline": entry.pipeline, "dataset": fspec.dataset,
                          "field": fspec.field, "error_mode": cfg.error_mode,
                          "error_bound": logical["error_bound"],
                          "status": "fail", "fail_phase": phase,
                          "error_type": type(e).__name__,
                          "error_message": msg})
            print(f"  [{idx}] FAIL {label}  -> {e}", flush=True)
        finally:
            # This includes adapter-owned benchmark scratch such as d_bench.bin, not
            # just the canonical artifacts returned by compress()/decompress().  Run
            # after append so the checksums, sizes, quality, and failure evidence are
            # durable before the regenerable data is removed.  `finally` also covers
            # interruptions and unexpected exception types.  See DESIGN.md D26.
            if not cfg.retain_decompressed and not cfg.retain_compressed:
                _cleanup_cell_artifacts(wd)

    # A previous merged result may coexist with newly appended shard rows during a
    # repair run.  Summarize this task's file, not that older canonical merge.
    rows = store.load_rows(canonical=False)
    unreliable = [r for r in rows if r.get("status") == "ok"
                  and r.get("timing_reliable") is False]
    if unreliable:
        print(f"\n[timing] {len(unreliable)} cell(s) flagged unreliable "
              f"(high variance / thermal throttle) — see timing_reliable in rows. "
              f"On unlocked GPUs prefer the *_device_ms_min columns.")
    return store


def _row(run_id, session_id, entry, f, cfg, prep, size, qual, ct, dt, bench,
         cht=None, dht=None) -> dict:
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
        "error_bound": None if cfg.error_mode == "lossless" else prep.eb,
        "rel_basis": qual.rel_basis,
        "eb_abs_effective": qual.eb_abs_effective,
        "err_over_bound": qual.err_over_bound,
        "compressed_bytes": size.compressed_bytes,
        "cr": size.cr,
        "bitrate_bits_per_elem": size.bitrate_bits_per_elem,
        "compress_device_ms_median": ct.median_ms,
        "compress_device_ms_min": ct.min_ms,
        "compress_device_ms_max": ct.max_ms,
        "decompress_device_ms_median": dt.median_ms,
        "decompress_device_ms_min": dt.min_ms,
        "decompress_device_ms_max": dt.max_ms,
        "compress_throughput_gbs": ct.throughput_gbs,
        "decompress_throughput_gbs": dt.throughput_gbs,
        "throughput_unit": "GB/s_decimal",
        # Host wall time around the same calls (None when the tool reports none).
        # `*_host_over_device` is the number to look at: 1.0 means the device figure
        # excludes nothing, and a large value means a device-only cross-tool
        # comparison is not measuring the same thing on both sides. FZGM split-mode
        # compress runs ~2.9x; nvCOMP ~1.0x. See DESIGN.md D33.
        "compress_host_ms_median": None if cht is None else cht.median_ms,
        "decompress_host_ms_median": None if dht is None else dht.median_ms,
        "compress_host_throughput_gbs": None if cht is None else cht.throughput_gbs,
        "decompress_host_throughput_gbs": None if dht is None else dht.throughput_gbs,
        "compress_host_over_device": (
            None if cht is None or ct.median_ms <= 0 else cht.median_ms / ct.median_ms),
        "decompress_host_over_device": (
            None if dht is None or dt.median_ms <= 0 else dht.median_ms / dt.median_ms),
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
        # Peak device memory. MB (1e6) to match the throughput columns' decimal
        # convention; the raw byte count is kept alongside so nothing is lost to
        # rounding when the ablation compares two close peaks. None when the tool
        # reports no peak — most native baselines — which is *not* the same as 0.
        "peak_device_bytes": bench.peak_device_bytes,
        "peak_device_mb": (None if bench.peak_device_bytes is None
                           else bench.peak_device_bytes / 1e6),
        # fzgm: which arm of the coloring ablation this row is. Rows with
        # coloring_enabled False are the "worst-case reservation" arm.
        "coloring_enabled": bench.coloring_enabled,
        # fzgm: stage-level events that change how a row should be compared.
        # `run_notes` is the full record; the boolean is the one case that affects
        # CR comparability today and is promoted so it can be filtered on directly.
        # False (not None) when the tool reported notes support and had none.
        "run_notes": bench.run_notes or None,
        "huffman_adaptive_fallback": (
            None if bench.coloring_enabled is None      # non-fzgm: concept N/A
            else any("huffman_adaptive_fallback" in v for v in bench.run_notes.values())),
        "stages": bench.stages,
        "stage_versions": bench.stage_versions,
        "graph_requested": bench.graph_requested,
        "graph_active": bench.graph_active,
        "graph_reason": bench.graph_reason,
        # Finalize-time compression specialization. The resolved CLI report is
        # authoritative: an Auto request may still fall back to staged execution.
        "fusion_policy": None if bench.fusion is None else bench.fusion.get("policy"),
        "fusion_legal_group_count": (
            None if bench.fusion is None else bench.fusion.get("legal_group_count")),
        "fusion_installed_group_count": (
            None if bench.fusion is None else bench.fusion.get("installed_group_count")),
        "fusion_installed_stage_count": (
            None if bench.fusion is None else bench.fusion.get("installed_stage_count")),
        "fusion_inverse_installed_group_count": (
            None if bench.fusion is None else bench.fusion.get("inverse_installed_group_count")),
        "fusion_inverse_installed_stage_count": (
            None if bench.fusion is None else bench.fusion.get("inverse_installed_stage_count")),
        "fusion_fallback_reason": (
            None if bench.fusion is None else bench.fusion.get("fallback_reason")),
        "fusion_groups": None if bench.fusion is None else bench.fusion.get("groups", []),
        "fusion_inverse_groups": (
            None if bench.fusion is None else bench.fusion.get("inverse_groups", [])),
        "status": "ok",
        "error_message": None,
    }
