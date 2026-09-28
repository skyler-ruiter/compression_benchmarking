#!/usr/bin/env python3
"""Peak device memory driver: native vs FZGM planning/specialization arms.

Protocol (docs/peak_memory_methodology.md, DESIGN D46/D47):
  * PRIMARY metric: NVML per-process resident memory (usedGpuMemory) of the tool's
    process tree, polled every sample_period_ms from a background thread; a
    process's peak is the max sampled value. A measurement that spawns several
    processes (native cuSZ and PFPL compress/decompress, FZGM split arm) takes the
    max over its processes. The cell's reported peak is the max over `reps`.
  * DECOMPOSITION metric: tools/cuda_mem_probe (LD_PRELOAD cudaMalloc-family
    interposer), live bytes. A subset of NVML's scope by construction.
  * FZGM self-report: Pipeline::getPeakMemoryUsage() from --report-json (pool
    working set only; excludes the CLI input buffer and the CUDA context).

Adapter argv comes from benchkit's own adapters (same mode conversion, dtype
retargeting and dims as the publication runs). Every subprocess.run() they make is
intercepted and measured. One JSONL line per (cell, rep) goes to raw.jsonl
(resumable); `aggregate` collapses them to one row per cell in cells.jsonl, each
carrying full provenance.

  python tools/peak_memory/run_peak_memory.py run configs/peak_memory/rq3_h100.yaml \
      --out results/peak_memory/<session> --fzgm-cli <pinned build>/bin/fzgmod-cli
  python tools/peak_memory/run_peak_memory.py aggregate results/peak_memory/<session>
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchkit.adapters.base import RunSpec  # noqa: E402
from benchkit.adapters.cusz_ref import CuszAdapter  # noqa: E402
from benchkit.adapters.cuszp import CuszpAdapter  # noqa: E402
from benchkit.adapters.fsz import FszAdapter  # noqa: E402
from benchkit.adapters.fzgm import FzgmAdapter  # noqa: E402
from benchkit.adapters.pfpl import PfplAdapter  # noqa: E402
from benchkit.config import DatasetCatalog  # noqa: E402
from nvml_sampler import Nvml, ProcessPeakSampler  # noqa: E402

PROBE_SO = REPO / "tools" / "cuda_mem_probe" / "libcudamemprobe.so"
CTX_BASELINE = REPO / "tools" / "peak_memory" / "ctx_baseline"
LMEM_DIR = REPO / "tools" / "peak_memory" / "lmem_probe"
STACK_SHIM = LMEM_DIR / "libstacklimit.so"
LMEM_ATTR_BIN = LMEM_DIR / "lmem_0"
SCHEMA = "peak_memory_cell/v1"

METRIC_DEFINITIONS = {
    "nvml_peak_bytes": (
        "max over reps of: max over the measurement's OS processes of: max over "
        "samples of the summed NVML usedGpuMemory of that process tree "
        "(nvmlDeviceGetComputeRunningProcesses_v3, polled every sample_period_ms)"),
    "probe_peak_bytes": (
        "max over reps/processes of cuda_mem_probe peak live cudaMalloc-family bytes "
        "(LD_PRELOAD; excludes context, modules, VMM, statically-linked cudart)"),
    "fzgm_self_peak_bytes": (
        "max over reps of FZGM --report-json memory.peak_device_bytes "
        "(Pipeline::getPeakMemoryUsage: pool working set; excludes CLI input buffer)"),
    "max_stack_limit_bytes": (
        "max over the measurement's processes of the CUDA context stack-size limit "
        "(cudaLimitStackSize, sampled after synchronizing calls by lmem_probe/"
        "libstacklimit.so); the driver raises it to the largest per-thread local frame "
        "launched and reserves it device-wide"),
    "lmem_reserved_above_default_bytes": (
        "(max_stack_limit_bytes - 1024) x maxThreadsPerSM x SM count: device memory the "
        "driver reserves for per-thread local memory beyond the default 1 KiB already "
        "inside the empty-context baseline (derived; validated by lmem_probe.cu)"),
    "input_bytes": "field element count x element size",
    "peak_over_input": "nvml_peak_bytes / input_bytes",
}


# --------------------------------------------------------------------------- utils

def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_state(repo: Path) -> dict:
    def g(*a):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *a], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"path": str(repo), "commit": g("rev-parse", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------- measured subprocess

class Recorder:
    """Replaces subprocess.run while active; measures every process it launches."""

    def __init__(self, nvml: Nvml, period_s: float, use_probe: bool, tmpdir: Path):
        self.nvml, self.period_s, self.use_probe, self.tmpdir = nvml, period_s, use_probe, tmpdir
        self.processes: list[dict] = []
        self._real_run = subprocess.run

    def run(self, argv, *args, **kw):
        capture = kw.pop("capture_output", False)
        text = kw.pop("text", False)
        timeout = kw.pop("timeout", None)
        env = dict(kw.pop("env", None) or os.environ)
        idx = len(self.processes)
        probe_out = self.tmpdir / f"probe_{idx}.json"
        probe_out.unlink(missing_ok=True)
        stack_out = self.tmpdir / f"stack_{idx}.json"
        stack_out.unlink(missing_ok=True)
        preload = [str(STACK_SHIM)]
        env["STACKLIMIT_OUT"] = str(stack_out)
        if self.use_probe:
            preload.append(str(PROBE_SO))
            env["CUDA_MEM_PROBE_OUT"] = str(probe_out)
        env["LD_PRELOAD"] = ":".join(preload)
        pre = set(self.nvml.processes())
        sampler = ProcessPeakSampler(self.nvml, self.period_s, preexisting=pre).start()
        t0 = time.monotonic()
        proc = subprocess.Popen(argv, *args, env=env, text=text,
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None, **kw)
        sampler.watch(proc.pid)
        try:
            out, err = proc.communicate(timeout=timeout)
        finally:
            sampler.stop()
        wall = time.monotonic() - t0
        probe = None
        if self.use_probe and probe_out.exists():
            try:
                probe = json.loads(probe_out.read_text())
            except json.JSONDecodeError:
                probe = {"parse_error": probe_out.read_text()[:200]}
        stack = None
        if stack_out.exists():
            with contextlib.suppress(json.JSONDecodeError):
                stack = json.loads(stack_out.read_text())
        self.processes.append({
            "argv0": Path(str(argv[0])).name,
            "argv": [str(a) for a in argv],
            "returncode": proc.returncode,
            "wall_s": round(wall, 4),
            "nvml_peak_bytes": sampler.peak_bytes,
            "nvml_samples": sampler.samples,
            "nvml_nonzero_samples": sampler.nonzero_samples,
            "foreign_processes": {str(k): v for k, v in sampler.foreign.items()},
            "preexisting_pids": sorted(pre),
            "probe": probe,
            "max_stack_limit_bytes": (stack or {}).get("max_stack_limit_bytes"),
        })
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)

    @contextlib.contextmanager
    def active(self):
        subprocess.run = self.run
        try:
            yield self
        finally:
            subprocess.run = self._real_run


@contextlib.contextmanager
def env_override(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------- cell plan

def family_entry(fam_cfg: dict, rank: int) -> tuple[dict, str]:
    if "by_rank" in fam_cfg:
        e = fam_cfg["by_rank"][rank]
        return e["native"], e["fzgm"]
    return fam_cfg["native"], fam_cfg["fzgm"]


def plan_cells(cfg: dict, catalog: DatasetCatalog, only_fields=None, only_families=None,
               only_arms=None):
    cells = []
    fz = cfg["fzgm"]
    for ds, fld in cfg["fields"]:
        key = f"{ds}/{fld}"
        if only_fields and key not in only_fields:
            continue
        fspec = catalog.resolve(ds, fld)
        cells.append({"cell_id": f"ctx_baseline|{key}", "family": "ctx_baseline",
                      "impl": "baseline", "arm": "ctx_baseline", "field": fspec})
        for fam, fcfg in cfg["families"].items():
            if only_families and fam not in only_families:
                continue
            native, fzgm_toml = family_entry(fcfg, len(fspec.dims))
            cells.append({"cell_id": f"{fam}|native|{key}", "family": fam, "impl": "native",
                          "arm": "native", "native": native, "field": fspec})
            for sc in native.get("static_control_fields", []):
                if sc == key:
                    cells.append({"cell_id": f"{fam}|native_static|{key}", "family": fam,
                                  "impl": "native", "arm": "native_static",
                                  "native": {**native, "bin_dir_env": native["static_bin_dir_env"]},
                                  "field": fspec})
            specs = ["off", "auto"] if fcfg["specializes"] else ["off"]
            for sp in specs:
                for pa, pcfg in fz["planning_arms"].items():
                    shapes = ["roundtrip"] + (["split"] if pa in fz.get("split_process_arms", []) else [])
                    for shape in shapes:
                        arm = f"fzgm:{pa}{'_split' if shape == 'split' else ''}:{sp}"
                        cells.append({"cell_id": f"{fam}|{arm}|{key}", "family": fam,
                                      "impl": "fzgm", "arm": arm, "planning": pa,
                                      "strategy": pcfg["strategy"], "coloring": pcfg["coloring"],
                                      "specialize": sp, "process_shape": shape,
                                      "pipeline": fzgm_toml, "field": fspec})
    if only_arms:
        cells = [c for c in cells if c["arm"] in only_arms or c["impl"] in only_arms]
    return cells


def build_native_adapter(native: dict):
    comp = native["compressor"]
    if comp == "cusz":
        return CuszAdapter(variant="cusz")
    if comp in ("cuszp2", "cuszp3"):
        return CuszpAdapter(version=int(comp[-1]), variant=f"{comp}_{native['pipeline']}")
    if comp == "pfpl":
        bin_dir = os.environ.get(native.get("bin_dir_env", "PFPL_BIN_DIR"))
        if not bin_dir:
            raise RuntimeError(f"set {native.get('bin_dir_env')} for PFPL")
        return PfplAdapter(variant="pfpl", cli_path=bin_dir)
    if comp == "fsz":
        a = FszAdapter(variant="fsz")
        if native.get("stock_cli"):
            a.hosttime_cli = None
        return a
    raise ValueError(comp)


# ---------------------------------------------------------------------- measuring

def measure(cell: dict, cfg: dict, args, nvml: Nvml, workroot: Path) -> dict:
    f = cell["field"]
    wd = workroot / cell["cell_id"].replace("|", "__").replace("/", "_").replace(":", "-")
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    rec = Recorder(nvml, cfg["sample_period_ms"] / 1000.0, not args.no_probe, wd)
    out: dict = {"status": "ok"}
    eb = float(cfg["error"]["bound"])
    mode = cfg["error"]["mode"]
    try:
        if cell["impl"] == "baseline":
            with rec.active():
                p = subprocess.run([str(CTX_BASELINE)], capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"ctx_baseline exit {p.returncode}: {p.stderr[-300:]}")
        elif cell["impl"] == "native":
            n = cell["native"]
            ad = build_native_adapter(n)
            spec = RunSpec(field=f, error_mode=mode, error_bound=eb,
                           pipeline=n["pipeline"], variant="reference")
            prep = ad.prepare(spec, wd)
            with rec.active():
                if n["compressor"] == "cusz":
                    # Native cuSZ is a two-process tool (-z, then -x from the archive).
                    c = ad.compress(spec, prep, wd)
                    ad.decompress(spec, c.compressed_path, wd)
                else:
                    ad.benchmark(spec, prep, 1, wd)
            out["native_argv_note"] = n
        else:
            ad = FzgmAdapter(variant=cell["family"], cli_path=args.fzgm_cli)
            spec = RunSpec(field=f, error_mode=mode, error_bound=eb,
                           pipeline=str(REPO / cell["pipeline"]), variant="fzgm")
            prep = ad.prepare(spec, wd)
            extra = ["--strategy", cell["strategy"]]
            if not cell["coloring"]:
                extra.append("--no-coloring")
            prep.config_args = [*prep.config_args, *extra]
            with env_override(FZ_SPECIALIZE=cell["specialize"]), rec.active():
                if cell["process_shape"] == "split":
                    c = ad.compress(spec, prep, wd)
                    ad.decompress(spec, c.compressed_path, wd)
                    reports = [wd / "z.json", wd / "x.json"]
                else:
                    ad.benchmark(spec, prep, 1, wd)
                    reports = [wd / "b.json"]
            out["fzgm_reports"] = [summarize_fzgm_report(r) for r in reports]
            shas = {r["git_sha"] for r in out["fzgm_reports"]}
            if shas != {cfg["fzgm"]["expected_git_sha"]}:
                raise RuntimeError(f"FZGM git_sha {shas} != expected "
                                   f"{cfg['fzgm']['expected_git_sha']}; wrong binary")
            out["pipeline_sha256"] = prep.pipeline_sha256
    except Exception as e:  # recorded, not fatal: failures are part of the evidence
        out["status"] = "failed"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-2000:]
        logs = sorted(wd.glob("*.log"))
        if logs:
            out["log_tail"] = logs[-1].read_text(errors="replace")[-1500:]
    out["processes"] = rec.processes
    bad = [p for p in rec.processes if p["returncode"] not in (0,)]
    if cell["impl"] == "native" and cell["native"]["compressor"] == "fsz":
        bad = [p for p in rec.processes if p["returncode"] >= 2]
    if bad and out["status"] == "ok":
        out["status"] = "failed"
        out["error"] = f"nonzero exit: {[p['returncode'] for p in bad]}"
    out["nvml_peak_bytes"] = max((p["nvml_peak_bytes"] for p in rec.processes), default=None)
    probes = [p["probe"]["peak_device_bytes"] for p in rec.processes
              if p.get("probe") and "peak_device_bytes" in p["probe"]]
    out["probe_peak_bytes"] = max(probes) if probes else None
    selfs = [r["peak_device_bytes"] for r in out.get("fzgm_reports", [])
             if r.get("peak_device_bytes") is not None]
    out["fzgm_self_peak_bytes"] = max(selfs) if selfs else None
    stacks = [p["max_stack_limit_bytes"] for p in rec.processes if p.get("max_stack_limit_bytes")]
    out["max_stack_limit_bytes"] = max(stacks) if stacks else None
    out["foreign_processes_seen"] = sorted({k for p in rec.processes
                                            for k in p["foreign_processes"]})
    if not args.keep_work:
        shutil.rmtree(wd, ignore_errors=True)
    return out


def summarize_fzgm_report(path: Path) -> dict:
    if not path.exists():
        return {"report": path.name, "missing": True, "git_sha": None}
    r = json.loads(path.read_text())
    tool = r.get("tool") or {}
    fusion = r.get("fusion") or {}
    return {
        "report": path.name,
        "status": r.get("status"),
        "git_sha": tool.get("git_sha") or r.get("git_sha"),
        "version": tool.get("version"),
        "peak_device_bytes": (r.get("memory") or {}).get("peak_device_bytes"),
        "memory_strategy": (r.get("config") or {}).get("memory_strategy"),
        "coloring_requested": (r.get("config") or {}).get("coloring"),
        "fusion": fusion,
        "quality": r.get("quality"),
        "compressed_bytes": (r.get("size") or {}).get("compressed_bytes"),
    }


# ---------------------------------------------------------------------- provenance

def capture_provenance(cfg: dict, cfg_path: Path, args, nvml: Nvml, fields) -> dict:
    virt = None
    with contextlib.suppress(Exception):
        q = subprocess.check_output(["nvidia-smi", "-q"], text=True)
        for line in q.splitlines():
            if "Virtualization Mode" in line:
                virt = line.split(":", 1)[1].strip()
                break
    natives = {}
    croot = Path(os.environ.get("COMPRESSORS_ROOT", Path.home() / "compressors"))
    for tool, pin in cfg.get("native_pins", {}).items():
        d = croot / pin["checkout"]
        head = None if pin.get("vendored") else git_state(d)["commit"]
        natives[tool] = {"pinned_commit": pin["commit"], "checkout": str(d),
                         "checkout_head": head, "vendored_fork": bool(pin.get("vendored")),
                         "head_matches_pin": (None if head is None else head == pin["commit"])}
    bins = {
        "cusz": os.environ.get("CUSZ_CLI"), "cuszp2": os.environ.get("CUSZP2_CLI"),
        "cuszp3": os.environ.get("CUSZP3_CLI"), "fsz": os.environ.get("FSZ_CLI"),
        "fzgmod-cli": args.fzgm_cli, "cuda_mem_probe": str(PROBE_SO),
        "ctx_baseline": str(CTX_BASELINE), "stacklimit_shim": str(STACK_SHIM),
    }
    for env in ("PEAKMEM_PFPL_SHADOW_BIN_DIR", "PFPL_BIN_DIR"):
        if os.environ.get(env):
            for b in sorted(Path(os.environ[env]).glob("f*/gpu/*_cuda")):
                bins[f"{env}:{b.parent.parent.name}/{b.name}"] = str(b)
    bin_hash = {k: (sha256_file(Path(v)) if v and Path(v).exists() else None)
                for k, v in bins.items()}
    fzgm_lib = Path(args.fzgm_cli).resolve().parents[1] / "libfzgmod.so"
    dev_attr = json.loads(subprocess.check_output([str(LMEM_ATTR_BIN)], text=True).strip())
    return {
        "schema": "peak_memory_session/v1",
        "created_utc": now(),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "config": cfg,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "machine_id_hint": os.environ.get("BENCHKIT_MACHINE_ID")},
        "gpu": {"name": nvml.device_name(), "uuid": nvml.device_uuid(),
                "driver": nvml.driver_version(), "virtualization_mode": virt,
                "cuda_module_loading": os.environ.get("CUDA_MODULE_LOADING", "(unset: LAZY default)"),
                "sm_count": dev_attr["sms"], "max_threads_per_sm": dev_attr["threads_per_sm"],
                "default_stack_limit_bytes": dev_attr["stack_limit"]},
        "benchmarking_repo": git_state(REPO),
        "fzgm": {"cli": args.fzgm_cli, "expected_git_sha": cfg["fzgm"]["expected_git_sha"],
                 "worktree": git_state(Path(args.fzgm_cli).resolve().parents[2]),
                 "libfzgmod_sha256": sha256_file(fzgm_lib) if fzgm_lib.exists() else None},
        "native_tools": natives,
        "binaries_sha256": bin_hash,
        "datasets": {f"{fs.dataset}/{fs.field}": {
            "path": str(fs.path), "dtype": fs.dtype, "dims": fs.dims,
            "input_bytes": fs.original_bytes, "expected_sha256": fs.expected_sha256,
            "sha256": sha256_file(fs.path)} for fs in fields},
        "metric_definitions": METRIC_DEFINITIONS,
        "sample_period_ms": cfg["sample_period_ms"],
        "probe_enabled": not args.no_probe,
    }


# ---------------------------------------------------------------------- commands

def wait_for_idle_gpu(nvml: Nvml, max_wait_s: float) -> list[int]:
    t0 = time.monotonic()
    while True:
        procs = nvml.processes()
        if not procs:
            return []
        if time.monotonic() - t0 > max_wait_s:
            return sorted(procs)
        time.sleep(5)


def cmd_run(args) -> int:
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.reps:
        cfg["reps"] = args.reps
    catalog = DatasetCatalog.load(REPO / "configs" / "datasets.yaml")
    cells = plan_cells(cfg, catalog, args.only_fields, args.only_families, args.only_arms)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nvml = Nvml(args.device)
    fields = {c["field"].path: c["field"] for c in cells}.values()
    print(f"[peakmem] {len(cells)} cells x {cfg['reps']} reps -> {out}", flush=True)
    if args.dry_run:
        for c in cells:
            print("  ", c["cell_id"])
        return 0
    for need in (CTX_BASELINE, STACK_SHIM, LMEM_ATTR_BIN):
        if not need.exists():
            raise SystemExit(f"build {need} first: tools/peak_memory/build_helpers.sh")

    prov_path = out / "session.json"
    if not prov_path.exists():
        print("[peakmem] hashing datasets + binaries for provenance ...", flush=True)
        prov = capture_provenance(cfg, cfg_path, args, nvml, fields)
        bad = [k for k, d in prov["datasets"].items()
               if d["expected_sha256"] and d["expected_sha256"] != d["sha256"]]
        if bad:
            raise SystemExit(f"dataset SHA-256 mismatch vs lock: {bad}")
        prov_path.write_text(json.dumps(prov, indent=1, default=str))

    raw = out / "raw.jsonl"
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            done.add((r["cell_id"], r["rep"]))
    workroot = out / "work"
    for rep in range(cfg["reps"]):
        for i, cell in enumerate(cells):
            if (cell["cell_id"], rep) in done:
                continue
            foreign = wait_for_idle_gpu(nvml, args.max_wait_idle_s)
            t = time.monotonic()
            m = measure(cell, cfg, args, nvml, workroot)
            row = {"cell_id": cell["cell_id"], "rep": rep, "utc": now(),
                   "gpu_busy_before": foreign,
                   **{k: v for k, v in cell.items() if k not in ("field", "native")},
                   "native": cell.get("native"),
                   "dataset": cell["field"].dataset, "field": cell["field"].field,
                   **m}
            with open(raw, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            mb = (m["nvml_peak_bytes"] or 0) / 1e6
            print(f"[peakmem] rep{rep} {i + 1}/{len(cells)} {cell['cell_id']:<58} "
                  f"{m['status']:<6} nvml={mb:9.1f}MB  ({time.monotonic() - t:5.1f}s)"
                  + (f"  !! {m.get('error', '')[:120]}" if m["status"] != "ok" else ""),
                  flush=True)
    return cmd_aggregate(argparse.Namespace(session=str(out)))


def cmd_aggregate(args) -> int:
    out = Path(args.session)
    prov = json.loads((out / "session.json").read_text())
    rows: dict[str, list[dict]] = {}
    for line in (out / "raw.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows.setdefault(r["cell_id"], []).append(r)
    prov_compact = {
        "host": prov["host"]["hostname"], "gpu": prov["gpu"],
        "benchmarking_commit": prov["benchmarking_repo"]["commit"],
        "benchmarking_dirty": prov["benchmarking_repo"]["dirty"],
        "fzgm_expected_git_sha": prov["fzgm"]["expected_git_sha"],
        "fzgm_cli": prov["fzgm"]["cli"], "libfzgmod_sha256": prov["fzgm"]["libfzgmod_sha256"],
        "native_tools": {k: v["pinned_commit"] for k, v in prov["native_tools"].items()},
        "config_sha256": prov["config_sha256"], "sample_period_ms": prov["sample_period_ms"],
        "metric_definitions": prov["metric_definitions"],
        "session_created_utc": prov["created_utc"],
    }
    cells_out = []
    for cid, reps in rows.items():
        reps.sort(key=lambda r: r["rep"])
        r0 = reps[0]
        ok = [r for r in reps if r["status"] == "ok"]
        ds_key = f"{r0['dataset']}/{r0['field']}"
        ds = prov["datasets"][ds_key]
        nv = [r["nvml_peak_bytes"] for r in ok if r["nvml_peak_bytes"]]
        pr = [r["probe_peak_bytes"] for r in ok if r.get("probe_peak_bytes")]
        sf = [r["fzgm_self_peak_bytes"] for r in ok if r.get("fzgm_self_peak_bytes")]
        sl = [r["max_stack_limit_bytes"] for r in ok if r.get("max_stack_limit_bytes")]
        g = prov["gpu"]
        lmem = ((max(sl) - g["default_stack_limit_bytes"]) * g["max_threads_per_sm"] * g["sm_count"]
                if sl else None)
        fusion = None
        for r in ok:
            for rep_ in r.get("fzgm_reports", []):
                if rep_.get("fusion"):
                    fusion = rep_["fusion"]
        inversions = sum(1 for r in ok for p in r["processes"]
                         if p.get("probe") and p["probe"].get("peak_device_bytes", 0) > p["nvml_peak_bytes"])
        cells_out.append({
            "schema": SCHEMA, "cell_id": cid,
            "family": r0["family"], "impl": r0["impl"], "arm": r0["arm"],
            "planning": r0.get("planning"), "strategy": r0.get("strategy"),
            "coloring": r0.get("coloring"), "specialize": r0.get("specialize"),
            "process_shape": r0.get("process_shape") or (
                "split" if r0["impl"] == "native" and r0.get("native", {}).get("compressor") in ("cusz", "pfpl")
                else "roundtrip"),
            "pipeline": r0.get("pipeline"), "native": r0.get("native"),
            "dataset": r0["dataset"], "field": r0["field"], "dtype": ds["dtype"],
            "dims": ds["dims"], "input_bytes": ds["input_bytes"], "dataset_sha256": ds["sha256"],
            "error": {"mode": prov["config"]["error"]["mode"], "bound": prov["config"]["error"]["bound"]},
            "status": "ok" if len(ok) == len(reps) and ok else ("partial" if ok else "failed"),
            "n_reps": len(reps), "n_ok": len(ok),
            "nvml_peak_bytes": max(nv) if nv else None,
            "nvml_rep_spread_bytes": (max(nv) - min(nv)) if nv else None,
            "probe_peak_bytes": max(pr) if pr else None,
            "fzgm_self_peak_bytes": max(sf) if sf else None,
            "max_stack_limit_bytes": max(sl) if sl else None,
            "lmem_reserved_above_default_bytes": lmem,
            "peak_over_input": (max(nv) / ds["input_bytes"]) if nv else None,
            "n_processes": len(r0["processes"]),
            "probe_gt_nvml_inversions": inversions,
            "foreign_processes_seen": sorted({p for r in reps for p in r.get("foreign_processes_seen", [])}),
            "fusion": fusion,
            "failure": None if ok else {"error": r0.get("error"), "log_tail": r0.get("log_tail")},
            "provenance": prov_compact,
        })
    cells_out.sort(key=lambda c: (c["input_bytes"], c["family"], c["arm"]))
    with open(out / "cells.jsonl", "w") as fh:
        for c in cells_out:
            fh.write(json.dumps(c, default=str) + "\n")
    n_ok = sum(c["status"] == "ok" for c in cells_out)
    print(f"[peakmem] aggregated {len(cells_out)} cells ({n_ok} ok) -> {out / 'cells.jsonl'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--out", required=True)
    r.add_argument("--fzgm-cli", required=True)
    r.add_argument("--device", type=int, default=0)
    r.add_argument("--reps", type=int)
    r.add_argument("--only-fields", nargs="*")
    r.add_argument("--only-families", nargs="*")
    r.add_argument("--only-arms", nargs="*", help="arm ids or impl names (native/fzgm/baseline)")
    r.add_argument("--no-probe", action="store_true")
    r.add_argument("--keep-work", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--max-wait-idle-s", type=float, default=1800)
    a = sub.add_parser("aggregate")
    a.add_argument("session")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_aggregate(args)


if __name__ == "__main__":
    sys.exit(main())
