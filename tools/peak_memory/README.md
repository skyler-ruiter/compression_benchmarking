# peak_memory

Peak device memory study for FZGM paper RQ3 (native vs FZGM staged/Auto, plus the
planning ablation). Protocol and findings: `docs/peak_memory_methodology.md`, DESIGN D47.

| File | Role |
|---|---|
| `run_peak_memory.py` | driver: `run` a config into `raw.jsonl`/`cells.jsonl`/`session.json`; `aggregate` |
| `nvml_sampler.py` | ctypes NVML per-process-tree peak sampler (0.5 ms), no pynvml dependency |
| `diagnose.py` | one-factor diagnostic cells (other binary, literal bound, pool off, `--runs N`) |
| `ctx_baseline.cu` | empty-CUDA-context floor |
| `lmem_probe/lmem_probe.cu` | local-memory (stack) reservation microbenchmark |
| `lmem_probe/stacklimit_shim.c` | LD_PRELOAD shim: max `cudaLimitStackSize` a process reached |
| `build_helpers.sh` | builds the three helpers above |
| `build_pfpl_shadow.sh` | PFPL rebuilt with `-cudart shared` so the probe can see it |

```bash
source scripts/env-jetstream2.sh
make -C tools/cuda_mem_probe && tools/peak_memory/build_helpers.sh
tools/peak_memory/build_pfpl_shadow.sh
export PEAKMEM_PFPL_SHADOW_BIN_DIR=$HOME/compressors-shadow/pfpl-cudart-shared/bin
python tools/peak_memory/run_peak_memory.py run configs/peak_memory/rq3_h100.yaml \
    --out results/peak_memory/<session> --fzgm-cli <d511ebc build>/bin/fzgmod-cli
python scripts/build_peak_memory_artifact.py --session results/peak_memory/<session> \
    --out-dir <paper_organizer>/projects/FZGM/evidence/publication/memory
```

The driver waits for an idle GPU before every cell and records any foreign GPU process
seen during a measurement. It verifies FZGM's `git_sha` on every row and the dataset
SHA-256 against `configs/datasets.checksums.yaml`.
