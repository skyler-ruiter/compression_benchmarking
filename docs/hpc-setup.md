# HPC Machine Setup

Step-by-step instructions for setting up this benchmarking repo on a new HPC cluster.
BigRed200 (IU) is the reference machine; adapt paths and module names for other sites.

---

## Prerequisites

- Python 3.10+ (via `module load` or Conda)
- CUDA toolkit (via `module load cudatoolkit` or Spack)
- `fzgmod-cli` binary built from [FZGPUModules](https://github.com/szcompressor/FZGPUModules)
- Access to a scratch filesystem with ~200 GB free for SDRBench data + results

---

## 1. Clone the repo

```bash
git clone <repo-url> ~/research/compression_benchmarking
cd ~/research/compression_benchmarking
```

---

## 2. Python environment

```bash
module load python/3.12.11   # or whichever Python >= 3.10 your site provides
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify:
```bash
python -m benchkit --help
```

---

## 3. Build fzgmod-cli

If you don't already have a binary, build FZGPUModules:

```bash
git clone <fzgpumodules-url> ~/research/FZGPUModules
cd ~/research/FZGPUModules
mkdir build_bench && cd build_bench
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) fzgmod-cli
```

Note the path to the binary — you'll need it in step 4.

---

## 4. Site config

Copy the example and fill in your machine-specific paths:

```bash
cp configs/site.example.yaml configs/site.local.yaml
$EDITOR configs/site.local.yaml
```

`site.local.yaml` is gitignored (machine-specific). Minimal contents:

```yaml
# Path to the fzgmod-cli binary you built in step 3.
fzgmod_cli: /path/to/FZGPUModules/build_bench/bin/fzgmod-cli

# Where session output goes. Use scratch, not home (output can be large).
# Hardcode if $SCRATCH isn't set in login shells on your system.
results_root: /path/to/scratch/benchkit-results
```

**BigRed200 example:**
```yaml
fzgmod_cli: /N/u/<user>/BigRed200/research/FZGPUModules/build_bench/bin/fzgmod-cli
results_root: /N/scratch/<user>/benchkit-results
```

---

## 5. Download SDRBench data

Choose a scratch location with ~200 GB free, then run the download script via SLURM
(recommended — login nodes have time limits):

```bash
export DATA_DIR=/path/to/scratch/sdrbench_data

sbatch --account=<account> --partition=general --time=04:00:00 \
       --wrap="bash scripts/download-sdrbench.sh $DATA_DIR"
```

Or download interactively (may time out on login nodes):
```bash
bash scripts/download-sdrbench.sh /path/to/scratch/sdrbench_data
```

The script creates this layout under `$DATA_DIR`:
```
CESM_1800x3600/          79 f32 fields, 1800×3600
CESMATM_26x1800x3600/    33 f32 fields, 26×1800×3600
HURR_100x500x500/        13 f32 fields, 100×500×500
NYX_512x512x512/          6 f32 fields, 512×512×512
MIRANDA_256x384x384/      7 f64 fields, 256×384×384
HACCM_280953867/          6 f32 fields, 280,953,867 particles
EXAALT_2869440/           6 f32 fields as .dat2, 2,869,440 particles
QMCPACK/                  1 f32 array, 288×69×69×115
```

---

## 6. Tell benchkit where the data is

Set `BENCHKIT_DATA_ROOT` to the directory you passed to the download script:

```bash
# Add to ~/.bashrc (or your job script) so it's always set:
export BENCHKIT_DATA_ROOT=/path/to/scratch/sdrbench_data
```

`configs/datasets.yaml` uses `${BENCHKIT_DATA_ROOT}` to resolve all dataset paths.

---

## 7. Smoke test (interactive, single field)

Verify the end-to-end pipeline works before submitting the full matrix:

```bash
source .venv/bin/activate
export BENCHKIT_DATA_ROOT=/path/to/scratch/sdrbench_data

# Run a single dataset/field/bound to check fzgmod-cli is found and data resolves:
python -m benchkit run configs/experiments/smoke-bigred200.yaml
```

Create `configs/experiments/smoke-bigred200.yaml` (not tracked, machine-specific):
```yaml
name: smoke-bigred200
datasets: [CESM-2D]
fields:
  CESM-2D: [CLDHGH]
error:
  mode: rel_range
  bounds: [1.0e-3]
repetitions: 3
warmup_reps: 1
lock_clocks: false
runs:
  - {compressor: fzgm, variant: fzgpu, pipeline: configs/pipelines/fzgpu.toml}
```

A successful run prints a table and writes a `runs.jsonl` to the results directory.

---

## 8. Edit submit.slurm for your site

Open `scripts/submit.slurm` and update:

| Line | What to change |
|------|---------------|
| `--partition=gpu` | Your GPU partition name |
| `--account=r01156` | Your allocation account |
| `--array=0-7` | Keep in sync with `N=8` below it |
| `module load python/3.12.11` | Your site's Python module |
| `module load cudatoolkit` | Your site's CUDA module |
| `export FZGMOD_CLI=...` | Path to your fzgmod-cli binary |
| `export BENCHKIT_DATA_ROOT=...` | `$SCRATCH/sdrbench_data` or your data path |

---

## 9. Submit the full benchmark

```bash
# Submit the job array and capture the job ID:
ARRAY_ID=$(sbatch --parsable scripts/submit.slurm)
echo "Array job: $ARRAY_ID"

# Submit a merge job that runs after all shards finish:
sbatch --dependency=afterok:$ARRAY_ID --wrap \
  "cd $SLURM_SUBMIT_DIR && source .venv/bin/activate && \
   python -m benchkit merge $SCRATCH/benchkit-results/$ARRAY_ID"
```

Results land in `$SCRATCH/benchkit-results/<array-job-id>/`.
View the table after merging:
```bash
python -m benchkit report $SCRATCH/benchkit-results/<job-id>/
```

---

## Troubleshooting

**`fzgmod-cli not found`**
Check that `FZGMOD_CLI` is set and the binary exists:
```bash
echo $FZGMOD_CLI && ls -la $FZGMOD_CLI
```

**`data file not found`**
Check `BENCHKIT_DATA_ROOT` points to the right directory and the download completed:
```bash
echo $BENCHKIT_DATA_ROOT && ls $BENCHKIT_DATA_ROOT
```

**`$SCRATCH` is empty in login shell**
On some clusters `$SCRATCH` is only set inside SLURM jobs. Hardcode the path in
`configs/site.local.yaml` and your shell rc file rather than relying on the variable.

**MIRANDA fields fail**
Miranda is f64. The bundled pipeline TOMLs declare `input_type = "float32"`.
Verify fzgmod-cli f64 support or create f64-specific pipeline TOMLs before running.
Miranda is excluded from `sdrbench.yaml` by default for this reason.
