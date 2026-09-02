# Pipeline Specialization campaign — per-machine runbook

Refreshes the FZGM-vs-native comparison after the 2026-09 optimization work
(generalized warp-register fusion fwd+inverse, chunk-cooperative decompress
fusion, specialization-aware buffer coloring). See `docs/DESIGN.md` D42 and
`docs/adapters/fzgm.md`.

Experiments (each launched twice — `FZ_SPECIALIZE=off` then `auto`):
- `configs/experiments/specialization_vs_native_smoke.yaml`
- `configs/experiments/specialization_memory_smoke.yaml`
Driver: `scripts/run-specialization-smoke.sh <host-tag> [<env-script>]`
(`SPEC_FZGM_ONLY=1` for machines with no native CUDA compressors).

## Fleet

| Machine | GPU | Natives | env script | host-tag |
|---|---|---|---|---|
| JetStream2 | H100 | built | `scripts/env-jetstream2.sh` | `skyler-h100` |
| BigRed200 | A100 | built | `scripts/env-bigred200.sh` | `bigred200-a100` |
| Delta | H200 | **none** (CUDA-only refs) | `scripts/env-delta-h200.sh` | `delta-h200` |
| Delta | MI100 | **none** (cannot build) | `scripts/env-delta-mi100.sh` | `delta-mi100` |
| lair | (see env) | partial | `scripts/env-lair.sh` | `lair-<gpu>` |

## Required versions

- **compression_benchmarking**: commit with `specialization_vs_native_smoke.yaml`
  (this file's introduction) or later. `git log --oneline -1`.
- **FZGPUModules**: commit `6897f66` ("test: drop the fragile standalone-inverse
  assertion...") or later — the `ed94548`/`096a5ba`/`6897f66` chain. Must be
  **rebuilt** for the machine's arch — the specialization code is new.
  NOTE: FZGM history was rewritten on 2026-09-02 (Co-Authored-By trailer strip),
  so the old anchors `dc5c70c` / `4b0efdd` / `b57a333` / `8b1ea2a` are gone. A
  machine cloned before then must `git fetch origin && git reset --hard
  origin/main` (a plain `git pull` diverges). Same for compression_benchmarking.

## The setup-check + run prompt

Paste this into a Claude Code (or equivalent) session on the target machine, after
filling in `<HOST_TAG>` and `<ENV_SCRIPT>` from the fleet table. It verifies the
environment before spending an hour of GPU time, then runs the smoke and reports.

```
Set up and run the Pipeline Specialization smoke on this machine. Do NOT commit
anything. Fill HOST_TAG=<HOST_TAG> ENV_SCRIPT=<ENV_SCRIPT>. If a check fails, stop
and tell me what's wrong rather than trying to fix it.

1. REPOS  (history was rewritten 2026-09-02 — fetch + hard-reset, do NOT pull)
   - cd ~/compression_benchmarking && git fetch origin &&
     git status --porcelain
     If the working tree is dirty, show me the diff and stop. Otherwise
     git reset --hard origin/main && git log --oneline -1
     Confirm HEAD includes configs/experiments/specialization_vs_native_smoke.yaml
     (git log --oneline -- that path).
   - cd ~/FZGPUModules && git fetch origin
     If the working tree is dirty, stop. Otherwise git reset --hard origin/main &&
     git log --oneline -1
     Confirm HEAD is 6897f66 or a descendant (git merge-base --is-ancestor
     6897f66 origin/main; echo $?  -> 0). If not, stop.

2. FZGM BUILD (rebuild — the specialization code is new)
   - source ~/compression_benchmarking/<ENV_SCRIPT>  (sets FZGMOD_CLI, CUDA, venv)
   - Rebuild the benchmarking binary with the machine's usual preset
     (release / cuda-h200 / the HIP build): cmake --build <build-dir> -j
     where <build-dir> is whatever $FZGMOD_CLI points into
     (e.g. build_benchmarking/, build_sm90/, a cuda-h200 tree).
   - $FZGMOD_CLI --help | grep -q report-json   (must be a report-json build)
   - Sanity: FZ_SPECIALIZE=auto $FZGMOD_CLI -b \
       -c ~/compression_benchmarking/configs/pipelines/szp_composed.toml \
       -i <any CLDHGH.f32> -l 3600x1800 --runs 5 --report-json /tmp/s.json --compare <same file>
     then check /tmp/s.json: status ok, specialization.installed_group_count >= 1
     AND specialization.inverse_installed_group_count >= 1. If fusion does not
     install here, stop — the rest of the run is pointless.
   - Optional but recommended: cd ~/FZGPUModules && ctest --test-dir <test-build> -j1
     (58/58 expected on CUDA).

3. REFERENCE COMPRESSORS  (skip entirely if this machine is native-less — Delta)
   - cd ~/compression_benchmarking && python compressors/bootstrap.py status
     Every tool used by the smoke (cusz, cuszp2, cuszp3, fzgpu, pfpl) must be
     "clean" / present. If a tree is missing or drifted:
       source <ENV_SCRIPT> && python compressors/bootstrap.py         # build all
     or bootstrap.py <name> for one. See compressors/README.md (D41).
   - Confirm the env vars resolve: for v in CUSZ_CLI CUSZP2_CLI CUSZP3_CLI \
       FZGPU_CLI PFPL_BIN_DIR; do echo "$v=${!v}"; test -e "${!v}" && echo ok || echo MISSING; done

4. DATASETS
   - The smoke needs 4 fields: CESM-2D/CLDHGH, HURR/TC, NYX/temperature, HACC/vx.
   - source <ENV_SCRIPT> sets BENCHKIT_DATA_ROOT. Verify:
     python - <<'EOF'
     import os, yaml
     cfg = yaml.safe_load(os.path.expandvars(open('configs/datasets.yaml').read()))
     need = {'CESM-2D':'CLDHGH','HURR':'TC','NYX':'temperature','HACC':'vx'}
     for ds,fld in need.items():
         sp = cfg[ds]; f = sp['fields'][fld]
         p = os.path.join(sp['root'], f['path']); w = 8 if sp['dtype']=='f64' else 4
         n = 1
         for d in f['dims']: n *= d
         ok = os.path.exists(p) and os.path.getsize(p) == n*w
         print(('OK ' if ok else 'MISSING ')+f'{ds}/{fld}  {p}')
     EOF
   - If a field is missing: SDRBENCH_DATASETS="<KEY>" bash scripts/download-sdrbench.sh "$BENCHKIT_DATA_ROOT"

5. RUN
   - GPU must be otherwise idle during timed kernels. Check: nvidia-smi
     (no other compute procs). On a shared SLURM node get an --exclusive
     allocation first.
   - cd ~/compression_benchmarking
     SPEC_DATE_TAG=<YYYYMMDD> SPEC_FZGM_ONLY=<0 or 1> \
       bash scripts/run-specialization-smoke.sh <HOST_TAG> <ENV_SCRIPT>
     (SPEC_FZGM_ONLY=1 on Delta. Pick ONE date tag and keep it fixed across any
      resume — the session ids embed it.)
   - ~45-75 min. It is resumable: if it dies, re-run the identical command and it
     skips completed cells. To pause: kill the process, then
     bash scripts/unlock_clocks.sh.

6. REPORT BACK
   - Per experiment x {off,auto}: cell counts (N ok / N failed), and any failed
     cells with their error.
   - From the AUTO vs-native session: for every fzgm *_sp / pfpl / szp_composed
     row, fusion_installed_group_count and fusion_inverse_installed_group_count
     (must be > 0 on compress AND inverse for every fzgm *_sp / pfpl /
     szp_composed row, including cuszp3 — the TiledLorenzo inverse now fuses too,
     though its 3-D gain is small by roofline). For *_hp / cusz /
     fzgpu rows they must be 0.
   - off -> auto compress and decompress throughput ratio, and peak_device_mb
     ratio, for szp_composed / cuszp2_outlier_sp / pfpl on each of the 4 fields.
   - Any row where CR differs between the off and auto session for the SAME fzgm
     variant (that would be a bug — the _sp payloads are byte-identical A/Bs).
   - `benchkit verify` output. It is EXPECTED to exit non-zero on a smoke
     (native eb misses on cusz-HACC / cuszp3-outlier / fzgpu; timing-reliability
     on the 3-rep memory smoke; publication provenance). List what it flags so we
     can confirm it is only those.
```

## After the run — snapshot to a baseline

```bash
for arm in vsnative-staged vsnative-auto memory-staged memory-auto; do :; done
# For each of the 4 sessions:
mkdir -p results/baselines/<gpu>-<site>-<YYYYMMDD>-spec-<arm>
cp ~/benchkit-results/<session>/{runs.jsonl,provenance.json} results/baselines/<...>/
# write metadata.yaml — copy the H100 one from
# results/baselines/h100-jetstream2-20260901-spec-vsnative-staged/ and edit.
```

Then compare across machines:
```bash
python scripts/build_comparison_artifact.py \
  results/baselines/h100-jetstream2-20260901-spec-vsnative-auto \
  results/baselines/<other>-spec-vsnative-auto -o /tmp/compare.html
```
The script diffs CR/PSNR first as a determinism check — an FZGM `*_sp` cell whose
CR differs between two machines (or between off/auto on one) is a real bug, not a
timing artifact.
