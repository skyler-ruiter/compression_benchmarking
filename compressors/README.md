# `compressors/` — portable third-party compressor tree

The ~20 reference compressors benchkit compares against are each a hand-built
source tree with machine-specific toolchain flags and local patches. Copying
`~/compressors/` between machines is ~3 GB and the build artifacts don't
transfer anyway. This directory is the portable representation instead:

| file | what it is |
|---|---|
| `manifest.toml` | every compressor: upstream repo + **pinned 40-char commit**, submodule flag, patch list, build recipe, CLI path |
| `patches/<name>/*.patch` | local source modifications (CLI timing harness, file I/O, build-config, bug fixes) as `git apply`-able diffs against the pinned commit |
| `build/<name>.sh` | the build recipe, parameterised by `$CUDA_ARCH` / `$CC` / `$CUDA_ROOT` / `$JOBS` (see `build/_common.sh`) — no machine-specific paths |
| `vendor/<name>/` | full source for the forks with **no usable upstream** (the cuSZp H100-optimization variants) |
| `bootstrap.py` | clone → checkout pin → submodules → apply patches → build. stdlib only (py ≥ 3.11). |

Total here is ~3 MB. The actual checkouts still land in `~/compressors/`
(override with `COMPRESSORS_ROOT`) at the **same paths as before**, so
`scripts/env-<machine>.sh` is unchanged.

## Moving to a new machine

```bash
source scripts/env-<machine>.sh     # toolchain, CUDA, and — importantly — CUDA_ARCH
python compressors/bootstrap.py     # all of them; ~30–60 min, MGARD dominates

# or a subset / one at a time
python compressors/bootstrap.py all cusz sz3 zfp
python compressors/bootstrap.py fetch cuszp2      # clone + patch only, skip build
python compressors/bootstrap.py build cuszp2      # rebuild only
python compressors/bootstrap.py status            # disk vs manifest, per compressor
```

`env-<machine>.sh` must export `CUDA_ARCH` (e.g. `80` A100, `90` H100). Without
it everything targets `90` (`manifest.toml [toolchain].default_cuda_arch`).
`CC`/`CXX`/`CUDA_ROOT`/`JOBS` are optional (see `build/_common.sh`).

## When you change a checkout locally

Edit the tree in `~/compressors/<X>` as usual, confirm the build, then:

```bash
python compressors/bootstrap.py capture <name>    # re-writes patches/<name>/0001-*.patch from `git diff`
git -C compression_benchmarking add compressors/patches/<name>
```

For a **new** compressor: add a `[compressor.<name>]` block to `manifest.toml`,
write `build/<name>.sh`, `bootstrap.py fetch <name>`, patch/build, `capture`,
then add the adapter (`benchkit/adapters/__init__.py`) and the
`scripts/env-*.sh` export.

To **bump a pin**: change `commit` (full SHA), `bootstrap.py fetch <name>`,
re-apply/refresh the patch (`--3way` handles drift; `capture` to re-freeze),
rebuild, re-run the adapter smoke test.

## Notes / gotchas

- **cuSZp**: even the "base" `cuSZp-V2.0.1` / `cuSZp-V3.0.0` dirs carry a local
  timing harness (a ~fixed per-launch dispatch tax on GPU-passthrough VMs makes
  single-shot `cudaEventElapsedTime` unusable — the patch times an averaged
  batch). Patch is large but applies cleanly to the pinned tags.
- **`_optimized` / `_split` variants** (`vendor/`): warp-cooperative kernel
  rewrites from the FZGM cuSZp H100 work. Too divergent from upstream for a
  patch series — snapshot only.
- **MGARD**: `build/mgard.sh` clones and installs its own nvcomp 2.2.0 / zstd
  1.5.0 / protobuf 3.19.4 into `install-cuda-<arch>/`. That `lib/` must be on
  `LD_LIBRARY_PATH` at runtime (the env scripts do this). ~15 min build.
- **cuSZ**: gcc ≥ 13 ICEs on `codec/hf/src/hf_hl.cc`; the fix (`const RMerge`
  → `const auto`) is in `patches/cusz/`, so a modern default compiler is fine
  now — no more `gcc-native/12.3` requirement for cuSZ specifically.
- **cuSZ-Hi**: `third_party/googletest` submodule is left uninitialised
  (`submodules = false`) — examples are off, it isn't needed.
- **nvcomp**: not a build. `build/nvcomp.sh` downloads the pinned redist
  tarball. `nvcomp_cli` (the thing that reports device time) is built
  separately inside benchkit — `scripts/build-nvcomp-cli.sh`.
- **SZp**: pinned but has no benchkit adapter and no recipe here — it's in
  `~/compressors/` because FZGPUModules consumes it. Build per the FZGM docs.
- **ROIBIN** used to live in `~/compressors/ROIBIN` but is a case study, not a
  compressor. Its source is now `case_studies/roibin/`
  (`roibin-prior-history.gitbundle` preserves its prior standalone git log).
