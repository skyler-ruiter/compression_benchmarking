# Reference-tool acquisition and build status

The repository does **not** vendor reference-compressor upstream source. It **does**
(since D41) carry a pinned, reproducible recipe for rebuilding every one of them:
`compressors/manifest.toml` (upstream repo + 40-char commit + submodules + patch list
+ build recipe + CLI path), `compressors/patches/` (every local source modification as
a `git apply`-able diff), `compressors/build/` (path-free build scripts), and
`compressors/bootstrap.py` (clone → checkout pin → apply patches → build). See
`compressors/README.md`. The 3 cuSZp H100-optimization forks, which have no upstream,
are the one thing vendored in full (`compressors/vendor/`).

So "rebuild the reference set on a new machine" is now `source scripts/env-<machine>.sh
&& python compressors/bootstrap.py`. What this still does **not** do: grant an artifact
evaluator access to third-party source or licenses, and it is not Git submodules.
Earlier design text described pinned submodules as the target; the manifest is the
form that landed instead.

| Tool family | Current source/build path |
|---|---|
| FZGM | External FZGPUModules tree and installed `fzgmod-cli`; `docs/adapters/fzgm.md`. |
| cuSZ, cuSZp2/3, cuSZ-Hi, FZ-GPU, PFPL | External source builds; exact patches and build commands are recorded in their adapter documents. |
| FSZ | External FSZ source build plus tracked `tools/fsz_hosttime/fsz_hosttime.cu` and `scripts/build-fsz-hosttime.sh`. |
| SZ3, zfp, MGARD-X, SPERR | External source builds; adapter documents give the working configuration and binary path. |
| nvCOMP | NVIDIA SDK selected by `NVCOMP_ROOT`; the repository builds its own CLI with `scripts/build-nvcomp-cli.sh`. |
| MANS, lsCOMP | Adapter scaffolding/coder-only support; not general error-bounded reference baselines. Build wrappers are tracked where usable. |

Publication runs must declare tool version, source commit, build flags, and local patch
identity. H3 also hashes executable files; H4 refuses incomplete build provenance. H5
ships the recorded identities and available build recipes, but it cannot grant an
artifact evaluator access to third-party source code or licenses.

Whether to vendor upstream source in full (beyond the cuSZp forks) is still open and
should be decided tool by tool after checking upstream license, repository size, and
whether the local correctness/timing patches have an upstreamable commit. The manifest
makes that a smaller question than it was: the patches and recipes are already captured,
so vendoring would only be swapping a pinned clone for an in-repo copy.
