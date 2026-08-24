# Reference-tool acquisition and build status

The repository does **not** currently vendor reference compressors as Git submodules.
They are externally acquired source trees or SDKs selected through environment variables
and per-run `cli_path`. Earlier design text described pinned submodules as the target;
claiming that target as current behavior would overstate AD/AE self-containment.

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

A future vendoring decision should be done tool by tool after checking upstream license,
repository size, submodule stability, and whether local correctness/timing patches have
an upstreamable commit. Until then, external acquisition is an explicit limitation—not
a pinned-submodule claim.
