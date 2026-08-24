# Helper tools

Only source and build recipes are portable repository artifacts. In particular,
`fsz_hosttime` and `lscomp_decode` link against machine-local CUDA/tool builds and must
not be committed as ELF binaries.

Build them on the target machine with:

```bash
scripts/build-fsz-hosttime.sh
scripts/build-lscomp-decode.sh
```

Their exact executable SHA-256, source commit, build flags, and local patch identity
belong in H3 session provenance. An executable copied from another machine is neither a
reproducible build nor trustworthy publication provenance.
