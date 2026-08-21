# lsCOMP adapter

lsCOMP operates on unsigned integers. The adapter uses the same documented
quarter-bound-bin uniform float quantizer as MANS, then runs lsCOMP with
`-b 1 1 1 1 -p 1`, disabling lsCOMP's own quantization and pooling so the
integer payload is lossless. u16/u32 is selected per field and bound. The
self-describing adapter container stores the quantizer and shape metadata; all
header bytes count toward CR.

The upstream example executable can only decode the stream it just compressed
inside the same process. `tools/lscomp_decode/lscomp_decode` links the native
library to provide a true standalone decode path, so a retained adapter artifact
does not depend on the original quantized input.

Build and configure:

```bash
source ~/load-env
bash scripts/build-lscomp-decode.sh
export LSCOMP_CLI=$HOME/compressors/lsCOMP/build/lsCOMP_uint32
export LSCOMP_UINT16_CLI=$HOME/compressors/lsCOMP/build/lsCOMP_uint16
export LSCOMP_DECODE_CLI=$PWD/tools/lscomp_decode/lscomp_decode
```

Field dimensions are reversed from benchkit's fast-to-slow convention and
padded to the CLI's required `(slow, middle, fast)` triple.

## Timing

The primary timing covers the implemented transform:

- compression: CPU quantization/q-file write wall time plus lsCOMP's native
  CUDA-event compression time;
- decompression: lsCOMP's native CUDA-event decompression time plus CPU
  dequantization/output-write wall time.

This mixed timing is useful for the wrapper as it exists today but is not a
pure GPU throughput measurement and must not be ranked directly with FSZ,
cuSZ, or FZGM device time. The adapter provenance records this explicitly.

The adapter is useful as an independent CR/quality/failure baseline even though
there is no corresponding FZGM pipeline.
