"""Minimal reader for FZGM's .fzm archive format (v3.0/v3.1).

Only enough of the format to pull one named output port's bytes back out of an
archive — that is all benchkit needs, and it needs it for exactly one reason:
the back-end-isolation experiment (docs/adapters/nvcomp.md) has to feed FZGM's
lossless coders and nvCOMP's the *identical* Lorenzo quant codes, and the only
way to get those codes out of FZGM is to run a predictor-only pipeline and read
its output buffer. See scripts/extract_quant_codes.py.

Layout (FZPUModules include/fzm_format.h):

    [FZMHeaderCore]                  80 bytes (72 for v3.0, before checksums)
    [FZMStageInfo  x num_stages]     256 bytes each
    [FZMBufferEntry x num_buffers]   256 bytes each
    [payload]                        starts at header_size

Deliberately a reader only. benchkit never writes .fzm files — fzgmod-cli owns
that format, and a second writer would be a second thing to keep in sync with it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

FZM_MAGIC = 0x464D5A32          # "FZM2" little-endian
_CORE_V30 = "<IHHQQQIHH4Q"      # 72 bytes
_CORE_V31 = _CORE_V30 + "II"    # 80 bytes: v3.1 appends data/header checksums
_CORE_V30_SIZE = struct.calcsize(_CORE_V30)
_STAGE_INFO_SIZE = 256
_BUFFER_ENTRY_SIZE = 256
# 250 bytes of fields; the struct is padded to 256 for 8-byte alignment.
_BUFFER_FMT = "<HHBBH64sQQQQ128sI14s"

# DataType enum -> numpy-style token, for callers that want to interpret a port.
_DATA_TYPE = {0: "u8", 1: "u16", 2: "u32", 3: "u64",
              4: "i8", 5: "i16", 6: "i32", 7: "i64",
              8: "f32", 9: "f64"}


class FzmError(RuntimeError):
    pass


@dataclass
class FzmBuffer:
    """One output port's segment within the archive payload."""
    name: str
    data_type: str            # "u16", "f32", ... ("unknown(N)" if unrecognized)
    data_size: int            # bytes actually stored for this port
    uncompressed_size: int    # bytes after fully decompressing this port
    byte_offset: int          # offset within the payload, not within the file
    stage_type_id: int

    def read(self, path: str | Path, payload_start: int) -> bytes:
        with open(path, "rb") as fh:
            fh.seek(payload_start + self.byte_offset)
            blob = fh.read(self.data_size)
        if len(blob) != self.data_size:
            raise FzmError(
                f"{path}: port '{self.name}' claims {self.data_size} bytes at payload "
                f"offset {self.byte_offset} but only {len(blob)} were readable — "
                f"truncated archive?")
        return blob


@dataclass
class FzmArchive:
    path: Path
    version_major: int
    version_minor: int
    uncompressed_size: int
    compressed_size: int
    header_size: int
    num_stages: int
    buffers: list[FzmBuffer]

    @classmethod
    def read(cls, path: str | Path) -> "FzmArchive":
        p = Path(path)
        raw = p.read_bytes()
        if len(raw) < _CORE_V30_SIZE:
            raise FzmError(f"{p}: too short to be an .fzm archive ({len(raw)} bytes)")

        magic, version = struct.unpack_from("<IH", raw, 0)
        if magic != FZM_MAGIC:
            raise FzmError(
                f"{p}: bad magic 0x{magic:08X} (expected 0x{FZM_MAGIC:08X}). "
                f"Not an .fzm archive.")
        # Pre-split files stored a bare integer (e.g. 3) rather than major<<8|minor.
        major = version if version <= 0xFF else version >> 8
        minor = 0 if version <= 0xFF else version & 0xFF
        if major != 3:
            raise FzmError(
                f"{p}: FZM major version {major} is not supported by this reader "
                f"(expects 3.x). Regenerate with a current fzgmod-cli.")

        core_fmt = _CORE_V31 if minor >= 1 else _CORE_V30
        core_size = struct.calcsize(core_fmt)
        (_, _, num_buffers, uncomp, comp, header_size,
         num_stages, _num_sources, _flags, *_rest) = struct.unpack_from(core_fmt, raw, 0)
        buf_base = core_size + num_stages * _STAGE_INFO_SIZE
        want = buf_base + num_buffers * _BUFFER_ENTRY_SIZE
        if header_size < want or len(raw) < header_size:
            raise FzmError(
                f"{p}: header claims {header_size} bytes for {num_stages} stages and "
                f"{num_buffers} buffers, which needs at least {want} (file is "
                f"{len(raw)} bytes) — truncated or a format the reader predates.")

        buffers = []
        for i in range(num_buffers):
            off = buf_base + i * _BUFFER_ENTRY_SIZE
            (stype, _sver, dtype, _pidx, _dagid, name,
             data_size, _alloc, uncompressed, byte_offset,
             _cfg, _cfgsz, _res) = struct.unpack_from(_BUFFER_FMT, raw, off)
            buffers.append(FzmBuffer(
                name=name.split(b"\x00", 1)[0].decode("utf-8", "replace"),
                data_type=_DATA_TYPE.get(dtype, f"unknown({dtype})"),
                data_size=data_size,
                uncompressed_size=uncompressed,
                byte_offset=byte_offset,
                stage_type_id=stype,
            ))

        return cls(path=p, version_major=major, version_minor=minor,
                   uncompressed_size=uncomp, compressed_size=comp,
                   header_size=header_size, num_stages=num_stages, buffers=buffers)

    def port(self, name: str) -> FzmBuffer:
        for b in self.buffers:
            if b.name == name:
                return b
        raise FzmError(
            f"{self.path}: no output port named '{name}' "
            f"(have: {[b.name for b in self.buffers]})")

    def read_port(self, name: str) -> bytes:
        return self.port(name).read(self.path, self.header_size)
