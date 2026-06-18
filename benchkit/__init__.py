"""benchkit — GPU error-bounded lossy compressor benchmarking harness.

See docs/DESIGN.md for the architecture. M1 scope: the core loop on a single
compressor (FZGM) — config -> runner -> adapter -> harness-owned metrics ->
append-only JSONL -> summary table.
"""

__version__ = "0.1.0"
