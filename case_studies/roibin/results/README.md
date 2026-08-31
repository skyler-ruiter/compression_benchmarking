# Raw sweep output

One row per (frame x arm). `status: ok` alone is not a usable row — apply the
per-region gate (`roi_bound_ok`, and `bg_bound_ok` where it is not `n/a`).

| file | what it is |
|---|---|
| `exafel_eb10_100.csv` | **authoritative** EXAFEL headline run (copy of the re-run) |
| `exafel_eb10_100_rerun.csv` | the same, under its original name |
| `exafel_eb10_100_CONTAMINATED.csv` | first EXAFEL run. CR/PSNR/bounds are valid and identical; its **throughput column is not** — another session's sweep shared the GPU for ~4 min. Kept as evidence, not for quotation. Medians agreed within 1.2%. |
| `cxidb21_eb10_100.csv` | CXIDB 21 headline run |
| `exafel_eb{10_50,10_1000,5_50,1_10}.csv` | bound-sensitivity subsets (40 frames) |
| `exafel_eb1_10.csv` | eb_roi=1.0; cuSZ-Hi run with the **full tuning grid unpinned** |
| `cuszhi_failure_envelope.csv` | cuSZ-Hi success/failure per frame per bound |
| `volume_b2_*.json` | whole-volume (130-frame) run |
