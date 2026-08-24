# Mechanical session verification

`benchkit verify` is the publication-completion gate for a native H1–H3 session:

```bash
python -m benchkit verify "$BENCHKIT_RESULTS_ROOT/<session-id>"
```

It writes `verification.json` atomically and returns 0 only when every check passes.
A benchmark process exiting 0, a plausible row count, or every row having `status: ok`
does not make a session publication-grade.

The verifier checks:

- every canonical and raw attempt row against the versioned result schema;
- every immutable provenance manifest and every row-to-provenance join;
- archived experiment, dataset, checksum-lock, site, resolved-dataset, harness-patch,
  and rendered-pipeline SHA-256 values;
- unique run IDs, collision-free identity payloads, and unique canonical logical cells;
- that sharded sessions have a current `runs.jsonl` matching merge precedence;
- exact expected matrix coverage reconstructed from the archived experiment and dataset
  YAML, including dataset/run scoping and error-bound expansion;
- H3 dataset-integrity joins and complete tool build provenance;
- failures, validity-gating exclusions, and unreliable timing against explicit policy.

The JSON report contains every failure detail rather than only console examples. The
console limits long lists to keep job logs readable.

## Explicit acceptance policy

Exceptions live in the experiment YAML so they are archived with the run. Each policy
entry needs a non-empty selector and written reason; a blanket empty selector is
rejected. Selectors may use `compressor`, `variant`, `pipeline`, `dataset`, `field`,
`error_mode`, `error_bound`, `error_type`, and `fail_phase`.

```yaml
verification:
  accepted_failures:
    - match:
        compressor: fzgm
        error_type: RuntimeError
        fail_phase: compress
        dataset: HACC
      reason: >-
        Quantizer representable-spacing refusal at the declared tight bound; retained
        as an expected capability boundary.

  accepted_exclusions:
    - codes: [degenerate_field]
      match: {dataset: CESM-2D, field: SFCLDICE}
      reason: Field is constant in this locked SDRBench snapshot.

  accepted_unreliable_timing:
    - match: {compressor: sz3, dataset: NWCHEM}
      reason: CPU baseline retained for quality only; timing is not used in claims.
```

Accepted exclusions are limited to gating codes: `degenerate_field`,
`eb_violated_severe`, and `psnr_nonfinite`. Measured compression expansion is retained
by the validity layer and requires no waiver. Lossless exact reconstruction likewise
does not fail verification.

Policy is not a substitute for investigation. Reasons should state why the exception is
scientifically expected and how it affects claims. H5 bundles the verification report
alongside its archived policy and source rows.
