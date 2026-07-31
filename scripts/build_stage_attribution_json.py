#!/usr/bin/env python3
"""Roll a stage-attribution session into the JSON an artifact page consumes.

Reads a session produced by configs/experiments/fzgm_stage_attribution.yaml and emits,
per FZGM pipeline, the share of device time each stage accounts for -- separately for
compress and decompress, and separately per dataset, so "is the split consistent across
data?" is answerable by eye.

Requires an FZGPUModules build from 2026-07-29 or later; before that fix, `stages[]`
carried decompress entries only and the compress half comes back empty. The script says
so loudly rather than emitting a half-populated file.

Usage:
    python scripts/build_stage_attribution_json.py <session-dir-or-runs.jsonl> [-o out.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchkit import validity  # noqa: E402

LABEL = {
    'cusz': 'cuSZ', 'cuszhi_tp': 'cuSZ-Hi (tp)', 'cuszhi_cr': 'cuSZ-Hi (cr)',
    'cuszp2_plain': 'cuSZp2 (plain)', 'cuszp2_outlier': 'cuSZp2 (outlier)',
    'cuszp3_plain': 'cuSZp3 (plain)', 'cuszp3_outlier': 'cuSZp3 (outlier)',
    'fzgpu': 'FZ-GPU', 'pfpl': 'PFPL',
}
ORDER = list(LABEL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('-o', '--out', default='-')
    args = ap.parse_args()

    p = Path(args.session)
    runs = p / 'runs.jsonl' if p.is_dir() else p
    rows = [json.loads(l) for l in runs.read_text().splitlines() if l.strip()]
    rows = [r for r in validity.annotate(rows) if validity.is_valid(r)]
    rows = [r for r in rows if r.get('compressor') == 'fzgm' and r.get('stages')]
    if not rows:
        print('error: no valid FZGM rows with stages[]', file=sys.stderr)
        return 1

    phases = collections.Counter(s['phase'] for r in rows for s in r['stages'])
    if not phases.get('compress'):
        print('error: no COMPRESS-phase stage entries in this session. The FZGM build '
              'predates the 2026-07-29 profiling fix (setMemoryStrategy() silently '
              'disabled profiling on the compress DAG). Rebuild and re-run.',
              file=sys.stderr)
        return 1

    out = {'pipelines': [], 'phase_counts': dict(phases)}
    for var in ORDER:
        sub = [r for r in rows if r['variant'] == var]
        if not sub:
            continue
        # Stage order is taken from the DECOMPRESS trace and reused for compress so a
        # stage keeps its colour and its slot in both bars. Compress order is the DAG
        # forward order, which is *usually* the reverse of decompress -- but not on
        # branching DAGs with repeated stages, so it is not derived by reversing.
        seen: list[str] = []
        for r in sub:
            for s in r['stages']:
                if s['name'] not in seen:
                    seen.append(s['name'])
        entry = {'key': var, 'label': LABEL[var], 'stages': seen, 'rows': []}
        by = collections.defaultdict(lambda: collections.defaultdict(float))
        tot = collections.defaultdict(float)
        nfield = collections.defaultdict(set)
        for r in sub:
            for s in r['stages']:
                k = (r['dataset'], s['phase'])
                by[k][s['name']] += s['device_ms']
                tot[k] += s['device_ms']
                nfield[k].add(r['field'])
        for (ds, phase), t in sorted(tot.items()):
            if t <= 0:
                continue
            n = len(nfield[(ds, phase)])
            entry['rows'].append({
                'dataset': ds, 'phase': phase,
                'ms': t / n,
                'nfields': n,
                'split': {k: 100.0 * v / t for k, v in by[(ds, phase)].items()},
            })
        # Consistency: does the same stage dominate everywhere, and how much does its
        # share move? Shares are averaged only over datasets where the stage is actually
        # PRESENT -- some pipelines swap a stage by geometry (cuszp3 runs TiledLorenzo on
        # 2-D/3-D and plain Lorenzo on its 1-D preset), and scoring an absent stage as 0%
        # would report that swap as wild inconsistency instead of as a different pipeline.
        for phase in ('compress', 'decompress'):
            rs = [r for r in entry['rows'] if r['phase'] == phase]
            if not rs:
                continue
            tops = collections.Counter()
            for r in rs:
                tops[max(r['split'], key=r['split'].get)] += 1
            dom, cnt = tops.most_common(1)[0]
            present = [r['split'][dom] for r in rs if dom in r['split']]
            entry.setdefault('consistency', {})[phase] = {
                'dominant': dom, 'datasets': len(rs), 'dominant_in': cnt,
                'present_in': len(present),
                'share_min': min(present), 'share_max': max(present),
            }
        entry['note'] = ''
        out['pipelines'].append(entry)

    txt = json.dumps(out)
    if args.out == '-':
        print(txt)
    else:
        Path(args.out).write_text(txt)
        print(f'wrote {args.out}: {len(out["pipelines"])} pipelines', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
