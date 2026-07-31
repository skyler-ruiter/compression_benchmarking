#!/usr/bin/env python3
"""Build the JSON payload for the four-GPU throughput artifact.

Two questions, two different sets of baselines, and they must not be conflated:

  1. "How does FZGM compare to the native compressors?"  Only A100 and H100 can
     answer this -- Delta does not build the natives, so the H200 and MI100
     baselines are FZGM-only. Answered per GPU and NEVER pooled: the CR ratios
     reproduce across architectures to three decimals, the throughput ratios do
     not and are not expected to.

  2. "How does FZGM itself move across four GPUs?"  Answered over the 4,860 FZGM
     cells common to all four baselines, restricted to those passing the validity
     gate on ALL four, so a cell that is unusable anywhere is dropped everywhere
     rather than quietly changing the denominator per GPU.

Peak-bandwidth figures are VENDOR SPEC, not measured. They are used only to
normalise, and a normalised number inherits the error of its denominator -- treat
"% of peak" as an ordering, not as an achieved-bandwidth measurement.

Usage:
    python scripts/build_throughput_artifact.py -o thru4.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from benchkit import validity  # noqa: E402

BASE = ROOT / 'results' / 'baselines'

# baseline_id, peak HBM GB/s (vendor spec), whether clocks were pinned
GPUS = {
    'A100':  dict(dir='a100-bigred200-fullcorpus-slurm7828800', peak=1555.0, locked=False,
                  full='A100-SXM4-40GB', site='IU BigRed200', natives=True),
    'H100':  dict(dir='h100-jetstream2-20260729-fullcorpus', peak=3350.0, locked=True,
                  full='H100 80GB HBM3', site='JetStream2', natives=True),
    'H200':  dict(dir='h200-delta-fullcorpus-20559887', peak=4800.0, locked=False,
                  full='H200 SXM', site='NCSA Delta', natives=False),
    'MI100': dict(dir='mi100-delta-fullcorpus-20568712', peak=1229.0, locked=False,
                  full='Instinct MI100 (gfx908)', site='NCSA Delta', natives=False),
}
REF = 'A100'   # the only GPU present in every generation of this corpus

LABEL = {
    'cusz': 'cuSZ', 'cuszhi_tp': 'cuSZ-Hi (tp)', 'cuszhi_cr': 'cuSZ-Hi (cr)',
    'cuszp2_plain': 'cuSZp2 (plain)', 'cuszp2_outlier': 'cuSZp2 (outlier)',
    'cuszp3_plain': 'cuSZp3 (plain)', 'cuszp3_outlier': 'cuSZp3 (outlier)',
    'fzgpu': 'FZ-GPU', 'pfpl': 'PFPL',
}
VARIANTS = list(LABEL)
EBS = ['0.01', '0.001', '0.0001']

# Why a (variant, dataset) pair has no ratio. Kept distinct because they mean
# opposite things about FZGM: `native_crashed` is a point in FZGM's favour,
# `fzgm_absent` is a point against, and collapsing them into one "no data" grey
# erases exactly the robustness result the table is supposed to show.
REASONS = {
    'fzgm_absent':    'FZGM pipeline does not run on this data (structural)',
    'native_absent':  'no native reference for this pairing',
    'native_crashed': 'every native cell crashed - nothing to compare against',
    'gated':          'both sides ran, but no cell passed the validity gate',
}

SIZE_BINS = [(0, 4), (4, 16), (16, 64), (64, 256), (256, float('inf'))]


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def load(gpu):
    p = BASE / GPUS[gpu]['dir'] / 'runs.jsonl'
    return validity.annotate([json.loads(l) for l in p.read_text().splitlines() if l.strip()])


def pairkey(r):
    return (r['dataset'], r['field'], r['variant'], r.get('error_mode'), r.get('error_bound'))


def build_native(rows):
    """Per variant / dataset / eb: geomean throughput of both sides, or a reason code."""
    natives = {pairkey(r): r for r in rows if r['compressor'] != 'fzgm' and 'error_mode' in r}
    fz = [r for r in rows if r['compressor'] == 'fzgm']
    out = {}
    for var in VARIANTS:
        out[var] = {}
        fv = [r for r in fz if r['variant'] == var]
        for ds in sorted({r['dataset'] for r in rows}):
            per_eb = {}
            for eb in EBS:
                ebf = float(eb)
                mine = [r for r in fv if r['dataset'] == ds and r.get('error_bound') == ebf]
                theirs = [r for k, r in natives.items()
                          if k[0] == ds and k[2] == var and k[4] == ebf]
                if not mine:
                    per_eb[eb] = {'reason': 'fzgm_absent'}
                    continue
                if not theirs:
                    per_eb[eb] = {'reason': 'native_absent'}
                    continue
                if not any(r.get('status') == 'ok' for r in theirs):
                    per_eb[eb] = {'reason': 'native_crashed',
                                  'crashed': len(theirs), 'fzgm_ok': sum(
                                      1 for r in mine if r.get('status') == 'ok')}
                    continue
                pairs = [(r, natives[pairkey(r)]) for r in mine if pairkey(r) in natives]
                pairs = [(a, b) for a, b in pairs
                         if validity.is_valid(a) and validity.is_valid(b)]
                if not pairs:
                    per_eb[eb] = {'reason': 'gated'}
                    continue
                per_eb[eb] = {
                    'n': len(pairs),
                    'fc': gmean([a['compress_throughput_gbs'] for a, _ in pairs]),
                    'nc': gmean([b['compress_throughput_gbs'] for _, b in pairs]),
                    'fd': gmean([a['decompress_throughput_gbs'] for a, _ in pairs]),
                    'nd': gmean([b['decompress_throughput_gbs'] for _, b in pairs]),
                }
            out[var][ds] = per_eb
    return out


def build_overall(rows):
    natives = {pairkey(r): r for r in rows if r['compressor'] != 'fzgm' and 'error_mode' in r}
    out = {}
    for var in VARIANTS:
        pairs = []
        for r in rows:
            if r['compressor'] != 'fzgm' or r.get('variant') != var:
                continue
            n = natives.get(pairkey(r))
            if n and validity.is_valid(r) and validity.is_valid(n):
                pairs.append((r, n))
        if pairs:
            out[var] = {
                'n': len(pairs),
                'c': gmean([a['compress_throughput_gbs'] / b['compress_throughput_gbs']
                            for a, b in pairs]),
                'd': gmean([a['decompress_throughput_gbs'] / b['decompress_throughput_gbs']
                            for a, b in pairs]),
                'cr': gmean([a['cr'] / b['cr'] for a, b in pairs]),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default='-')
    args = ap.parse_args()

    R = {g: load(g) for g in GPUS}

    # ---- corpus cards -------------------------------------------------------
    seen = {}
    for r in R[REF]:
        # Failed rows carry no geometry, so the cards are built from ok rows only.
        if r.get('status') != 'ok':
            continue
        d = seen.setdefault(r['dataset'], {'fields': set(), 'mb': [], 'nd': len(r['dims']),
                                           'dtype': r['dtype']})
        d['fields'].add(r['field'])
        d['mb'].append(r['original_bytes'] / 1e6)
    datasets = [{'key': k, 'nd': v['nd'], 'dtype': v['dtype'],
                 'mb': round(gmean(v['mb']), 1), 'nfields': len(v['fields'])}
                for k, v in sorted(seen.items())]

    # ---- FZGM common-cell universe -----------------------------------------
    F = {g: {r['cell_key']: r for r in R[g] if r['compressor'] == 'fzgm'} for g in GPUS}
    common = set.intersection(*[set(F[g]) for g in GPUS])
    # Gate on ALL four so the denominator is one fixed cell set, not a per-GPU one.
    vc = sorted(k for k in common if all(validity.is_valid(F[g][k]) for g in GPUS))
    varof = lambda k: k.split('|')[1]      # noqa: E731
    dsof = lambda k: k.split('|')[3]       # noqa: E731

    def agg(ks):
        e = {}
        for g in GPUS:
            c = gmean([F[g][k]['compress_throughput_gbs'] for k in ks])
            d = gmean([F[g][k]['decompress_throughput_gbs'] for k in ks])
            e[g] = {
                'c': c, 'd': d,
                'sc': gmean([F[g][k]['compress_throughput_gbs']
                             / F[REF][k]['compress_throughput_gbs'] for k in ks]),
                'sd': gmean([F[g][k]['decompress_throughput_gbs']
                             / F[REF][k]['decompress_throughput_gbs'] for k in ks]),
                'pc': 100 * c / GPUS[g]['peak'], 'pd': 100 * d / GPUS[g]['peak'],
            }
        e['n'] = len(ks)
        return e

    byvar = collections.defaultdict(list)
    byds = collections.defaultdict(list)
    for k in vc:
        byvar[varof(k)].append(k)
        byds[dsof(k)].append(k)

    xgpu = {
        'all': agg(vc),
        'byVariant': {v: agg(ks) for v, ks in byvar.items()},
        'byDataset': {d: agg(ks) for d, ks in byds.items()},
        # variant x dataset, for the cross-machine heat grid
        'grid': {v: {d: agg([k for k in ks if dsof(k) == d])
                     for d in {dsof(k) for k in ks}} for v, ks in byvar.items()},
    }

    # ---- size buckets -------------------------------------------------------
    size = []
    for lo, hi in SIZE_BINS:
        ks = [k for k in vc if lo <= F[REF][k]['original_bytes'] / 1e6 < hi]
        if not ks:
            continue
        e = agg(ks)
        e['lo'], e['hi'] = lo, (None if hi == float('inf') else hi)
        size.append(e)

    # ---- CR agreement across all four (the architecture-independence claim) --
    okc = [k for k in common if all(F[g][k].get('status') == 'ok' for g in GPUS)]
    ident, byv, worst = 0, collections.Counter(), (0.0, None)
    for k in okc:
        crs = [F[g][k]['cr'] for g in GPUS]
        if len(set(crs)) == 1:
            ident += 1
        else:
            byv[varof(k)] += 1
            dev = (max(crs) - min(crs)) / min(crs)
            if dev > worst[0]:
                worst = (dev, k)
    cragree = {'n': len(okc), 'identical': ident, 'byVariant': dict(byv),
               'worst': worst[0], 'worstCell': worst[1]}

    # ---- MI100 failures -----------------------------------------------------
    fails = [r for r in R['MI100'] if r.get('status') != 'ok']
    mi_keys = {r['cell_key'] for r in fails}
    mi = {
        'n': len(fails),
        'byVariant': dict(collections.Counter(r.get('variant') for r in fails).most_common()),
        'byDataset': dict(collections.Counter(r.get('dataset') for r in fails).most_common()),
        # The decisive fact: every one of these succeeds on all three NVIDIA GPUs, so it
        # is a device limit and not a bad cell.
        'okElsewhere': {g: sum(1 for k in mi_keys if F[g].get(k, {}).get('status') == 'ok')
                        for g in GPUS},
        'msg': (fails[0].get('error_message') or '') if fails else '',
    }

    out = {
        'gpus': {g: {k: v for k, v in m.items()} for g, m in GPUS.items()},
        'ref': REF,
        'datasets': datasets,
        'variants': [{'key': k, 'label': LABEL[k]} for k in VARIANTS],
        'ebs': EBS,
        'reasons': REASONS,
        'native': {g: build_native(R[g]) for g in GPUS if GPUS[g]['natives']},
        'overall': {g: build_overall(R[g]) for g in GPUS if GPUS[g]['natives']},
        'xgpu': xgpu,
        'size': size,
        'crAgreement': cragree,
        'mi100': mi,
    }

    txt = json.dumps(out)
    if args.out == '-':
        print(txt)
    else:
        Path(args.out).write_text(txt)
        print(f'wrote {args.out}  ({len(txt)/1024:.0f} KB): '
              f'{len(vc)} FZGM cells valid on all four GPUs', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
