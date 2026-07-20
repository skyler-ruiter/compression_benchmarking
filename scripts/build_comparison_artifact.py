#!/usr/bin/env python3
"""Build an interactive HTML comparison artifact from two result baselines.

Usage:
    python scripts/build_comparison_artifact.py <baseline_dir_a> <baseline_dir_b> -o out.html

Each baseline_dir must contain runs.jsonl (harness schema) and, ideally,
metadata.yaml (see results/baselines/README.md for the schema). Before
charting throughput, this script cross-checks CR/PSNR agreement between the
two baselines for every matched cell (same compressor/variant/side/dataset/
eb) — they should be near-identical for deterministic algorithms run on the
same input, and a real disagreement is worth surfacing rather than silently
charting through, which is exactly how the cuSZp2 excl_sum bug (DESIGN.md
D23) was found: two baselines disagreed on cells that should have matched.

The output is a single self-contained HTML file (inline CSS/JS, no external
requests) suitable for publishing as a Claude artifact or opening directly
in a browser.
"""
import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_baseline(path: Path):
    path = Path(path)
    runs_path = path / "runs.jsonl"
    if not runs_path.exists():
        raise SystemExit(f"no runs.jsonl in {path}")
    rows = [json.loads(l) for l in runs_path.read_text().splitlines() if l.strip()]

    meta = {}
    meta_path = path / "metadata.yaml"
    if meta_path.exists() and yaml is not None:
        meta = yaml.safe_load(meta_path.read_text()) or {}

    prov = {}
    prov_path = path / "provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())

    gpu_name = meta.get("gpu") or (prov.get("gpu") or {}).get("name") or path.name
    label = f"{gpu_name}"
    return {"rows": rows, "meta": meta, "provenance": prov, "label": label, "dir": path.name}


def index_rows(rows):
    """key -> {side: row}. key = (variant, dataset, str(eb)); side = native|fzgm."""
    idx = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        variant = r.get("variant")
        dataset = r.get("dataset")
        eb = r.get("error_bound")
        if variant is None or dataset is None or eb is None:
            continue
        side = "fzgm" if r.get("compressor") == "fzgm" else "native"
        key = (variant, dataset, str(eb))
        idx.setdefault(key, {})[side] = {
            "status": "ok",
            "cr": r.get("cr"),
            "psnr": r.get("psnr"),
            "cgbs": r.get("compress_throughput_gbs"),
            "dgbs": r.get("decompress_throughput_gbs"),
            "eb_ok": r.get("eb_satisfied"),
            "tok": r.get("timing_reliable"),
        }
    return idx


DEFAULT_LABELS = {
    "cusz": "cuSZ", "cuszhi_tp": "cuSZ-Hi (tp)", "cuszhi_cr": "cuSZ-Hi (cr)",
    "cuszp2_plain": "cuSZp2 (plain)", "cuszp2_outlier": "cuSZp2 (outlier)",
    "cuszp3_plain": "cuSZp3 (plain)", "cuszp3_outlier": "cuSZp3 (outlier)",
    "fzgpu": "FZ-GPU", "pfpl": "PFPL",
}


def build_data(base_a, base_b, anomaly_psnr_db=5.0, anomaly_cr_pct=0.2):
    idx_a = index_rows(base_a["rows"])
    idx_b = index_rows(base_b["rows"])
    all_keys = set(idx_a) | set(idx_b)

    variants = sorted({k[0] for k in all_keys})
    datasets = sorted({k[1] for k in all_keys})
    ebs = sorted({float(k[2]) for k in all_keys}, reverse=True)

    data = {}
    anomalies = []
    max_cr_reldiff = 0.0
    max_psnr_absdiff = 0.0
    for variant in variants:
        data[variant] = {}
        for ds in datasets:
            data[variant][ds] = {}
            for eb in ebs:
                key = (variant, ds, str(eb))
                ra = idx_a.get(key, {})
                rb = idx_b.get(key, {})
                entry = {}
                for side in ("native", "fzgm"):
                    a_rec, b_rec = ra.get(side), rb.get(side)
                    if not a_rec and not b_rec:
                        continue
                    combo = {"a": a_rec, "b": b_rec}
                    if a_rec and b_rec and a_rec["psnr"] is not None and b_rec["psnr"] is not None:
                        pd = abs(a_rec["psnr"] - b_rec["psnr"])
                        if a_rec["cr"] and b_rec["cr"]:
                            crd = abs(a_rec["cr"] - b_rec["cr"]) / a_rec["cr"]
                        else:
                            crd = 0.0
                        if pd > anomaly_psnr_db or crd > anomaly_cr_pct:
                            combo["anomaly"] = True
                            anomalies.append((variant, side, ds, eb, pd, crd))
                        else:
                            max_psnr_absdiff = max(max_psnr_absdiff, pd)
                            max_cr_reldiff = max(max_cr_reldiff, crd)
                    entry[side] = combo
                data[variant][ds][str(eb)] = entry

    pairings = [{"key": v, "label": DEFAULT_LABELS.get(v, v)} for v in variants]

    meta = {
        "a": {"label": base_a["label"], "dir": base_a["dir"], "meta": base_a["meta"]},
        "b": {"label": base_b["label"], "dir": base_b["dir"], "meta": base_b["meta"]},
        "datasets": datasets,
        "ebs": ebs,
        "pairings": pairings,
        "matched_cells": len(all_keys),
        "max_cr_reldiff": max_cr_reldiff,
        "max_psnr_absdiff": max_psnr_absdiff,
        "anomaly_count": len(anomalies),
    }
    return {"meta": meta, "data": data}, anomalies


TEMPLATE = r"""<title>__TITLE__</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ee; --border: #e2e0d9;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8880;
  --series-a: #c9752f; --series-b: #2a78d6;
  --status-good: #0ca30c; --status-critical: #d03b3b; --status-warning: #fab219;
  --grid-line: #d8d6cd;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #222221; --border: #34332f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8a8880;
    --series-a: #e08e46; --series-b: #3987e5;
    --status-good: #0ca30c; --status-critical: #e66767; --status-warning: #fab219;
    --grid-line: #34332f;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #222221; --border: #34332f;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8a8880;
  --series-a: #e08e46; --series-b: #3987e5;
  --status-good: #0ca30c; --status-critical: #e66767; --status-warning: #fab219;
  --grid-line: #34332f;
}
* { box-sizing: border-box; } body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  padding: 32px 24px 64px; max-width: 1280px; margin: 0 auto; }
header.hero h1 { font-size: 1.5rem; margin: 0 0 4px; }
header.hero p.sub { color: var(--text-secondary); margin: 0 0 20px; font-size: 0.92rem; max-width: 76ch; line-height: 1.5; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 22px; }
.tile { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.tile .value { font-size: 1.2rem; font-weight: 600; }
.tile .value.good { color: var(--status-good); }
.controls { display: flex; flex-wrap: wrap; gap: 20px; align-items: center; margin-bottom: 6px; padding: 12px 14px;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; }
.control-group { display: flex; align-items: center; gap: 8px; }
.control-group .cg-label { font-size: 0.78rem; color: var(--text-secondary); font-weight: 600; }
.segmented { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.segmented button { font: inherit; font-size: 0.82rem; padding: 6px 12px; border: none; background: var(--surface-1);
  color: var(--text-secondary); cursor: pointer; border-right: 1px solid var(--border); }
.segmented button:last-child { border-right: none; }
.segmented button[aria-pressed="true"] { background: var(--series-b); color: #fff; font-weight: 600; }
.legend { display: flex; gap: 18px; align-items: center; margin: 16px 2px 10px; font-size: 0.82rem; color: var(--text-secondary); flex-wrap: wrap; }
.legend .item { display: flex; align-items: center; gap: 6px; }
.legend .swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.legend .swatch.anomaly { background: repeating-linear-gradient(45deg, var(--status-warning), var(--status-warning) 2px, transparent 2px, transparent 4px); border: 1px solid var(--status-warning); }
.legend .swatch.missing { background: transparent; border: 1.5px dashed var(--text-muted); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px; margin-bottom: 30px; }
.panel { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px 8px; }
.panel h3 { margin: 0 0 8px; font-size: 0.9rem; font-weight: 600; }
.panel svg { width: 100%; height: auto; display: block; overflow: visible; }
.axis-tick text { fill: var(--text-muted); font-size: 8px; }
.grid-line { stroke: var(--grid-line); stroke-width: 1; }
.bar.a { fill: var(--series-a); } .bar.b { fill: var(--series-b); }
.bar-group:hover .bar { opacity: 0.82; }
.xlabel { fill: var(--text-secondary); font-size: 8.5px; text-anchor: middle; }
.anomaly-mark { stroke: var(--status-warning); stroke-width: 1.5; }
.missing-mark { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2 2; fill: none; }
#tooltip { position: fixed; pointer-events: none; z-index: 50; background: var(--text-primary); color: var(--surface-1);
  font-size: 0.78rem; line-height: 1.45; padding: 8px 10px; border-radius: 8px; max-width: 260px; opacity: 0;
  transition: opacity 0.08s ease; box-shadow: 0 4px 16px rgba(0,0,0,0.25); }
#tooltip .t-title { font-weight: 700; margin-bottom: 2px; }
#tooltip .t-row { display: flex; justify-content: space-between; gap: 10px; }
#tooltip .t-warn { margin-top: 4px; color: #ffb84d; }
.callout { border: 1px solid var(--status-warning); background: color-mix(in srgb, var(--status-warning) 10%, var(--surface-1));
  border-radius: 10px; padding: 14px 16px; margin: 8px 0 20px; font-size: 0.86rem; line-height: 1.55; }
.callout.good { border-color: var(--status-good); background: color-mix(in srgb, var(--status-good) 10%, var(--surface-1)); }
.callout h3 { margin: 0 0 8px; font-size: 0.92rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
table.data-table { border-collapse: collapse; width: 100%; font-size: 0.8rem; white-space: nowrap; }
table.data-table th, table.data-table td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; }
table.data-table th:nth-child(1), table.data-table td:nth-child(1),
table.data-table th:nth-child(2), table.data-table td:nth-child(2) { text-align: left; }
table.data-table thead th { position: sticky; top: 0; background: var(--surface-2); color: var(--text-secondary);
  font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
table.data-table tbody tr:hover { background: var(--surface-2); }
.status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; }
.status-dot.warn { background: var(--status-warning); }
.speedup-up { color: var(--status-good); font-weight:600; } .speedup-down { color: var(--status-critical); font-weight:600; }
footer.note { color: var(--text-muted); font-size: 0.78rem; margin-top: 24px; line-height: 1.6; }
h2.section { font-size: 1.05rem; margin: 30px 0 10px; }
</style>
<div class="viz-root">
  <header class="hero">
    <h1>__TITLE__</h1>
    <p class="sub">__SUBTITLE__</p>
  </header>
  __CALLOUT__
  <div class="tiles">
    <div class="tile"><div class="label">Matched cells</div><div class="value" id="tile-total">&mdash;</div></div>
    <div class="tile"><div class="label">Max CR diff (clean)</div><div class="value good" id="tile-crdiff">&mdash;</div></div>
    <div class="tile"><div class="label">Max PSNR diff (clean)</div><div class="value good" id="tile-psnrdiff">&mdash;</div></div>
    <div class="tile"><div class="label">Flagged anomalies</div><div class="value" id="tile-anomalies">&mdash;</div></div>
  </div>
  <div class="controls">
    <div class="control-group"><span class="cg-label">Error bound</span><div class="segmented" id="eb-control"></div></div>
    <div class="control-group"><span class="cg-label">Metric</span><div class="segmented" id="metric-control"></div></div>
    <div class="control-group"><span class="cg-label">Side</span><div class="segmented" id="side-control"></div></div>
    <div class="control-group"><span class="cg-label">Y axis</span><div class="segmented" id="scale-control"></div></div>
  </div>
  <div class="legend" id="legend"></div>
  <div class="grid" id="chart-grid"></div>
  <h2 class="section">Exact values — current selection</h2>
  <div class="table-wrap">
    <table class="data-table" id="data-table">
      <thead><tr><th>Pairing</th><th>Dataset</th><th id="th-a">A</th><th id="th-b">B</th><th>Ratio (B/A)</th><th>Flag</th></tr></thead>
      <tbody id="data-table-body"></tbody>
    </table>
  </div>
  <footer class="note" id="footer-note"></footer>
  <div id="tooltip"></div>
</div>
<script>
const DATA = __DATA_JSON__;
const METRICS = [
  { key: 'cgbs', label: 'Compress GB/s', fmt: v => v.toFixed(1) },
  { key: 'dgbs', label: 'Decompress GB/s', fmt: v => v.toFixed(1) },
  { key: 'cr', label: 'Compression ratio', fmt: v => v.toFixed(v < 10 ? 2 : 1) + '×' },
];
let state = { eb: String(DATA.meta.ebs[Math.floor(DATA.meta.ebs.length/2)]), metric: 'cgbs', side: 'native', scale: 'log' };

function buildControls() {
  const ebC = document.getElementById('eb-control');
  DATA.meta.ebs.forEach(eb => {
    const b = document.createElement('button');
    b.textContent = String(eb);
    b.setAttribute('aria-pressed', String(eb) === state.eb);
    b.onclick = () => { state.eb = String(eb); render(); };
    ebC.appendChild(b);
  });
  const mC = document.getElementById('metric-control');
  METRICS.forEach(m => {
    const b = document.createElement('button');
    b.textContent = m.label;
    b.setAttribute('aria-pressed', m.key === state.metric);
    b.onclick = () => { state.metric = m.key; render(); };
    mC.appendChild(b);
  });
  const sdC = document.getElementById('side-control');
  [['native','Native reference'], ['fzgm','FZGM port']].forEach(([key,label]) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', key === state.side);
    b.onclick = () => { state.side = key; render(); };
    sdC.appendChild(b);
  });
  const scC = document.getElementById('scale-control');
  [['log','Log'], ['linear','Linear']].forEach(([key,label]) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.setAttribute('aria-pressed', key === state.scale);
    b.onclick = () => { state.scale = key; render(); };
    scC.appendChild(b);
  });
  document.getElementById('legend').innerHTML = `
    <div class="item"><span class="swatch" style="background:var(--series-a)"></span>${DATA.meta.a.label}</div>
    <div class="item"><span class="swatch" style="background:var(--series-b)"></span>${DATA.meta.b.label}</div>
    <div class="item"><span class="swatch anomaly"></span>Flagged anomaly (CR/PSNR disagreement — see table)</div>
    <div class="item"><span class="swatch missing"></span>Not run on one baseline</div>`;
  document.getElementById('th-a').textContent = DATA.meta.a.label;
  document.getElementById('th-b').textContent = DATA.meta.b.label;
  document.getElementById('footer-note').innerHTML =
    `Baseline A: <code>${DATA.meta.a.dir}</code> (${DATA.meta.a.label}) &middot; Baseline B: <code>${DATA.meta.b.dir}</code> (${DATA.meta.b.label}) &middot; "Ratio" &gt; 1× means B is higher.`;
}
function syncControlStates() {
  document.querySelectorAll('#eb-control button').forEach(b => b.setAttribute('aria-pressed', b.textContent === state.eb));
  document.querySelectorAll('#metric-control button').forEach(b => {
    const m = METRICS.find(m => m.label === b.textContent);
    b.setAttribute('aria-pressed', m && m.key === state.metric);
  });
  document.querySelectorAll('#side-control button').forEach(b => {
    const key = b.textContent === 'Native reference' ? 'native' : 'fzgm';
    b.setAttribute('aria-pressed', key === state.side);
  });
  document.querySelectorAll('#scale-control button').forEach(b => b.setAttribute('aria-pressed', b.textContent.toLowerCase() === state.scale));
}
function fillTiles() {
  document.getElementById('tile-total').textContent = DATA.meta.matched_cells;
  document.getElementById('tile-crdiff').textContent = (DATA.meta.max_cr_reldiff*100).toFixed(2) + '%';
  document.getElementById('tile-psnrdiff').textContent = DATA.meta.max_psnr_absdiff.toFixed(3) + ' dB';
  document.getElementById('tile-anomalies').textContent = DATA.meta.anomaly_count;
}
const tooltip = document.getElementById('tooltip');
function showTip(evt, html) { tooltip.innerHTML = html; tooltip.style.opacity = 1; tooltip.style.left = (evt.clientX+14)+'px'; tooltip.style.top = (evt.clientY+14)+'px'; }
function hideTip() { tooltip.style.opacity = 0; }
function logY(v, dmax, h) { const dmin=1; const cl=Math.max(dmin,Math.min(dmax,v)); const t=(Math.log10(cl)-Math.log10(dmin))/(Math.log10(dmax)-Math.log10(dmin)); return h-t*h; }
function linY(v, dmax, h) { const cl=Math.max(0,Math.min(dmax,v)); return h-(cl/dmax)*h; }
function yOf(v, dmax, h, scale) { return scale==='linear' ? linY(v,dmax,h) : logY(v,dmax,h); }
function linTicks(dmax) { const raw=dmax/4; const mag=Math.pow(10,Math.floor(Math.log10(raw))); const norm=raw/mag;
  const step=(norm<1.5)?mag:(norm<3)?2*mag:(norm<7)?5*mag:10*mag; const ticks=[]; for(let t=step;t<=dmax+step*0.01;t+=step) ticks.push(Math.round(t*100)/100); return ticks; }
function cellFor(pairing, ds) { const c = DATA.data[pairing.key][ds][state.eb] || {}; return c[state.side]; }
function panelMax(pairing) {
  let m = 1;
  DATA.meta.datasets.forEach(ds => { const combo = cellFor(pairing, ds); if (!combo) return;
    ['a','b'].forEach(s => { const rec = combo[s]; if (rec && rec[state.metric] != null) m = Math.max(m, rec[state.metric]); }); });
  return m;
}
function renderPanel(pairing) {
  const W=260,H=150,padL=22,padB=16,padT=4,padR=4,plotW=W-padL-padR,plotH=H-padT-padB;
  const metric = METRICS.find(m => m.key===state.metric);
  const dmax = state.scale==='linear' ? panelMax(pairing)*1.08 : Math.max(1000, panelMax(pairing)*1.2);
  const groupW = plotW/DATA.meta.datasets.length, barW=groupW*0.30, barGap=groupW*0.06;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${pairing.label} ${metric.label}">`;
  const ticks = state.scale==='linear' ? linTicks(dmax) : [1,10,100,1000,10000].filter(t => t<=dmax);
  ticks.forEach(t => { const y=padT+yOf(t,dmax,plotH,state.scale);
    svg += `<line class="grid-line" x1="${padL}" x2="${W-padR}" y1="${y}" y2="${y}" />`;
    svg += `<g class="axis-tick"><text x="${padL-4}" y="${y+3}" text-anchor="end">${t>=1000?(t/1000)+'k':t}</text></g>`; });
  DATA.meta.datasets.forEach((ds,i) => {
    const gx = padL+i*groupW, combo = cellFor(pairing, ds), centerX = gx+groupW/2;
    svg += `<text class="xlabel" x="${centerX}" y="${H-3}">${ds.replace('-2D','')}</text>`;
    ['a','b'].forEach((s,si) => {
      const bx = gx+groupW/2-barW-barGap/2+si*(barW+barGap);
      const rec = combo ? combo[s] : null;
      if (!rec) { svg += `<rect class="missing-mark" x="${bx}" y="${padT+plotH-18}" width="${barW}" height="18" />`; return; }
      const anomaly = combo.anomaly;
      const val = rec[state.metric];
      if (val == null) { svg += `<rect class="missing-mark" x="${bx}" y="${padT+plotH-18}" width="${barW}" height="18" />`; return; }
      const y = padT+yOf(val,dmax,plotH,state.scale), h=(padT+plotH)-y;
      svg += `<g class="bar-group"><rect class="bar ${s}" x="${bx}" y="${y}" width="${barW}" height="${Math.max(h,1)}" rx="2"
        opacity="${anomaly?0.45:1}" data-side="${s}" data-ds="${ds}" data-pair="${pairing.key}" data-anomaly="${anomaly?'1':''}" />
        ${anomaly ? `<text x="${bx+barW/2}" y="${y-3}" text-anchor="middle" font-size="9" fill="var(--status-warning)">&#9888;</text>` : ''}</g>`;
    });
  });
  svg += `</svg>`;
  const el = document.createElement('div'); el.className='panel'; el.innerHTML = `<h3>${pairing.label}</h3>${svg}`;
  return el;
}
function attachHover(grid) {
  grid.querySelectorAll('rect[data-pair]').forEach(rect => {
    rect.addEventListener('mousemove', evt => {
      const { pair, ds, side: s, anomaly } = rect.dataset;
      const combo = cellFor({key: pair}, ds);
      const rec = combo ? combo[s] : null;
      const label = s === 'a' ? DATA.meta.a.label : DATA.meta.b.label;
      if (!rec) { showTip(evt, `<div class="t-title">${label} — no result</div><div>${ds}, eb=${state.eb}</div>`); return; }
      const metric = METRICS.find(m => m.key===state.metric);
      let body = `<div class="t-title">${label} — ${ds}</div>
        <div class="t-row"><span>eb</span><span>${state.eb}</span></div>
        <div class="t-row"><span>CR</span><span>${rec.cr!=null?rec.cr.toFixed(2)+'×':'–'}</span></div>
        <div class="t-row"><span>PSNR</span><span>${rec.psnr!=null?rec.psnr.toFixed(1)+' dB':'–'}</span></div>
        <div class="t-row"><span>Compress</span><span>${rec.cgbs!=null?rec.cgbs.toFixed(1)+' GB/s':'–'}</span></div>
        <div class="t-row"><span>Decompress</span><span>${rec.dgbs!=null?rec.dgbs.toFixed(1)+' GB/s':'–'}</span></div>`;
      if (anomaly) body += `<div class="t-warn">&#9888; CR/PSNR disagreement flagged — check both baselines' metadata.yaml known_issues</div>`;
      showTip(evt, body);
    });
    rect.addEventListener('mouseleave', hideTip);
  });
}
function renderGrid() {
  const grid = document.getElementById('chart-grid'); grid.innerHTML = '';
  DATA.meta.pairings.forEach(p => grid.appendChild(renderPanel(p)));
  attachHover(grid);
}
function renderTable() {
  const tbody = document.getElementById('data-table-body'); tbody.innerHTML = '';
  const metric = METRICS.find(m => m.key===state.metric);
  DATA.meta.pairings.forEach(p => {
    DATA.meta.datasets.forEach(ds => {
      const combo = cellFor(p, ds); if (!combo) return;
      const a = combo.a, b = combo.b;
      const aVal = a ? a[state.metric] : null, bVal = b ? b[state.metric] : null;
      let ratioStr='–', ratioClass='';
      if (aVal!=null && bVal!=null && aVal>0) { const r=bVal/aVal; ratioClass = r>=1?'speedup-up':'speedup-down'; ratioStr = r.toFixed(2)+'×'; }
      const flag = combo.anomaly ? '<span class="status-dot warn"></span>flagged' : '';
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${p.label}</td><td>${ds}</td>
        <td>${aVal!=null?metric.fmt(aVal):'–'}</td><td>${bVal!=null?metric.fmt(bVal):'–'}</td>
        <td class="${ratioClass}">${ratioStr}</td><td>${flag}</td>`;
      tbody.appendChild(tr);
    });
  });
}
function render() { syncControlStates(); renderGrid(); renderTable(); }
fillTiles(); buildControls(); render();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline_a", help="path to results/baselines/<id> directory")
    ap.add_argument("baseline_b", help="path to results/baselines/<id> directory")
    ap.add_argument("-o", "--output", default="comparison_artifact.html")
    ap.add_argument("--anomaly-psnr-db", type=float, default=5.0,
                     help="flag a cell if PSNR differs by more than this many dB between baselines (default 5.0)")
    ap.add_argument("--anomaly-cr-pct", type=float, default=0.2,
                     help="flag a cell if CR differs by more than this fraction between baselines (default 0.2 = 20%%)")
    args = ap.parse_args()

    base_a = load_baseline(args.baseline_a)
    base_b = load_baseline(args.baseline_b)
    payload, anomalies = build_data(base_a, base_b, args.anomaly_psnr_db, args.anomaly_cr_pct)

    title = f"{base_a['label']} vs. {base_b['label']} — Compressor Comparison"
    subtitle = (f"Comparing {payload['meta']['matched_cells']} matched cells between "
                f"<code>{base_a['dir']}</code> and <code>{base_b['dir']}</code>. "
                "CR/PSNR are checked for agreement before charting throughput.")

    if anomalies:
        rows = "".join(
            f"<tr><td>{v}</td><td>{s}</td><td>{ds}</td><td>{eb}</td>"
            f"<td>{pd:.2f} dB</td><td>{crd*100:.1f}%</td></tr>"
            for v, s, ds, eb, pd, crd in anomalies
        )
        callout = (
            '<div class="callout"><h3>&#9888; '
            f'{len(anomalies)} cell(s) flagged — CR/PSNR disagree between baselines more than expected</h3>'
            '<p>Deterministic algorithms on identical input should give near-identical CR/PSNR regardless of '
            'GPU. A real disagreement usually means a bug in one of the two builds, not measurement noise — '
            'see each baseline\'s <code>metadata.yaml</code> <code>known_issues</code> field before trusting '
            'these specific cells.</p>'
            '<table><thead><tr><th>Pairing</th><th>Side</th><th>Dataset</th><th>eb</th>'
            '<th>PSNR diff</th><th>CR diff</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
        )
    else:
        callout = ('<div class="callout good"><h3>&#9989; No CR/PSNR anomalies found</h3>'
                   '<p>Every matched cell agrees within tolerance between the two baselines.</p></div>')

    html = (TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__CALLOUT__", callout)
            .replace("__DATA_JSON__", json.dumps(payload)))

    Path(args.output).write_text(html)
    print(f"wrote {args.output} ({len(html)} bytes)")
    if anomalies:
        print(f"{len(anomalies)} anomalies flagged (see callout in the output):")
        for v, s, ds, eb, pd, crd in anomalies:
            print(f"  {v}/{s}/{ds}/eb={eb}: PSNR diff {pd:.2f}dB, CR diff {crd*100:.1f}%")


if __name__ == "__main__":
    main()
