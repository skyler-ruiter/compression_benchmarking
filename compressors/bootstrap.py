#!/usr/bin/env python3
"""Reproduce ~/compressors/ on a new machine from manifest.toml.

    source scripts/env-<machine>.sh          # toolchain, CUDA_ARCH, PATHs
    python compressors/bootstrap.py           # everything
    python compressors/bootstrap.py fetch cusz sz3     # just clone+patch, no build
    python compressors/bootstrap.py build cusz         # just (re)build
    python compressors/bootstrap.py status             # what's on disk vs the manifest
    python compressors/bootstrap.py capture cusz       # re-extract cusz's patch from its checkout

Stdlib only (tomllib, py>=3.11). Idempotent: re-running fetches nothing new,
re-applies no patch already in the tree, and only rebuilds what you name.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.toml"
PATCHES = HERE / "patches"
BUILD = HERE / "build"
VENDOR = HERE / "vendor"


def load():
    with open(MANIFEST, "rb") as f:
        m = tomllib.load(f)
    root = os.environ.get("COMPRESSORS_ROOT") or os.path.expanduser(
        m.get("defaults", {}).get("clone_root", "~/compressors")
    )
    return m, Path(root).expanduser()


def sh(cmd, cwd=None, env=None):
    print(f"  $ {' '.join(str(c) for c in cmd)}" + (f"   (in {cwd})" if cwd else ""))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True,
                   env={**os.environ, **(env or {})})


def out(cmd, cwd=None) -> str:
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def entries(m: dict):
    for name, e in m.get("compressor", {}).items():
        yield name, e


def dest_dir(root: Path, name: str, e: dict) -> Path:
    return root / e.get("clone_dir", name)


# ── fetch: clone at pin / checkout, submodules, patches ──────────────────────

def patch_applied(repo: Path, patch: Path) -> bool:
    r = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                       cwd=repo, capture_output=True)
    return r.returncode == 0


def apply_patches(name: str, e: dict, d: Path):
    for rel in e.get("patches", []):
        p = PATCHES / rel
        if not p.exists():
            sys.exit(f"{name}: patch missing: {p}")
        if patch_applied(d, p):
            print(f"  patch already in tree: {rel}")
            continue
        # --3way resolves context drift against the pinned blobs; on a clean
        # re-clone at the pinned commit it applies exactly.
        sh(["git", "apply", "--3way", "--whitespace=nowarn", str(p)], cwd=d)


def fetch_one(name: str, e: dict, root: Path):
    kind = e.get("kind", "git")
    d = dest_dir(root, name, e)
    print(f"\n=== {name}  ->  {d} ===")

    if kind == "vendored":
        src = VENDOR / name          # vendor/<manifest-key>/
        if not src.exists():
            sys.exit(f"{name}: vendored source missing: {src}")
        if d.exists():
            print("  exists, refreshing source (build/ kept)")
        d.mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-a", "--delete",
            "--exclude=build", "--exclude=build_sm90", "--exclude=.git",
            f"{src}/", f"{d}/"])
        return

    if kind == "redist":
        d.mkdir(parents=True, exist_ok=True)
        sh([BUILD / e["build"], d],
           env={"NVCOMP_URL": e["url"], "NVCOMP_VERSION": e.get("version", "")})
        return

    # git
    if not d.exists():
        sh(["git", "clone", e["repo"], d])
    have = out(["git", "rev-parse", "HEAD"], cwd=d)
    if have != e["commit"]:
        try:
            sh(["git", "fetch", "--tags", "origin"], cwd=d)
        except subprocess.CalledProcessError:
            pass
        sh(["git", "-c", "advice.detachedHead=false", "checkout", e["commit"]], cwd=d)
    else:
        print(f"  at pinned commit {have[:12]}")
    if e.get("submodules"):
        sh(["git", "submodule", "update", "--init", "--recursive"], cwd=d)
    apply_patches(name, e, d)


# ── build ───────────────────────────────────────────────────────────────────

def build_one(name: str, e: dict, root: Path):
    script = e.get("build")
    if not script:
        print(f"\n=== {name}: no build recipe (see manifest notes), skipping ===")
        return
    d = dest_dir(root, name, e)
    if not d.exists():
        sys.exit(f"{name}: not fetched yet ({d} missing) — run `fetch` first")
    print(f"\n=== build {name} ===")
    if e.get("kind") == "redist":
        return  # nvcomp.sh is the 'build' and runs in fetch
    sh([BUILD / script, d])
    cli = e.get("cli")
    if cli:
        p = d / cli
        print(f"  -> {p}   {'OK' if p.exists() else 'MISSING (check build log)'}")


# ── status / capture ────────────────────────────────────────────────────────

def status(m, root):
    print(f"COMPRESSORS_ROOT = {root}\n")
    print(f"{'name':<18} {'on disk':<10} {'commit':<14} {'cli':<8} notes")
    for name, e in entries(m):
        d = dest_dir(root, name, e)
        on_disk = "yes" if d.exists() else "-"
        commit = "-"
        if d.exists() and (d / ".git").exists():
            head = out(["git", "rev-parse", "HEAD"], cwd=d)
            commit = ("== " if head == e.get("commit") else "!! ") + head[:9]
        cli_ok = "-"
        if e.get("cli"):
            cli_ok = "yes" if (d / e["cli"]).exists() else "no"
        print(f"{name:<18} {on_disk:<10} {commit:<14} {cli_ok:<8} {e.get('notes','')[:60]}")


def capture(m, root, name):
    e = m["compressor"][name]
    d = dest_dir(root, name, e)
    if not (d / ".git").exists():
        sys.exit(f"{name}: not a git checkout, nothing to capture")
    pats = e.get("patches", [])
    if not pats:
        sys.exit(f"{name}: manifest lists no patch file to write to")
    target = PATCHES / pats[0]
    # raw, unstripped — trailing empty context lines are load-bearing in a diff
    diff = subprocess.run(["git", "diff"], cwd=d, check=True,
                          capture_output=True, text=True).stdout
    if not diff:
        sys.exit(f"{name}: `git diff` is empty — nothing to capture "
                 f"(is the checkout clean / already committed?)")
    if len(pats) > 1:
        print(f"warning: {name} has {len(pats)} patch files; writing the combined "
              f"diff to the first ({pats[0]}) only")
    target.write_text(diff)
    print(f"wrote {target}  ({diff.count(chr(10))} lines)")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", nargs="?", default="all",
                    choices=["all", "fetch", "build", "status", "capture"])
    ap.add_argument("names", nargs="*", help="compressor keys (default: all)")
    a = ap.parse_args()
    m, root = load()

    if a.action == "status":
        return status(m, root)
    if a.action == "capture":
        if not a.names:
            sys.exit("capture needs a compressor name")
        for n in a.names:
            capture(m, root, n)
        return

    sel = a.names or [n for n, _ in entries(m)]
    unknown = [n for n in sel if n not in m["compressor"]]
    if unknown:
        sys.exit(f"unknown: {unknown}\nknown: {[n for n,_ in entries(m)]}")

    root.mkdir(parents=True, exist_ok=True)
    for n in sel:
        e = m["compressor"][n]
        if a.action in ("all", "fetch"):
            fetch_one(n, e, root)
        if a.action in ("all", "build"):
            build_one(n, e, root)

    print("\ndone.")
    if a.action in ("all", "build"):
        print("CLI paths are unchanged from before — scripts/env-<machine>.sh still points at them.")


if __name__ == "__main__":
    if not shutil.which("git"):
        sys.exit("git not found on PATH")
    main()
