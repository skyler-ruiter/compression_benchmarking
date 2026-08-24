import hashlib
import os
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from benchkit.cli import main
from benchkit.schema import load_result_file


FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, struct, sys
from pathlib import Path

args = sys.argv[1:]
def value(flag): return args[args.index(flag) + 1]
report = Path(value("--report-json"))
phase = "benchmark" if "-b" in args else "compress" if "-z" in args else "decompress"
if os.environ.get("BENCHKIT_FAKE_FAIL") == phase:
    report.write_text(json.dumps({"status":"fail", "error_message":"forced fake failure"}))
    raise SystemExit(7)
source = Path(value("-i"))
if phase == "compress":
    data = source.read_bytes()
    output = Path(value("-o")); output.write_bytes(data[:max(1, len(data)//2)])
    payload = {"status":"ok", "size":{"original_bytes":len(data),
               "compressed_bytes":output.stat().st_size}}
elif phase == "decompress":
    original = Path(value("--compare")).read_bytes()
    values = list(struct.unpack("<" + "f" * (len(original)//4), original))
    values[0] += 1.0e-4
    Path(value("-o")).write_bytes(struct.pack("<" + "f" * len(values), *values))
    payload = {"status":"ok"}
else:
    runs = int(value("--runs")); original = Path(value("--compare")).stat().st_size
    payload = {"status":"ok", "size":{"original_bytes":original,
               "compressed_bytes":max(1, original//2)},
               "timing":{"compress":{"device_ms":{"all":[1.0]*runs},
                                       "host_wall_ms":{"all":[1.1]*runs}},
                         "decompress":{"device_ms":{"all":[2.0]*runs},
                                         "host_wall_ms":{"all":[2.1]*runs}}},
               "stages":[], "stage_versions":{}, "memory":{"peak_device_bytes":1024},
               "config":{"coloring":True}, "run_notes":{}}
report.write_text(json.dumps(payload))
'''


class FakeCliEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        raw = struct.pack("<4f", 0.0, 1.0, 2.0, 3.0)
        (self.data / "f.bin").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        datasets = self.root / "datasets.yaml"
        datasets.write_text(yaml.safe_dump({
            "d": {"dtype": "f32", "dim_order": "fast-to-slow", "root": str(self.data),
                  "fields": {"f": {"dims": [4], "path": "f.bin"}}}}, sort_keys=False))
        (self.root / "datasets.checksums.yaml").write_text(yaml.safe_dump({
            "checksum_schema_version": 1, "algorithm": "sha256",
            "byte_scope": "declared_prefix", "datasets": {"d": {"f": digest}}},
            sort_keys=False))
        self.datasets = datasets
        self.cli = self.root / "fake-fzgmod-cli"
        self.cli.write_text(FAKE_CLI)
        self.cli.chmod(0o755)
        self.experiment = self.root / "experiment.yaml"
        self.experiment.write_text(yaml.safe_dump({
            "name": "fake-e2e", "datasets": ["d"], "fields": "all",
            "error": {"mode": "rel_range", "bounds": [1e-3, 1e-2]},
            "repetitions": 2, "warmup_reps": 1,
            "runs": [{"compressor": "fzgm", "variant": "fake",
                      "pipeline": "configs/pipelines/cusz.toml", "cli_path": str(self.cli),
                      "tool_version": "fixture-v1", "tool_source_commit": "fixture-commit",
                      "tool_build_flags": "fixture-build", "tool_patch_sha256": "clean"}]},
            sort_keys=False))
        self.results = self.root / "results"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *extra):
        argv = ["run", str(self.experiment), "--datasets", str(self.datasets),
                "--results-root", str(self.results), "--session-id", "session", *extra]
        with redirect_stdout(StringIO()):
            return main(argv)

    def test_run_resume_shards_merge_verify_and_artifact(self):
        self.assertEqual(self._run("--shard", "0/2"), 0)
        shard = self.results / "session" / "runs.shard-0-of-2.jsonl"
        before = shard.read_text()
        self.assertEqual(self._run("--shard", "0/2"), 0)
        self.assertEqual(shard.read_text(), before)
        self.assertEqual(self._run("--shard", "1/2"), 0)
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["merge", str(self.results / "session")]), 0)
            self.assertEqual(main(["verify", str(self.results / "session")]), 0)
            self.assertEqual(main(["artifact", "build", str(self.results / "session"),
                                   str(self.root / "bundle")]), 0)
            self.assertEqual(main(["artifact", "verify", str(self.root / "bundle")]), 0)
        rows = load_result_file(self.results / "session" / "runs.jsonl")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "ok" for row in rows))

    def test_failure_is_append_only_evidence(self):
        with patch.dict(os.environ, {"BENCHKIT_FAKE_FAIL": "compress"}):
            self.assertEqual(self._run(), 0)
        rows = load_result_file(self.results / "session" / "runs.jsonl")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "fail" for row in rows))
        self.assertTrue(all(row["fail_phase"] == "compress" for row in rows))


if __name__ == "__main__":
    unittest.main()
