"""Guards that the demonstration script points at current artefacts.

The demo reads finished results back instead of recomputing them, so it can go
stale silently: it once displayed CPU INT8 numbers from the benchmark that
predates the optimiser fix, contradicting the report. These tests fail if the
demo references a missing file, or a benchmark whose schema predates the
per-run optimisation record.
"""

import json
import os
import re
import unittest

from pathlib import Path as _Path
HAVE_OUTPUTS = (_Path(__file__).resolve().parents[1] / "outputs").is_dir()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "scripts", "demo.sh")

# Files the demo reads for numbers. Data products are not required, because a
# fresh clone has none; results committed under outputs/ are.
REQUIRED_PREFIXES = ("outputs/bench/", "outputs/quant/", "outputs/figures/",
                     "outputs/checkpoints/")


def referenced_paths():
    with open(DEMO) as handle:
        text = handle.read()
    paths = re.findall(r'open\(\s*"([^"]+)"', text)
    paths += re.findall(r'open\(\s*os\.path\.join\("([^"]+)"', text)
    return [path for path in paths if not path.startswith("/")]


@unittest.skipUnless(HAVE_OUTPUTS, "generated outputs are not in the repository")
class DemoReferenceTests(unittest.TestCase):

    def test_demo_script_exists(self):
        self.assertTrue(os.path.exists(DEMO), "demo.sh is a submission deliverable")

    def test_result_paths_referenced_by_demo_exist(self):
        missing = [path for path in referenced_paths()
                   if path.startswith(REQUIRED_PREFIXES)
                   and not os.path.exists(os.path.join(ROOT, path))]
        self.assertEqual(missing, [], "demo.sh reads missing results: %s" % missing)

    def test_benchmark_used_by_demo_records_optimisation(self):
        bench = [path for path in referenced_paths()
                 if "bench" in path and path.endswith(".json")]
        self.assertTrue(bench, "demo.sh should read a provider benchmark")
        for relative in bench:
            with open(os.path.join(ROOT, relative)) as handle:
                payload = json.load(handle)
            for name, entry in payload["runs"].items():
                self.assertIn("optimisation", entry,
                              "%s: %s has no optimisation level, so it predates "
                              "the QDQ correction" % (relative, name))

    def test_demo_does_not_read_superseded_benchmark(self):
        superseded = [path for path in referenced_paths()
                      if path == "outputs/bench/oscd_validation_providers/"
                      "validation_benchmark.json"]
        self.assertEqual(superseded, [],
                         "the CPU INT8 numbers in that benchmark are wrong; the "
                         "demo must read oscd_validation_providers_v2")


if __name__ == "__main__":
    unittest.main()
