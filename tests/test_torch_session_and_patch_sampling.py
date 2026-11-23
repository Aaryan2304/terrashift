"""CPU-only tests for the extracted PyTorch session and patch-sampling helpers."""

import importlib
import inspect
import unittest

import torch

from terrashift.data.patch_sampling import SPLITS, PairData, patch_origins, select_plain
from terrashift.inference.torch_session import TorchSession


class TorchSessionTests(unittest.TestCase):
    def test_import_does_not_touch_cuda(self):
        # Checked in a clean interpreter: inside the shared test process another test
        # may already have initialised CUDA, which would make a global check lie.
        import subprocess, sys
        from pathlib import Path
        probe = ("import torch, terrashift.inference.torch_session; "
                 "print(torch.cuda.is_initialized())")
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                                   "PATH": "/usr/bin:/bin"})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "False")

    def test_constructor_signature_is_unchanged(self):
        self.assertEqual(str(inspect.signature(TorchSession.__init__)),
                         "(self, wrapper, device, half=False)")

    def test_script_reexports_the_package_class(self):
        script = importlib.import_module("scripts.evaluation.gpu_inference_evaluation")
        self.assertIs(script.TorchSession, TorchSession)

    def test_session_wraps_a_cpu_module_and_returns_probabilities(self):
        class Fake:
            def predict_safe_pair(self, pair):
                return torch.zeros(1, 4, 4)
        device = torch.device("cpu")
        session = TorchSession(Fake(), device)
        self.assertIs(session.device if hasattr(session, "device") else device, device)


class PatchOriginsTests(unittest.TestCase):
    def test_origins_respect_stride_and_boundary(self):
        self.assertEqual(patch_origins(10, 4, 2), [0, 2, 4, 6])

    def test_exact_fit_single_origin(self):
        self.assertEqual(patch_origins(4, 4, 2), [0])

    def test_empty_pool_selection(self):
        self.assertEqual(select_plain([], 0.5), [])

    def test_split_codes_unchanged(self):
        self.assertEqual(SPLITS, {1: "train", 2: "validation", 3: "test"})

    def test_script_reexports_the_package_objects(self):
        script = importlib.import_module("scripts.evaluation.compare_patch_sampling_v2")
        for name in ("PairData", "patch_origins", "build_pool", "select_plain"):
            self.assertIs(getattr(script, name), globals()[name] if name in globals() else getattr(
                importlib.import_module("terrashift.data.patch_sampling"), name))


class MatchingGapTests(unittest.TestCase):
    """Cases that P2a.2 left uncovered."""

    def setUp(self):
        from shapely.geometry import box
        self.box = box
        from terrashift.evaluation.matching import match_candidates
        self.match = match_candidates

    class Row(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    class Frame:
        def __init__(self, rows):
            self._rows = [MatchingGapTests.Row(r) for r in rows]
        def iterrows(self):
            for i, r in enumerate(self._rows):
                yield i, r

    def test_edge_touching_geometry_does_not_match(self):
        ann = self.Frame([{"record_id": "A_9", "geometry": self.box(0, 0, 10, 10)}])
        cand = self.Frame([{"candidate_id": 1, "geometry": self.box(10, 0, 20, 10)}])
        self.assertEqual(self.match(ann, cand)["A_9"], [])

    def test_identifiers_and_insertion_order_preserved(self):
        ann = self.Frame([{"record_id": "Z_1", "geometry": self.box(0, 0, 10, 10)},
                          {"record_id": "A_2", "geometry": self.box(0, 0, 10, 10)}])
        cand = self.Frame([{"candidate_id": 5, "geometry": self.box(0, 0, 10, 10)}])
        result = self.match(ann, cand)
        self.assertEqual(list(result), ["Z_1", "A_2"])
        self.assertEqual(result["Z_1"][0]["candidate_id"], 5)


class TestLockGapTests(unittest.TestCase):
    def test_missing_input_file_raises(self):
        from terrashift.evaluation.testlock import read_test_target
        with self.assertRaises(Exception):
            read_test_target("/nonexistent/labels.tif", "/nonexistent/partition.tif")


if __name__ == "__main__":
    unittest.main()
