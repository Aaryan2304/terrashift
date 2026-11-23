"""Tests for provider verification in terrashift/inference/onnx_runtime.py.

The failure being guarded against is silent: ONNX Runtime drops a provider whose
libraries are missing, logs once, and still runs. A benchmark then records a CPU
number under a GPU label. These tests pin the guard that prevents it.
"""

import os
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from terrashift.inference.onnx_runtime import (CPU, CUDA, create_session, preload_cuda_libraries,
                               require_provider)


def write_tiny_model(path):
    """A one-node Relu model, enough to build a real session."""
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


class RequireProviderTests(unittest.TestCase):
    def test_present_provider_passes(self):
        require_provider([CUDA, CPU], CUDA)
        require_provider([CPU], CPU)

    def test_missing_provider_raises(self):
        with self.assertRaisesRegex(RuntimeError, "is not active"):
            require_provider([CPU], CUDA)

    def test_message_names_the_provider_and_the_fix(self):
        with self.assertRaises(RuntimeError) as caught:
            require_provider([CPU], CUDA)
        message = str(caught.exception)
        self.assertIn(CUDA, message)
        self.assertIn("preload_cuda_libraries", message)
        self.assertIn("mislabelled", message)
        self.assertIn(CPU, message)

    def test_silent_fallback_shape_is_rejected(self):
        # This is exactly what ORT returns when the CUDA libraries are missing.
        with self.assertRaises(RuntimeError):
            require_provider(["CPUExecutionProvider"], "CUDAExecutionProvider")


class CreateSessionTests(unittest.TestCase):
    def test_cpu_session_reports_its_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_tiny_model(os.path.join(temporary, "tiny.onnx"))
            session, active = create_session(path, [CPU], require=CPU)
            self.assertIn(CPU, active)
            result = session.run(None, {"x": np.ones((1, 1, 2, 2), np.float32)})[0]
            self.assertTrue(np.array_equal(result, np.ones((1, 1, 2, 2), np.float32)))

    def test_requiring_cuda_from_a_cpu_only_request_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_tiny_model(os.path.join(temporary, "tiny.onnx"))
            with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
                create_session(path, [CPU], require=CUDA)

    def test_no_requirement_returns_whatever_was_built(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = write_tiny_model(os.path.join(temporary, "tiny.onnx"))
            _, active = create_session(path, [CPU])
            self.assertEqual(active, [CPU])


class PreloadTests(unittest.TestCase):
    def test_preload_reports_whether_it_ran(self):
        # On this build preload_dlls exists, so the call is made and returns
        # True. The assertion covers both cases so the test stays portable.
        import onnxruntime as ort
        made = preload_cuda_libraries()
        self.assertEqual(made, hasattr(ort, "preload_dlls"))

    def test_preload_is_safe_to_repeat(self):
        preload_cuda_libraries()
        preload_cuda_libraries()


if __name__ == "__main__":
    unittest.main()
