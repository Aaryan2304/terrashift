"""Regression tests for the QDQ Conv -> QLinearConv fusion defect.

The defect: at ORT_ENABLE_EXTENDED and above, the CPU execution provider fuses
Conv into QLinearConv, and the fused kernel reuses the int32 bias with the scale
``x_scale * w_scale`` instead of the bias's own dequantisation scale. The ONNX
QLinearConv specification requires those to be equal; when they are not, the
fused kernel is silently wrong. Upstream issue 24711, fixed in PR 28229.

Two things are pinned here:

1. The minimal reproducer behaves as the upstream analysis says it does, so the
   workaround in ``utils.ort_session`` stays justified.
2. CPU sessions pinned to ORT_ENABLE_BASIC reproduce the ONNX reference
   semantics, which is what the workaround relies on.

The models are built at opset 21 directly, an opset the bundled ONNX reference
evaluator supports, so no existing model is edited and no opset label is faked.
"""

import os
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from utils.ort_session import CPU, CUDA, create_session

ORT_VERSION = ort.__version__
# The defect was fixed by PR 28229, released in 1.27.0. On that or later
# versions the fused path is expected to be correct and the defect test skips.
DEFECT_EXPECTED = tuple(int(part) for part in ORT_VERSION.split(".")[:2]) < (1, 27)


def build_qdq_conv(bias_scale_factor=1.0, bias_zero_point=0, channels=3):
    """A one-conv QDQ model with per-channel weights and bias.

    ``bias_scale_factor`` multiplies the canonical bias scale
    ``x_scale * w_scale``. 1.0 is the QLinearConv-conforming case; anything else
    violates the invariant the fused kernel assumes. ``bias_zero_point`` is the
    int32 bias zero point, normally 0.
    """
    rng = np.random.default_rng(0)
    x_scale = np.float32(0.02)
    x_zp = np.uint8(128)
    w_scales = np.linspace(0.004, 0.008, channels).astype(np.float32)
    y_scale = np.float32(0.05)
    y_zp = np.uint8(128)

    weights = rng.standard_normal((channels, 3, 3, 3)).astype(np.float32)
    bias_float = rng.standard_normal(channels).astype(np.float32)

    canonical = x_scale * w_scales
    bias_scale = (canonical * bias_scale_factor).astype(np.float32)
    bias_int32 = np.rint((bias_float - bias_zero_point * bias_scale)
                         / bias_scale).astype(np.int32)

    initializers = [
        numpy_helper.from_array(weights, "weights"),
        numpy_helper.from_array(bias_int32, "bias_q"),
        numpy_helper.from_array(np.array(x_scale, np.float32), "x_scale"),
        numpy_helper.from_array(np.array(x_zp, np.uint8), "x_zp"),
        numpy_helper.from_array(w_scales, "w_scale"),
        numpy_helper.from_array(bias_scale, "bias_scale"),
        numpy_helper.from_array(np.full(channels, bias_zero_point, np.int32), "bias_zp"),
        numpy_helper.from_array(np.array(y_scale, np.float32), "y_scale"),
        numpy_helper.from_array(np.array(y_zp, np.uint8), "y_zp"),
    ]

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["x_q"],
                         output_dtype=TensorProto.UINT8),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]),
        helper.make_node("QuantizeLinear", ["weights", "w_scale"], ["w_q"],
                         axis=0, output_dtype=TensorProto.INT8),
        helper.make_node("DequantizeLinear", ["w_q", "w_scale"], ["w_dq"], axis=0),
        helper.make_node("DequantizeLinear", ["bias_q", "bias_scale", "bias_zp"],
                         ["b_dq"], axis=0),
        helper.make_node("Conv", ["x_dq", "w_dq", "b_dq"], ["conv_out"], name="conv"),
        helper.make_node("QuantizeLinear", ["conv_out", "y_scale", "y_zp"], ["y_q"],
                         output_dtype=TensorProto.UINT8),
        helper.make_node("DequantizeLinear", ["y_q", "y_scale", "y_zp"], ["y"]),
    ]

    graph = helper.make_graph(
        nodes, "qdq_conv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, channels, 6, 6])],
        initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def run_session(model, level, providers=(CPU,)):
    """Run the model in a fresh session at one optimisation level."""
    data = np.random.default_rng(1).standard_normal((1, 3, 8, 8)).astype(np.float32)
    with tempfile.TemporaryDirectory() as temporary:
        path = os.path.join(temporary, "model.onnx")
        onnx.save(model, path)
        session, _ = create_session(path, list(providers),
                                    require=providers[0],
                                    optimization_level=level)
        return session.run(None, {"x": data})[0].copy(), data


def fused_conv_count(model, level):
    """QLinearConv nodes in the optimised graph, or None if no dump was written."""
    with tempfile.TemporaryDirectory() as temporary:
        path = os.path.join(temporary, "model.onnx")
        dump = os.path.join(temporary, "optimised.onnx")
        onnx.save(model, path)
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.optimized_model_filepath = dump
        ort.InferenceSession(path, sess_options=options, providers=[CPU])
        graph = onnx.load(dump).graph
        return sum(1 for node in graph.node if node.op_type == "QLinearConv")


class FusionHappensAtExtendedOnly(unittest.TestCase):
    def test_extended_fuses_and_basic_does_not(self):
        model = build_qdq_conv()
        self.assertEqual(fused_conv_count(
            model, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC), 0)
        self.assertEqual(fused_conv_count(
            model, ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED), 1)
        self.assertEqual(fused_conv_count(
            model, ort.GraphOptimizationLevel.ORT_ENABLE_ALL), 1)


class ReferenceAgreement(unittest.TestCase):
    def reference_output(self, model, inputs):
        return ReferenceEvaluator(model).run(None, {"x": inputs})[0]

    def test_conforming_bias_scale_matches_reference_at_every_level(self):
        model = build_qdq_conv(bias_scale_factor=1.0, bias_zero_point=0)
        for level, name in ((ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "BASIC"),
                            (ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "ALL")):
            output, inputs = run_session(model, level)
            reference = self.reference_output(model, inputs)
            deviation = float(np.abs(output - reference).max())
            self.assertLess(deviation, 1e-4,
                            "%s deviates from the reference by %g" % (name, deviation))

    def test_nonzero_bias_zero_point_still_matches_at_basic(self):
        model = build_qdq_conv(bias_scale_factor=1.0, bias_zero_point=-7)
        output, inputs = run_session(model, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        reference = self.reference_output(model, inputs)
        self.assertLess(float(np.abs(output - reference).max()), 1e-4)

    def test_mismatched_bias_scale_is_wrong_when_fused(self):
        """The defect itself, pinned to versions that still have it."""
        if not DEFECT_EXPECTED:
            self.skipTest("ORT %s is expected to reject the fusion" % ORT_VERSION)
        model = build_qdq_conv(bias_scale_factor=2.0, bias_zero_point=0)
        basic, inputs = run_session(model, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        fused, _ = run_session(model, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        reference = self.reference_output(model, inputs)
        self.assertLess(float(np.abs(basic - reference).max()), 1e-4,
                        "the unfused path must be correct")
        self.assertGreater(float(np.abs(fused - reference).max()), 1e-3,
                           "ORT %s should still show the fusion defect" % ORT_VERSION)

    def test_mismatched_bias_scale_with_nonzero_zero_point(self):
        if not DEFECT_EXPECTED:
            self.skipTest("ORT %s is expected to reject the fusion" % ORT_VERSION)
        model = build_qdq_conv(bias_scale_factor=0.5, bias_zero_point=11)
        basic, inputs = run_session(model, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        reference = self.reference_output(model, inputs)
        self.assertLess(float(np.abs(basic - reference).max()), 1e-4)


class ProjectModelParity(unittest.TestCase):
    """CPU BASIC against CUDA on the real quantised model, when both exist."""

    INT8 = os.path.join("outputs", "quant", "oscd_baseline_seed42_v2",
                        "onboard_int8.onnx")

    def test_cpu_basic_agrees_with_cuda(self):
        if CUDA not in ort.get_available_providers():
            self.skipTest("CUDAExecutionProvider is not available")
        if not os.path.isfile(self.INT8):
            self.skipTest("quantised model not present: %s" % self.INT8)
        rng = np.random.default_rng(0)
        feeds = {"image1": rng.standard_normal((1, 4, 128, 128)).astype(np.float32),
                 "image2": rng.standard_normal((1, 4, 128, 128)).astype(np.float32)}
        cpu_session, _ = create_session(
            self.INT8, [CPU], require=CPU,
            optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)
        cuda_session, _ = create_session(
            self.INT8, [CUDA, CPU], require=CUDA, preload_cuda=True,
            optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        cpu = cpu_session.run(None, feeds)[0].astype(np.float64)
        cuda = cuda_session.run(None, feeds)[0].astype(np.float64)
        mae = float(np.abs(cpu - cuda).mean())
        correlation = float(np.corrcoef(cpu.ravel(), cuda.ravel())[0, 1])
        self.assertLess(mae, 0.02, "CPU BASIC and CUDA differ by MAE %g" % mae)
        self.assertGreater(correlation, 0.98,
                           "CPU BASIC and CUDA correlate only %.4f" % correlation)

    def test_cpu_extended_is_rejected_by_the_workaround(self):
        """Guard that the workaround is actually in force in the tooling."""
        import quantize
        self.assertEqual(quantize.CPU_QDQ_OPTIMISATION,
                         ort.GraphOptimizationLevel.ORT_ENABLE_BASIC)


if __name__ == "__main__":
    unittest.main()
