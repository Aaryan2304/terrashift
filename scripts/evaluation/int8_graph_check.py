"""Validate the generated INT8 graph and smoke-test it on ONNX Runtime CPU.

Graph checks only: node census, Q/DQ counts, custom or non-standard operators, and a single
batch-1 inference at 256x256. No validation-set inference and no benchmarking.

    python scripts/evaluation/int8_graph_check.py \
        --quant-dir outputs/quant/local_manual_v2_seed43_posweight_0p834_int8_v1 \
        --report outputs/evaluation/int8_quantisation_execution_v1/int8_graph_check.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantize import CPU_QDQ_OPTIMISATION, INPUT_NAMES, OUTPUT_NAME, PATCH_SIZE
from terrashift.inference.onnx_runtime import CPU, create_session, optimisation_label

#: Operators that are standard ONNX and expected in a QDQ graph of this model.
EXPECTED_OPS = {
    "Conv", "ConvTranspose", "MaxPool", "Identity", "Sub", "Abs", "Concat", "Sigmoid",
    "Add", "Mul", "Relu", "Reshape", "Transpose", "Flatten", "Gemm",
    "QuantizeLinear", "DequantizeLinear",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    fp32_path = Path(args.quant_dir) / "onboard_fp32.onnx"
    int8_path = Path(args.quant_dir) / "onboard_int8.onnx"
    model = onnx.load(str(int8_path))
    onnx.checker.check_model(model)

    census = collections.Counter(node.op_type for node in model.graph.node)
    domains = sorted({node.domain or "ai.onnx" for node in model.graph.node})
    unexpected = sorted(op for op in census if op not in EXPECTED_OPS)
    q_dq = {op: int(census[op]) for op in ("QuantizeLinear", "DequantizeLinear")
            if op in census}

    def dims(value):
        return [dim.dim_value or dim.dim_param for dim in value.type.tensor_type.shape.dim]

    session, providers = create_session(int8_path, [CPU], require=CPU,
                                       optimization_level=CPU_QDQ_OPTIMISATION)
    feed = {name: np.random.rand(1, 4, PATCH_SIZE, PATCH_SIZE).astype(np.float32)
            for name in INPUT_NAMES}
    output = np.asarray(session.run(None, feed)[0])

    payload = {
        "int8_path": str(int8_path), "int8_sha256": sha256(int8_path),
        "int8_bytes": int8_path.stat().st_size,
        "fp32_path": str(fp32_path), "fp32_sha256": sha256(fp32_path),
        "fp32_bytes": fp32_path.stat().st_size,
        "size_ratio": round(int8_path.stat().st_size / fp32_path.stat().st_size, 4),
        "checker_passed": True,
        "opset": [(imp.domain or "ai.onnx", imp.version) for imp in model.opset_import],
        "nodes": len(model.graph.node),
        "op_census": dict(sorted(census.items(), key=lambda item: -item[1])),
        "quantisation_nodes": q_dq,
        "domains_present": domains,
        "unexpected_operators": unexpected,
        "custom_or_nonstandard_domain": [domain for domain in domains
                                         if domain not in ("ai.onnx", "")],
        "graph_inputs": {value.name: dims(value) for value in model.graph.input},
        "graph_outputs": {value.name: dims(value) for value in model.graph.output},
        "runtime": {
            "providers": providers,
            "optimisation": optimisation_label(CPU_QDQ_OPTIMISATION),
            "batch": 1, "patch": PATCH_SIZE,
            "output_shape": list(output.shape),
            "expected_shape": [1, 1, PATCH_SIZE, PATCH_SIZE],
            "shape_ok": list(output.shape) == [1, 1, PATCH_SIZE, PATCH_SIZE],
            "finite": bool(np.isfinite(output).all()),
            "in_probability_range": bool(output.min() >= 0.0 and output.max() <= 1.0),
            "min": float(output.min()), "max": float(output.max()),
            "mean": float(output.mean()),
        },
    }
    payload["passed"] = bool(
        payload["checker_passed"] and not payload["unexpected_operators"]
        and not payload["custom_or_nonstandard_domain"]
        and payload["runtime"]["shape_ok"] and payload["runtime"]["finite"]
        and payload["runtime"]["in_probability_range"]
        and payload["quantisation_nodes"].get("QuantizeLinear", 0) > 0
        and payload["quantisation_nodes"].get("DequantizeLinear", 0) > 0)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))
    if not payload["passed"]:
        raise SystemExit("INT8 graph check FAILED")


if __name__ == "__main__":
    main()
