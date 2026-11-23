"""ONNX Runtime session construction with provider verification.

ONNX Runtime does not load the CUDA libraries itself. When they are missing it drops the requested
provider, logs one line and builds a CPU-only session, so inference runs silently at CPU speed. That
is worse than a crash, because a benchmark then records a plausible wrong number.

Two guards. ``preload_cuda_libraries()`` before the first session loads CUDA 12 and cuDNN 9 from the
``nvidia-*`` packages without importing PyTorch. ``create_session(..., require=...)`` raises when the
provider actually built is not the one requested.
"""

import onnxruntime as ort

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def preload_cuda_libraries(directory=None):
    """Load the CUDA and cuDNN libraries ONNX Runtime does not load itself.

    Returns True when the preload call exists and was made, False when this
    ONNX Runtime build has no ``preload_dlls``. The call is safe to repeat.
    """
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return False
    if directory is None:
        preload()
    else:
        preload(directory=directory)
    return True


def require_provider(active, required):
    """Raise when a requested execution provider is not in the session.

    ``active`` is ``session.get_providers()``. The message names the provider,
    what did run instead, and the usual cause, because the log line that ONNX
    Runtime emits is easy to miss.
    """
    if required in active:
        return
    raise RuntimeError(
        "requested execution provider %r is not active; the session was built "
        "with %r. ONNX Runtime drops a provider silently when its libraries are "
        "missing, so any timing or accuracy figure from this session would be a "
        "CPU result mislabelled as GPU. Call preload_cuda_libraries() first, or "
        "import torch before creating the session."
        % (required, list(active)))


def create_session(model_path, providers, sess_options=None, require=None,
                   preload_cuda=False, directory=None, optimization_level=None):
    """Build a session and verify the providers it actually got.

    Returns ``(session, active_providers)``. Pass ``require`` to fail loudly
    instead of accepting a silent fallback, and ``optimization_level`` to pin the
    graph optimiser.

    Pinning matters for QDQ models on CPU: ORT 1.24.4's QDQ selector fuses Conv
    into QLinearConv at ORT_ENABLE_EXTENDED and above, and that fusion produces
    wrong output when a convolution's bias quantisation does not satisfy the
    ONNX QLinearConv invariant ``bias_scale == x_scale * w_scale``. See upstream
    issue 24711 and the fix in PR 28229, and
    ``docs/decisions/009-cpu-qdq-optimiser-defect.md`` here. Models quantised
    from a shared Siamese bias hit this.
    """
    if preload_cuda:
        preload_cuda_libraries(directory)
    options = sess_options
    if options is None:
        options = ort.SessionOptions()
    if optimization_level is not None:
        options.graph_optimization_level = optimization_level
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=list(providers))
    active = list(session.get_providers())
    if require is not None:
        require_provider(active, require)
    return session, active


OPTIMISATION_LABELS = {
    ort.GraphOptimizationLevel.ORT_DISABLE_ALL: "ORT_DISABLE_ALL",
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC: "ORT_ENABLE_BASIC",
    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED: "ORT_ENABLE_EXTENDED",
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL: "ORT_ENABLE_ALL",
}


def optimisation_label(level):
    """Readable name for a graph optimisation level, for reports."""
    return OPTIMISATION_LABELS.get(level, str(level))
