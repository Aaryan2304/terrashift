"""Inference: sliding-window prediction, checkpoint loading, ONNX export and quantisation.

The root-level scripts (``predict_safe_pair.py``, ``quantize.py``, ``predict_folder.py``) remain
CLI entry points and import from here. Evaluation helpers stay in ``quantize.py`` until the
evaluation migration moves them.
"""
