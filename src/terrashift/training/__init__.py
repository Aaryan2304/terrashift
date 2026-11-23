"""Training utilities shared by the three training entry points.

Only helpers that were duplicated or self-contained are collected here. Each entry point keeps
its own epoch loop, because the three workflows differ in objective, data pipeline and
checkpoint-selection rule and cannot be validated against each other without training runs.
"""
