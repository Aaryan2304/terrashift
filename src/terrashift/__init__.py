"""TerraShift, bi-temporal Sentinel-2 man-made change screening.

The reusable implementation lives in this package. The former top-level ``data/``,
``models/``, ``utils/`` and ``preprocess/`` directories were migrated here, so nothing at the
repository root is importable as a library any more. The root scripts are thin CLI entry
points, and ``import terrashift`` stays cheap: nothing heavy is imported here.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]


