"""Canonical patch-grid origin generation.

Kept in its own module so patch sampling does not depend on the OSCD dataset module,
which imports torch and rasterio. Every other name for this function re-exports this
object.
"""

from __future__ import annotations

__all__ = ["patch_origins"]

def patch_origins(length: int, patch_size: int, stride: int) -> list[int]:
    """Return unique windows that cover the full dimension, including its end."""
    if patch_size <= 0 or stride <= 0 or stride > patch_size:
        raise ValueError("patch size and stride must be positive, with stride <= patch size")
    if length < patch_size:
        return []
    origins = list(range(0, length - patch_size + 1, stride))
    edge = length - patch_size
    if not origins or origins[-1] != edge:
        origins.append(edge)
    return sorted(set(origins))
