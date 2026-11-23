"""Data access for TerraShift: manifests, SAFE loading, masks, validity and datasets.

This package holds the data-access layer only. Registration and normalisation stay in the
legacy top-level ``data/`` directory until Task 5, along with the two modules that depend on
them (``local_patches``, ``validation_windows``).

Nothing is imported here on purpose: importing ``terrashift.data`` must not pull in torch,
rasterio or any concrete module, so a caller pays only for what it uses.
"""
