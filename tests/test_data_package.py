"""Contract tests for the migrated data package.

These assert the *packaging* contract: the modules live inside ``terrashift.data``, the
constants that define the data interface are unchanged, and the mask/validity semantics that
the loaders promise still hold. They use synthetic arrays and the checked-in manifest, so no
Sentinel-2 dataset is required.
"""

from __future__ import annotations

import os
import pathlib
import unittest
from unittest import mock

import numpy as np

import terrashift.data.grids as grids
import terrashift.data.manifest as manifest
import terrashift.data.masks as masks
import terrashift.data.oscd_original as oscd_original
import terrashift.data.safeloader as safeloader
import terrashift.data.validity as validity

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "terrashift" / "data"


class PackagePathTests(unittest.TestCase):
    """The modules must resolve inside the installed package, not the legacy directory."""

    def test_modules_resolve_into_the_package(self):
        for module in (grids, manifest, masks, oscd_original, safeloader, validity):
            path = pathlib.Path(module.__file__).resolve()
            self.assertEqual(path.parent, PACKAGE,
                             "%s is not served from the package" % module.__name__)
            self.assertTrue(module.__name__.startswith("terrashift.data."),
                            "%s has a legacy module name" % module.__name__)

    def test_legacy_directory_holds_no_migrated_copy(self):
        for name in ("manifest", "safeloader", "masks", "validity", "grids",
                     "dataset", "augment", "oscd_original", "osm"):
            legacy = REPO / "data" / ("%s.py" % name)
            self.assertFalse(legacy.exists(),
                             "duplicate implementation left at data/%s.py" % name)


class BandOrderTests(unittest.TestCase):
    """Band order is a data contract: changing it silently corrupts every model input."""

    def test_default_bands_are_the_four_used_by_the_model(self):
        self.assertEqual(validity.DEFAULT_BANDS, ("B02", "B03", "B04", "B08"))
        self.assertEqual(oscd_original.DEFAULT_BANDS, ("B02", "B03", "B04", "B08"))

    def test_full_band_order_is_the_thirteen_l1c_bands(self):
        order = list(safeloader.BAND_ORDER)
        self.assertEqual(len(order), 13, "L1C carries 13 bands")
        self.assertEqual(order[0], "B01")
        for band in ("B02", "B03", "B04", "B08", "B8A", "B11", "B12"):
            self.assertIn(band, order)


class MaskConstantTests(unittest.TestCase):
    """The scene-classification and quality flag names map onto MSK band order."""

    def test_classi_bands(self):
        self.assertEqual(masks.CLASSI_BANDS, ["opaque_cloud", "cirrus", "snow_ice"])

    def test_qualit_bands_keep_their_names(self):
        self.assertIn("msi_lost", masks.QUALIT_BANDS)
        self.assertGreaterEqual(len(masks.QUALIT_BANDS), 8)

    def test_crs_constant_is_the_rasterio_type(self):
        # `masks.CRS` is the rasterio CRS class used for annotations, not an EPSG string.
        from rasterio.crs import CRS as RasterioCRS
        self.assertIs(masks.CRS, RasterioCRS)


class GridContractTests(unittest.TestCase):
    """Grid is the raster contract every loader and mask agrees on."""

    def test_grid_fields(self):
        self.assertEqual(grids.Grid._fields, ("crs", "transform", "width", "height"))

    def test_grid_preserves_crs_and_transform(self):
        from affine import Affine
        grid = grids.Grid(crs="EPSG:32643", transform=Affine(10, 0, 500000, 0, -10, 1000000),
                          width=128, height=64)
        self.assertEqual(str(grid.crs), "EPSG:32643")
        self.assertEqual(grid.transform.a, 10)
        self.assertEqual((grid.width, grid.height), (128, 64))


class PairedValidityTests(unittest.TestCase):
    """paired_validity is the observation-validity contract for a bi-temporal pair."""

    def test_both_dates_must_be_valid(self):
        a = np.array([[True, True], [False, True]])
        b = np.array([[True, False], [True, True]])
        paired = validity.paired_validity(a, b)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(paired, expected)

    def test_coverage_masks_out_the_warp_border(self):
        a = np.ones((2, 2), dtype=bool)
        coverage = np.array([[True, False], [True, True]])
        paired = validity.paired_validity(a, a, coverage=coverage)
        np.testing.assert_array_equal(paired, coverage)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            validity.paired_validity(np.ones((2, 2), bool), np.ones((2, 3), bool))
        with self.assertRaises(ValueError):
            validity.paired_validity(np.ones((2, 2), bool), np.ones((2, 2), bool),
                                     coverage=np.ones((3, 3), bool))


class ManifestTests(unittest.TestCase):
    """Manifest parsing must keep the pair/tile contract without needing the imagery."""

    def setUp(self):
        self.path = REPO / "configs" / "data_manifest.yaml"
        if not self.path.exists():
            self.skipTest("data manifest not present")
        # load_manifest expands ${SENTINEL_DATA_ROOT}; a dummy root is enough to test the
        # parse and the pair/tile contract without any imagery on disk.
        self._env = mock.patch.dict(os.environ, {"SENTINEL_DATA_ROOT": "/tmp/terrashift-unused"})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_pairs_and_tiles(self):
        loaded = manifest.load_manifest(str(self.path))
        pairs = manifest.all_pairs(loaded)
        self.assertEqual(sorted(pairs), ["A", "B"])
        for pair in pairs:
            config = manifest.pair_config(loaded, pair)
            self.assertIn("tile", config)
            self.assertEqual(config["crs"], "EPSG:32643")

    def test_product_map_is_keyed_by_id(self):
        products = manifest.product_map(manifest.load_manifest(str(self.path)))
        self.assertEqual(len(products), 4, "two pairs, two dates each")
        for product in products.values():
            self.assertTrue(product["id"])


if __name__ == "__main__":
    unittest.main()
