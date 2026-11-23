"""Regression tests for the paired observation-validity chain.

These tests build small synthetic rasters and a synthetic product stub. They
prove the control flow and the fail-closed behaviour. They do not prove that
the real Sentinel-2 quality masks look the way the specification says; that
check needs the supplied products and is recorded as NOT RUN in the task
report.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_geom

from terrashift.data.grids import (Grid, grid_mismatches, require_expected_crs,
                        require_same_grid)
from terrashift.data.masks import QualityMasks
from terrashift.registration import coverage_mask, estimate_shift
from terrashift.data.validity import (observation_valid, paired_validity,
                           registration_coverage)
from terrashift.preprocessing.normalize import reference_scaled_change_score
from terrashift.preprocessing.water import water_masks
from scripts.utilities.prepare_local_partitions import build_pair
from scripts.review.rasterize_manual_labels import rasterize_annotations
from scripts.utilities.run_classical_baseline import candidate_mask_from_score

CRS_UTM = CRS.from_epsg(32643)
X0, Y0 = 500000.0, 3000000.0


def _write_bands(path, array, transform, crs, dtype="uint8"):
    """Write a 2-D array as one band, or a 3-D array as (bands, rows, cols)."""
    array = np.asarray(array)
    count = 1 if array.ndim == 2 else array.shape[0]
    height, width = array.shape[-2], array.shape[-1]
    with rasterio.open(path, "w", driver="GTiff", height=height, width=width,
                       count=count, dtype=dtype, crs=crs,
                       transform=transform) as ds:
        if count == 1:
            ds.write(array.astype(dtype), 1)
        else:
            ds.write(array.astype(dtype))


class _FakeProduct:
    """Minimal stand-in for SafeProduct with pre-built reflectance bands."""

    def __init__(self, granule, bands, transform, crs):
        self.granule = str(granule)
        self.bands = bands
        self.transform = transform
        self.crs = crs

    def band_path(self, band):
        return None

    def read_band(self, band, window=None, out_shape=None, resampling=None):
        array = self.bands[band]
        if out_shape is not None and tuple(out_shape) != array.shape:
            raise AssertionError("unexpected out_shape %s for band %s"
                                 % (out_shape, band))
        return array, self.transform, self.crs


def _grid(rows=8, cols=8, pixel=20.0):
    return Grid(CRS_UTM, Affine(pixel, 0.0, X0, 0.0, -pixel, Y0), cols, rows)


def _quality_dir(root, classi_shape=(4, 4), classi_pixel=40.0,
                 qualit_shape=(16, 16), qualit_pixel=10.0,
                 classi_flags=None, qualit_flags=None):
    """Create a QI_DATA directory with MSK_CLASSI and MSK_QUALIT files."""
    qi = Path(root) / "QI_DATA"
    qi.mkdir(parents=True, exist_ok=True)

    classi = np.zeros((3,) + classi_shape, dtype=np.uint8)
    for index, value in (classi_flags or {}).items():
        classi[index] = value
    _write_bands(qi / "MSK_CLASSI_B00.jp2", classi,
                 Affine(classi_pixel, 0.0, X0, 0.0, -classi_pixel, Y0), CRS_UTM)

    qualit = np.zeros((8,) + qualit_shape, dtype=np.uint8)
    for index, value in (qualit_flags or {}).items():
        qualit[index] = value
    for band in ("B02", "B03", "B04", "B08"):
        _write_bands(qi / ("MSK_QUALIT_%s.jp2" % band), qualit,
                     Affine(qualit_pixel, 0.0, X0, 0.0, -qualit_pixel, Y0), CRS_UTM)
    return qi


def _bands(shape=(8, 8), value=0.2):
    return {band: np.full(shape, value, dtype=np.float32)
            for band in ("B02", "B03", "B04", "B08")}


class QualityMaskStructureTests(unittest.TestCase):
    def test_classi_requires_three_bands(self):
        with tempfile.TemporaryDirectory() as temp:
            qi = _quality_dir(temp)
            _write_bands(qi / "MSK_CLASSI_B00.jp2", np.zeros((2, 4, 4), np.uint8),
                         Affine(40.0, 0.0, X0, 0.0, -40.0, Y0), CRS_UTM)
            product = _FakeProduct(qi.parent, _bands(), _grid().transform, CRS_UTM)
            with self.assertRaisesRegex(ValueError, "defines 3"):
                QualityMasks(product).read_classi()

    def test_qualit_requires_eight_bands(self):
        with tempfile.TemporaryDirectory() as temp:
            qi = _quality_dir(temp)
            _write_bands(qi / "MSK_QUALIT_B08.jp2", np.zeros((3, 16, 16), np.uint8),
                         Affine(10.0, 0.0, X0, 0.0, -10.0, Y0), CRS_UTM)
            product = _FakeProduct(qi.parent, _bands(), _grid().transform, CRS_UTM)
            with self.assertRaisesRegex(ValueError, "defines 8"):
                QualityMasks(product).read_qualit("B08")

    def test_missing_quality_mask_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            qi = _quality_dir(temp)
            os.remove(qi / "MSK_CLASSI_B00.jp2")
            product = _FakeProduct(qi.parent, _bands(), _grid().transform, CRS_UTM)
            with self.assertRaises(FileNotFoundError):
                QualityMasks(product).read_classi()


class ObservationValidityTests(unittest.TestCase):
    def test_cloud_cirrus_saturated_and_nodata_are_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            grid = _grid()
            classi_cloud = np.zeros((4, 4), np.uint8)
            classi_cloud[1, 1] = 1          # covers grid rows/cols 2-3
            qualit_saturated = np.zeros((16, 16), np.uint8)
            qualit_saturated[9, 9] = 1      # covers grid pixel (4, 4)
            qi = _quality_dir(temp,
                              classi_flags={0: classi_cloud},
                              qualit_flags={7: qualit_saturated})
            bands = _bands()
            bands["B08"][6, 6] = np.nan     # source NoData
            product = _FakeProduct(qi.parent, bands, grid.transform, CRS_UTM)

            valid, detail = observation_valid(product, grid)

            for row, col in ((2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (6, 6)):
                self.assertFalse(valid[row, col],
                                 "pixel (%d, %d) should be invalid" % (row, col))
            self.assertTrue(valid[0, 0])
            self.assertEqual(int((~valid).sum()), 6)
            self.assertEqual(detail["classi_opaque_cloud"], 4)
            self.assertEqual(detail["total_invalid"], 6)

    def test_date2_invalid_excludes_pixel_from_pair(self):
        date1 = np.ones((4, 4), dtype=bool)
        date2 = np.ones((4, 4), dtype=bool)
        date2[1, 2] = False
        paired = paired_validity(date1, date2)
        self.assertFalse(paired[1, 2])
        self.assertEqual(int(paired.sum()), 15)

    def test_dynamic_shoreline_is_valid_water_state(self):
        ndwi1 = np.array([[0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
        ndwi2 = np.array([[-0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
        observation = np.ones((2, 2), dtype=bool)
        masks = water_masks(ndwi1, ndwi2, observation_valid=observation)
        # Water on date 1 and land on date 2 is a reclamation candidate: it is
        # a water-state class, not an invalid observation.
        self.assertTrue(masks["dynamic_shoreline"][0, 0])
        self.assertTrue(masks["valid"][0, 0])

    def test_flag_outside_validity_cannot_be_reintroduced_by_water(self):
        ndwi1 = np.full((2, 2), 0.5, dtype=np.float32)
        ndwi2 = np.full((2, 2), 0.5, dtype=np.float32)
        observation = np.ones((2, 2), dtype=bool)
        observation[0, 1] = False
        masks = water_masks(ndwi1, ndwi2, observation_valid=observation)
        self.assertFalse(masks["valid"][0, 1])
        self.assertFalse(masks["stable_water"][0, 1])
        self.assertFalse(masks["land"][0, 1])


class RegistrationCoverageTests(unittest.TestCase):
    def test_shifted_border_is_not_coverage(self):
        shape = (16, 16)
        covered = coverage_mask(shape, 3.0, 0.0)
        # A positive displacement moves content right, so the left columns
        # receive no source pixel and must be invalid.
        self.assertFalse(covered[0, 0])
        self.assertFalse(covered[0, 2])
        self.assertTrue(covered[0, 3])
        self.assertTrue(covered[0, 15])
        self.assertLess(int(covered.sum()), 16 * 16)
        self.assertEqual(int(covered.sum()), 16 * 13)

    def test_validity_alias_matches_register(self):
        np.testing.assert_array_equal(
            registration_coverage((10, 12), 2.0, -1.0),
            coverage_mask((10, 12), 2.0, -1.0))

    def test_masked_ecc_ignores_flagged_pixels(self):
        rng = np.random.default_rng(0)
        y, x = np.mgrid[0:160, 0:160].astype(np.float32)
        base = (np.sin(x / 6.0) + np.cos(y / 4.0)
                + 0.2 * rng.normal(0, 1, (160, 160))).astype(np.float32)
        from terrashift.registration import apply_shift
        moved = apply_shift(base, 2.0, -1.0)

        corrupt = moved.copy()
        corrupt[0:60, 0:60] = 9.0        # a synthetic bright cloud
        validity = np.ones((160, 160), dtype=bool)
        validity[0:60, 0:60] = False

        unmasked = estimate_shift(base, corrupt)
        masked = estimate_shift(base, corrupt, mov_valid=validity)
        self.assertTrue(masked["converged"])
        self.assertLess(abs(masked["dx"] - 2.0), 0.6)
        self.assertLess(abs(masked["dy"] + 1.0), 0.6)
        # The masked estimate must be at least as accurate as the unmasked one.
        # The unmasked estimate may fail to converge at all, which also counts
        # as "not better".
        if unmasked["converged"]:
            self.assertLessEqual(abs(masked["dx"] - 2.0),
                                 abs(unmasked["dx"] - 2.0) + 1e-9)


class GridContractTests(unittest.TestCase):
    def test_same_grid_passes(self):
        grid = _grid()
        self.assertEqual(grid_mismatches(grid, grid), [])
        require_same_grid(grid, grid)

    def test_origin_shape_and_resolution_mismatches_raise(self):
        grid = _grid()
        shifted = grid._replace(transform=Affine(20.0, 0.0, X0 + 20.0,
                                                 0.0, -20.0, Y0))
        smaller = grid._replace(width=4)
        coarse = grid._replace(transform=Affine(40.0, 0.0, X0, 0.0, -40.0, Y0))
        for bad, reason in ((shifted, "origin"), (smaller, "shape"), (coarse, "resolution")):
            with self.assertRaisesRegex(ValueError, reason):
                require_same_grid(bad, grid, "candidate", "reference")

    def test_expected_crs_mismatch_raises(self):
        grid = _grid()
        require_expected_crs(grid, "EPSG:32643")
        with self.assertRaisesRegex(ValueError, "expected"):
            require_expected_crs(grid, "EPSG:32644")


class ScoreAndCandidateTests(unittest.TestCase):
    def test_score_is_nan_outside_valid(self):
        rng = np.random.default_rng(3)
        ref = [rng.random((4, 4)).astype(np.float32)]
        mov = [rng.random((4, 4)).astype(np.float32)]
        valid = np.ones((4, 4), dtype=bool)
        valid[2, 2] = False
        score, complete = reference_scaled_change_score(ref, mov, valid)
        self.assertTrue(np.isnan(score[2, 2]))
        self.assertFalse(complete[2, 2])
        self.assertTrue(np.isfinite(score[0, 0]))

    def test_zero_spread_fails_instead_of_returning_a_number(self):
        ref = [np.full((4, 4), 0.3, np.float32)]
        mov = [np.full((4, 4), 0.3, np.float32)]
        with self.assertRaisesRegex(ValueError, "spread"):
            reference_scaled_change_score(ref, mov, np.ones((4, 4), dtype=bool))

    def test_candidate_mask_never_marks_invalid_pixels(self):
        # 100 x 100 keeps the synthetic positive block below 1% of the valid
        # area, which is the regime the real 99th-percentile threshold uses.
        rng = np.random.default_rng(1)
        score = rng.random((100, 100)).astype(np.float32)
        score[50:56, 50:56] = 4.0
        valid = np.ones((100, 100), dtype=bool)
        valid[0:10, 0:10] = False
        score[0:10, 0:10] = 100.0        # huge scores inside the invalid block
        mask, _ = candidate_mask_from_score(score, valid, 99.0)
        self.assertFalse(mask[0:10, 0:10].any())
        self.assertTrue(mask[51:55, 51:55].all())
        self.assertTrue(mask.any())

    def test_candidate_mask_rejects_empty_valid(self):
        score = np.zeros((4, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            candidate_mask_from_score(score, np.zeros((4, 4), dtype=bool), 99.0)


class WindowValidityTests(unittest.TestCase):
    def test_invalid_window_aligns_masks_to_the_image_window(self):
        with tempfile.TemporaryDirectory() as temp:
            classi_flag = np.zeros((4, 4), np.uint8)
            classi_flag[0, 0] = 1          # covers x 500000-500040, y 2999960-3000000
            qualit_flag = np.zeros((16, 16), np.uint8)
            qualit_flag[2, 5] = 1          # covers x 500050-500060, y 2999970-2999980
            qi = _quality_dir(temp,
                              classi_flags={0: classi_flag},
                              qualit_flags={7: qualit_flag})
            product = _FakeProduct(qi.parent, _bands(), _grid().transform, CRS_UTM)
            masks = QualityMasks(product)
            transform = Affine(10.0, 0.0, X0, 0.0, -10.0, Y0)
            invalid, detail = masks.invalid_window(
                transform, (12, 12), CRS_UTM, qualit_bands=("B08",))

            self.assertTrue(invalid[0:4, 0:4].all())
            self.assertTrue(invalid[2, 5])
            self.assertEqual(int(invalid.sum()), 17)
            self.assertEqual(detail["classi_opaque_cloud"], 16)
            self.assertEqual(detail["total_pixels"], 144)

    def test_invalid_window_rejects_a_window_outside_the_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            qi = _quality_dir(temp)
            product = _FakeProduct(qi.parent, _bands(), _grid().transform, CRS_UTM)
            far_transform = Affine(10.0, 0.0, X0 + 100000.0, 0.0, -10.0, Y0 - 100000.0)
            with self.assertRaisesRegex(ValueError, "outside"):
                QualityMasks(product).invalid_window(
                    far_transform, (8, 8), CRS_UTM, qualit_bands=("B08",))


class AnnotationValidityTests(unittest.TestCase):
    def _feature(self, box_utm=(X0, Y0 - 100.0, X0 + 100.0, Y0)):
        x0, y0, x1, y1 = box_utm
        geom = {"type": "Polygon", "coordinates": [[
            [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
        wgs = transform_geom("EPSG:32643", "EPSG:4326", geom, precision=8)
        return {"type": "Feature", "geometry": wgs, "properties": {
            "record_id": "r1", "pair": "A", "label_class": "change",
            "reviewer": "tester", "reviewed_at": "2026-11-14",
            "evidence_source": "before_after_visual",
            "evidence": "synthetic fixture for a guard test",
            "confidence": "high"}}

    def _labels(self, root, feature):
        path = Path(root) / "labels.geojson"
        path.write_text(json.dumps(
            {"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
        return path

    def _raster_fixture(self, root):
        """20 m score/validity pair and a 10 m partition over the same extent."""
        scores = Path(root) / "scores"
        parts = Path(root) / "parts"
        scores.mkdir()
        parts.mkdir()
        score = np.full((20, 20), -9999.0, dtype=np.float32)
        score[:, 10:] = 1.0
        with rasterio.open(scores / "change_score_A.tif", "w", driver="GTiff",
                           height=20, width=20, count=1, dtype="float32",
                           crs=CRS_UTM, transform=Affine(20.0, 0.0, X0, 0.0, -20.0, Y0),
                           nodata=-9999.0) as ds:
            ds.write(score, 1)
        validity = np.zeros((20, 20), dtype=np.uint8)
        validity[:, 10:] = 1
        _write_bands(scores / "observation_valid_A.tif", validity,
                     Affine(20.0, 0.0, X0, 0.0, -20.0, Y0), CRS_UTM)
        with rasterio.open(scores / "observation_valid_A.tif", "r+") as ds:
            ds.nodata = 0
        _write_bands(parts / "pair_A_partition_10m.tif", np.ones((40, 40), np.uint8),
                     Affine(10.0, 0.0, X0, 0.0, -10.0, Y0), CRS_UTM)
        return scores, parts

    def test_rasterizer_uses_validity_raster_and_reprojects_to_10m(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores, parts = self._raster_fixture(root)
            # Valid half of the tile: x from X0+200 to X0+400.
            labels = self._labels(root, self._feature(
                (X0 + 220.0, Y0 - 220.0, X0 + 320.0, Y0 - 120.0)))
            result = rasterize_annotations("A", labels, root / "out", parts, scores,
                                           validity_dir=scores,
                                           expected_crs="EPSG:32643")
            self.assertEqual(result["feature_count"], 1)
            with rasterio.open(root / "out" / "manual_labels_A.tif") as ds:
                mask = ds.read(1)
                self.assertEqual(ds.tags()["validity_raster"],
                                 "observation_valid_A.tif")
            self.assertEqual(set(np.unique(mask).tolist()) <= {0, 1, 255}, True)
            self.assertGreater(int((mask == 1).sum()), 0)

    def test_rasterizer_rejects_polygon_over_invalid_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores, parts = self._raster_fixture(root)
            # Invalid half of the tile: x below X0+200.
            labels = self._labels(root, self._feature(
                (X0 + 20.0, Y0 - 220.0, X0 + 120.0, Y0 - 120.0)))
            with self.assertRaisesRegex(ValueError, "not valid observations"):
                rasterize_annotations("A", labels, root / "out", parts, scores,
                                      validity_dir=scores,
                                      expected_crs="EPSG:32643")

    def test_rasterizer_rejects_invalid_without_blaming_water(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores, parts = self._raster_fixture(root)
            labels = self._labels(root, self._feature(
                (X0 + 20.0, Y0 - 220.0, X0 + 120.0, Y0 - 120.0)))
            with self.assertRaises(ValueError) as caught:
                rasterize_annotations("A", labels, root / "out", parts, scores,
                                      validity_dir=scores,
                                      expected_crs="EPSG:32643")
            message = str(caught.exception)
            self.assertIn("not valid observations", message)
            self.assertNotIn("water", message.lower())

    def test_rasterizer_requires_validity_raster(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.geojson"
            labels.write_text(json.dumps(
                {"type": "FeatureCollection", "features": [self._feature()]}),
                encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "observation-validity"):
                rasterize_annotations("A", labels, root / "out", root / "parts",
                                      root / "scores", validity_dir=root / "scores",
                                      expected_crs="EPSG:32643")

    def test_partition_builder_requires_validity_raster(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            score_path = root / "change_score_A.tif"
            with rasterio.open(score_path, "w", driver="GTiff", height=8, width=8,
                               count=1, dtype="float32", crs=CRS_UTM,
                               transform=_grid().transform, nodata=-9999.0) as ds:
                ds.write(np.ones((8, 8), np.float32), 1)
            manifest = {
                "products": [
                    {"id": "p1", "path": "/nonexistent/p1.SAFE"},
                    {"id": "p2", "path": "/nonexistent/p2.SAFE"},
                ],
                "pairs": {"A": {"tile": "T43QBB", "crs": "EPSG:32643",
                                "date1": "p1", "date2": "p2"}},
            }
            with self.assertRaisesRegex(FileNotFoundError, "observation-validity"):
                build_pair("A", manifest, score_path, root / "out", 500.0, 0.7, 0.15,
                           validity_path=root / "observation_valid_A.tif")

    def test_partition_builder_rejects_wrong_crs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            score_path = root / "change_score_A.tif"
            wrong = Grid(CRS.from_epsg(32644), _grid().transform, 8, 8)
            with rasterio.open(score_path, "w", driver="GTiff", height=8, width=8,
                               count=1, dtype="float32", crs=wrong.crs,
                               transform=wrong.transform, nodata=-9999.0) as ds:
                ds.write(np.ones((8, 8), np.float32), 1)
            manifest = {
                "products": [
                    {"id": "p1", "path": "/nonexistent/p1.SAFE"},
                    {"id": "p2", "path": "/nonexistent/p2.SAFE"},
                ],
                "pairs": {"A": {"tile": "T43QBB", "crs": "EPSG:32643",
                                "date1": "p1", "date2": "p2"}},
            }
            with self.assertRaisesRegex(ValueError, "expected EPSG:32643"):
                build_pair("A", manifest, score_path, root / "out", 500.0, 0.7, 0.15,
                           validity_path=root / "observation_valid_A.tif")


if __name__ == "__main__":
    unittest.main()
