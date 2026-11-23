"""Tests for area and length uncertainty, on synthetic fixtures only.

Every geometry here is constructed in the test. None of these measurements is a
result from the assignment data, and none should be quoted as one.

The point of the suite is to pin down what the reported intervals mean:
a boundary-sensitivity band is not a confidence interval, and a global
translation changes neither an area nor a length.
"""

from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import (GeometryCollection, LineString, MultiPolygon,
                              Point, Polygon, box)

from terrashift.geospatial.geometry import (boundary_area_range, length_sensitivity, pixel_area,
                            polygon_area_sensitivity, registration_displacement,
                            skeleton_graph, skeleton_length)

TRANSFORM = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 3000000.0)


def _hollow(shell, target):
    """A square ring: outer box minus inner box, as one polygon with a hole.

    ``Polygon(a_polygon, holes)`` silently ignores the hole, so the shell has to
    be passed as a coordinate sequence. That mistake makes a ring look solid and
    every hole test pass for the wrong reason.
    """
    return Polygon(list(shell.exterior.coords), [list(target.exterior.coords)])


class RectangleAreaTests(unittest.TestCase):
    def test_square_of_known_size(self):
        # 100 m x 100 m = 10,000 m2. A 10 m boundary band gives
        # 80 x 80 = 6,400 and 120 x 120 = 14,400 with mitre corners.
        result = polygon_area_sensitivity(box(0.0, 0.0, 100.0, 100.0), 10.0,
                                          join_style="mitre")
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["nominal_m2"], 10000.0, places=6)
        self.assertAlmostEqual(result["min_m2"], 6400.0, places=6)
        self.assertAlmostEqual(result["max_m2"], 14400.0, places=6)
        self.assertAlmostEqual(result["minus_m2"], 3600.0, places=6)
        self.assertAlmostEqual(result["plus_m2"], 4400.0, places=6)
        self.assertFalse(result["disappeared"])

    def test_round_join_narrows_the_dilation_but_not_the_erosion(self):
        # Erosion of a convex polygon is exact whatever the join style: the
        # inner offset of a 100 m square is an 80 m square, 6,400 m2. Dilation
        # differs, because mitre adds sharp corners while round adds quarter
        # circles: 10,000 + 4,000 + pi * 100 = 14,314 m2 against 14,400 m2.
        square = box(0.0, 0.0, 100.0, 100.0)
        rounded = polygon_area_sensitivity(square, 10.0)
        mitred = polygon_area_sensitivity(square, 10.0, join_style="mitre")
        self.assertLess(rounded["max_m2"], mitred["max_m2"])
        self.assertAlmostEqual(rounded["min_m2"], mitred["min_m2"], places=6)
        self.assertAlmostEqual(rounded["max_m2"], 14313.65, delta=1.0)

    def test_zero_displacement_collapses_the_band(self):
        result = polygon_area_sensitivity(box(0.0, 0.0, 50.0, 50.0), 0.0)
        self.assertEqual(result["min_m2"], result["nominal_m2"])
        self.assertEqual(result["max_m2"], result["nominal_m2"])

    def test_negative_displacement_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            polygon_area_sensitivity(box(0.0, 0.0, 10.0, 10.0), -1.0)

    def test_raster_range_agrees_with_the_polygon_for_a_block(self):
        # A 4x4 block at 10 m: 1,600 m2 nominal, erosion and dilation each move
        # the boundary by one whole pixel, which is a 10 m band.
        mask = np.zeros((12, 12), dtype=bool)
        mask[4:8, 4:8] = True
        raster = boundary_area_range(mask, 100.0)
        polygon = polygon_area_sensitivity(box(0.0, 0.0, 40.0, 40.0), 10.0,
                                           join_style="mitre")
        self.assertEqual(raster["nominal_m2"], 1600)
        self.assertAlmostEqual(raster["min_m2"], polygon["min_m2"], places=6)
        self.assertAlmostEqual(raster["max_m2"], polygon["max_m2"], places=6)


class NarrowStructureTests(unittest.TestCase):
    def test_narrow_strip_erodes_to_nothing(self):
        # 4 m wide, boundary uncertain to 10 m: the structure cannot survive.
        strip = box(0.0, 0.0, 400.0, 4.0)
        result = polygon_area_sensitivity(strip, 10.0)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["disappeared"])
        self.assertEqual(result["min_m2"], 0.0)
        self.assertGreater(result["max_m2"], result["nominal_m2"])

    def test_raster_strip_reports_zero_after_erosion(self):
        mask = np.zeros((20, 60), dtype=bool)
        mask[10, 5:55] = True          # one pixel wide
        result = boundary_area_range(mask, 100.0)
        self.assertEqual(result["eroded_px"], 0)
        self.assertEqual(result["min_m2"], 0)

    def test_single_pixel_polygon(self):
        result = polygon_area_sensitivity(box(0.0, 0.0, 10.0, 10.0), 10.0)
        self.assertTrue(result["disappeared"])
        self.assertEqual(result["min_m2"], 0.0)


class PolygonHoleTests(unittest.TestCase):
    def test_hole_is_grown_by_dilation_and_by_erosion_of_the_ring(self):
        ring = _hollow(box(0.0, 0.0, 100.0, 100.0), box(40.0, 40.0, 60.0, 60.0))
        result = polygon_area_sensitivity(ring, 5.0, join_style="mitre")
        self.assertEqual(result["holes"], 1)
        # nominal 10,000 - 400 = 9,600
        self.assertAlmostEqual(result["nominal_m2"], 9600.0, places=6)
        # dilation: outer +5 per side, hole shrinks by 5 per side
        self.assertAlmostEqual(result["max_m2"], 110.0 * 110.0 - 10.0 * 10.0, places=6)
        # erosion: outer -5 per side, hole grows by 5 per side
        self.assertAlmostEqual(result["min_m2"], 90.0 * 90.0 - 30.0 * 30.0, places=6)

    def test_hole_larger_than_the_structure_disappears(self):
        thin_ring = _hollow(box(0.0, 0.0, 100.0, 100.0), box(2.0, 2.0, 98.0, 98.0))
        result = polygon_area_sensitivity(thin_ring, 10.0)
        self.assertTrue(result["disappeared"])


class MultipartTests(unittest.TestCase):
    def test_two_parts_are_summed_and_counted(self):
        parts = MultiPolygon([box(0.0, 0.0, 20.0, 20.0), box(100.0, 0.0, 120.0, 20.0)])
        result = polygon_area_sensitivity(parts, 5.0, join_style="mitre")
        self.assertEqual(result["parts"], 2)
        self.assertAlmostEqual(result["nominal_m2"], 800.0, places=6)
        self.assertGreater(result["max_m2"], result["min_m2"])

    def test_one_part_may_disappear_while_the_other_survives(self):
        parts = MultiPolygon([box(0.0, 0.0, 100.0, 100.0), box(200.0, 0.0, 204.0, 4.0)])
        result = polygon_area_sensitivity(parts, 10.0)
        self.assertEqual(result["parts"], 2)
        self.assertGreater(result["min_m2"], 0.0)     # the large part survives
        self.assertFalse(result["disappeared"])


class InvalidInputTests(unittest.TestCase):
    def test_none_geometry(self):
        result = polygon_area_sensitivity(None, 10.0)
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["nominal_m2"])

    def test_empty_polygon(self):
        result = polygon_area_sensitivity(Polygon(), 10.0)
        self.assertEqual(result["status"], "empty")

    def test_linestring_is_not_areal(self):
        result = polygon_area_sensitivity(LineString([(0, 0), (10, 0)]), 10.0)
        self.assertEqual(result["status"], "not_areal")

    def test_point_is_not_areal(self):
        result = polygon_area_sensitivity(Point(0, 0), 10.0)
        self.assertEqual(result["status"], "not_areal")

    def test_self_intersecting_polygon_is_repaired(self):
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        self.assertFalse(bowtie.is_valid)
        result = polygon_area_sensitivity(bowtie, 1.0)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["nominal_m2"], 0.0)

    def test_geometry_collection_is_not_areal(self):
        collection = GeometryCollection([box(0.0, 0.0, 10.0, 10.0)])
        result = polygon_area_sensitivity(collection, 1.0)
        self.assertEqual(result["status"], "not_areal")


class TranslationInvarianceTests(unittest.TestCase):
    """A global registration shift is not a boundary error for area or length."""

    def test_area_is_unchanged_by_a_shift(self):
        square = box(0.0, 0.0, 100.0, 100.0)
        from shapely.affinity import translate
        self.assertAlmostEqual(polygon_area_sensitivity(square, 10.0)["nominal_m2"],
                               polygon_area_sensitivity(translate(square, 7.0, -3.0),
                                                        10.0)["nominal_m2"], places=9)

    def test_raster_area_and_length_are_unchanged_by_a_whole_pixel_shift(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:14, 10:30] = True
        shifted = np.zeros_like(mask)
        shifted[12:16, 13:33] = mask[10:14, 10:30]
        self.assertEqual(int(mask.sum()), int(shifted.sum()))
        first = skeleton_length(mask, 10.0)["length_m"]
        second = skeleton_length(shifted, 10.0)["length_m"]
        self.assertAlmostEqual(first, second, places=6)


class PixelAreaTests(unittest.TestCase):
    def test_axis_aligned(self):
        self.assertAlmostEqual(pixel_area(Affine(10.0, 0.0, 0.0, 0.0, -10.0, 0.0)), 100.0)

    def test_rotated_transform_uses_the_determinant(self):
        # A rotation keeps the cell area; |a * e| alone would return a wrong value.
        angle = np.radians(30.0)
        a, b = 10.0 * np.cos(angle), 10.0 * np.sin(angle)
        d, e = 10.0 * np.sin(angle), -10.0 * np.cos(angle)
        self.assertAlmostEqual(pixel_area(Affine(a, b, 0.0, d, e, 0.0)), 100.0, places=6)


class DiagonalLengthTests(unittest.TestCase):
    def test_one_pixel_diagonal(self):
        mask = np.zeros((11, 11), dtype=bool)
        for i in range(11):
            mask[i, i] = True
        result = skeleton_graph(mask, 10.0)
        self.assertEqual(result["diagonal_edges"], 10)
        self.assertEqual(result["orthogonal_edges"], 0)
        self.assertAlmostEqual(result["length_m"], 10 * np.sqrt(2.0) * 10.0, places=6)
        self.assertEqual(result["endpoints"], 2)
        self.assertEqual(result["junctions"], 0)
        self.assertEqual(result["components"], 1)

    def test_thick_diagonal_keeps_the_diagonal_length(self):
        from scipy import ndimage as ndi
        line = np.zeros((21, 21), dtype=bool)
        for i in range(21):
            line[i, i] = True
        band = ndi.binary_dilation(line, np.ones((3, 3), dtype=bool))
        measured = skeleton_graph(band, 10.0)["length_m"]
        expected = 20 * np.sqrt(2.0) * 10.0
        self.assertAlmostEqual(measured, expected, delta=0.12 * expected)


class BranchingStructureTests(unittest.TestCase):
    def _cross(self):
        """Plus sign: four arms of 5 px from a centre junction."""
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 0:11] = True
        mask[0:11, 5] = True
        return mask

    def test_junction_edges_are_counted_once(self):
        result = skeleton_graph(self._cross(), 10.0)
        self.assertEqual(result["junctions"], 1)
        self.assertEqual(result["endpoints"], 4)
        # 10 horizontal + 10 vertical edges; no edge counted twice.
        self.assertEqual(result["orthogonal_edges"], 20)
        self.assertAlmostEqual(result["length_m"], 200.0, places=6)

    def test_trunk_with_branch(self):
        mask = np.zeros((11, 21), dtype=bool)
        mask[5, 0:21] = True           # 200 m trunk
        mask[5:11, 10] = True          # 50 m branch upward in row terms
        result = skeleton_graph(mask, 10.0)
        self.assertEqual(result["junctions"], 1)
        self.assertEqual(result["endpoints"], 3)
        self.assertAlmostEqual(result["length_m"], 250.0, places=6)

    def test_isolated_pixel_carries_no_length(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = skeleton_graph(mask, 10.0)
        self.assertEqual(result["length_m"], 0.0)
        self.assertEqual(result["isolated_px"], 1)

    def test_spur_pruning_removes_only_the_short_branch(self):
        mask = np.zeros((11, 21), dtype=bool)
        mask[5, 0:21] = True
        mask[5:8, 10] = True           # 2 px of spur beyond the trunk
        raw = skeleton_graph(mask, 10.0)
        pruned = skeleton_graph(mask, 10.0, spur_length_px=3)
        self.assertGreater(raw["length_m"], pruned["length_m"])
        self.assertAlmostEqual(pruned["length_m"], 200.0, places=6)
        self.assertEqual(pruned["spur_px_removed"], 2)

    def test_pruning_keeps_an_isolated_short_line(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 2:5] = True
        result = skeleton_graph(mask, 10.0, spur_length_px=5)
        self.assertAlmostEqual(result["length_m"], 20.0, places=6)
        self.assertEqual(result["spur_px_removed"], 0)


class LengthSensitivityTests(unittest.TestCase):
    def test_branching_road_envelope_contains_its_nominal(self):
        # Regression: the dilated centreline of a branching road measures
        # slightly shorter than the nominal one, because thinning trims each end
        # by about half the local width. Reporting the dilated value as the upper
        # bound produced an envelope of 0.0 to 1,664.14 around a nominal 1,670.00.
        mask = np.zeros((220, 220), dtype=bool)
        mask[160, 20:120] = True
        mask[160:200, 60] = True
        mask[160:190, 100] = True

        result = length_sensitivity(mask, 10.0)
        self.assertAlmostEqual(result["nominal_m"], 1670.0, places=6)
        self.assertLess(result["dilated_m"], result["nominal_m"])
        self.assertTrue(result["dilated_shorter_than_nominal"])
        self.assertEqual(result["max_m"], result["nominal_m"])
        self.assertEqual(result["min_m"], 0.0)
        self.assertLessEqual(result["min_m"], result["nominal_m"])
        self.assertGreaterEqual(result["max_m"], result["nominal_m"])

    def test_envelope_contains_nominal_for_every_fixture(self):
        fixtures = {}
        fixtures["block"] = np.zeros((40, 40), dtype=bool)
        fixtures["block"][10:20, 10:20] = True

        fixtures["thin_line"] = np.zeros((30, 30), dtype=bool)
        fixtures["thin_line"][15, 5:25] = True

        fixtures["diagonal"] = np.zeros((30, 30), dtype=bool)
        for step in range(20):
            fixtures["diagonal"][5 + step, 5 + step] = True

        fixtures["branching"] = np.zeros((60, 60), dtype=bool)
        fixtures["branching"][30, 5:50] = True
        fixtures["branching"][30:50, 20] = True

        fixtures["single_pixel"] = np.zeros((9, 9), dtype=bool)
        fixtures["single_pixel"][4, 4] = True

        fixtures["ring"] = np.zeros((40, 40), dtype=bool)
        fixtures["ring"][10:30, 10:30] = True
        fixtures["ring"][15:25, 15:25] = False

        for name, mask in fixtures.items():
            with self.subTest(fixture=name):
                result = length_sensitivity(mask, 10.0)
                self.assertLessEqual(result["min_m"], result["nominal_m"] + 1e-9)
                self.assertGreaterEqual(result["max_m"], result["nominal_m"] - 1e-9)
                self.assertLessEqual(result["min_m"], result["max_m"])
                for key in ("nominal_m", "min_m", "max_m", "eroded_m", "dilated_m"):
                    self.assertGreaterEqual(result[key], 0.0)

    def test_raw_eroded_and_dilated_values_are_kept(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[15, 5:25] = True
        result = length_sensitivity(mask, 10.0)
        self.assertEqual(result["eroded_m"], 0.0)
        self.assertGreater(result["dilated_m"], result["nominal_m"])
        # For this fixture the dilated measurement is the upper bound, so the raw
        # value and the envelope edge agree. The branching-road regression covers
        # the case where they do not.
        self.assertEqual(result["max_m"], result["dilated_m"])
        self.assertEqual(result["min_m"], result["eroded_m"])

    def test_a_wide_band_is_more_stable_than_a_one_pixel_road(self):
        # A one-pixel diagonal cannot survive a one-pixel erosion, so its band
        # spans its whole length. A three-row band loses a row per side and keeps
        # its length, so its band is tight.
        thin = np.zeros((21, 21), dtype=bool)
        for i in range(21):
            thin[i, i] = True
        thick = np.zeros((25, 25), dtype=bool)
        thick[11:14, 2:23] = True       # 21 px long, away from the border

        thin_range = length_sensitivity(thin, 10.0)
        thick_range = length_sensitivity(thick, 10.0)
        self.assertTrue(thin_range["disappeared_under_erosion"])
        self.assertEqual(thin_range["min_m"], 0.0)
        self.assertFalse(thick_range["disappeared_under_erosion"])
        self.assertGreater(thick_range["min_m"], 0.0)
        # Thinning trims the ends of a thick band by about half its width, so the
        # centreline is a little shorter than the band. 194 m against 210 m of
        # band length is the expected order, not an error.
        self.assertAlmostEqual(thick_range["nominal_m"], 194.1, delta=10.0)

        thin_spread = (thin_range["nominal_m"] - thin_range["min_m"]) / thin_range["nominal_m"]
        thick_spread = (thick_range["nominal_m"] - thick_range["min_m"]) / thick_range["nominal_m"]
        self.assertGreater(thin_spread, thick_spread)

    def test_empty_mask(self):
        result = length_sensitivity(np.zeros((4, 4), dtype=bool), 10.0)
        self.assertEqual(result["nominal_m"], 0.0)
        self.assertFalse(result["disappeared_under_erosion"])


class RegistrationDisplacementTests(unittest.TestCase):
    PAYLOAD = {
        "A": {"pixel_size_m": 10.0,
              "residual": {"n_converged": 4, "n_failed": 1,
                           "median_displacement_m": 1.2983883,
                           "p95_displacement_m": 1.8025820,
                           "rmse_mean": 0.6523}},
        "B": {"pixel_size_m": 10.0,
              "residual": {"n_converged": 5, "n_failed": 0,
                           "median_displacement_m": 0.5632417,
                           "p95_displacement_m": 0.8697597,
                           "rmse_mean": 1.0120}},
        "C": {"pixel_size_m": 10.0},
    }

    def test_reads_metres_and_converts_to_grid_pixels(self):
        result = registration_displacement(self.PAYLOAD, "A", grid_resolution_m=20.0)
        self.assertAlmostEqual(result["median_m"], 1.2983883, places=6)
        self.assertAlmostEqual(result["median_grid_px"], 1.2983883 / 20.0, places=9)
        self.assertEqual(result["n_failed"], 1)

    def test_derives_from_pixels_when_metres_are_absent(self):
        payload = {"A": {"pixel_size_m": 10.0,
                         "residual": {"median_dx_px": 0.3, "median_dy_px": 0.4}}}
        result = registration_displacement(payload, "A")
        self.assertAlmostEqual(result["median_m"], 5.0, places=6)
        self.assertIsNone(result["p95_m"])

    def test_unconverged_windows_are_listed_and_tile_coverage_is_denied(self):
        payload = {"A": {"pixel_size_m": 10.0,
                         "residual": {"n_converged": 4, "n_failed": 1,
                                      "median_displacement_m": 1.2983883,
                                      "windows": [
                                          {"converged": True, "window": [2000, 4000, 5000, 7000]},
                                          {"converged": False, "window": [7000, 9000, 4000, 6000]}],
                                      }}}
        result = registration_displacement(payload, "A")
        self.assertFalse(result["covers_whole_tile"])
        self.assertEqual(result["unconverged_windows"], [[7000, 9000, 4000, 6000]])

    def test_full_convergence_reports_tile_coverage(self):
        result = registration_displacement(self.PAYLOAD, "B")
        self.assertTrue(result["covers_whole_tile"])
        self.assertEqual(result["unconverged_windows"], [])

    def test_reads_the_nested_residual_written_by_the_masked_registration(self):
        nested = {"A": {"pixel_size_m": 10.0,
                        "initial": {"residual": {"median_displacement_m": 1.2983883,
                                                 "p95_displacement_m": 1.8025820,
                                                 "n_converged": 4, "n_failed": 1}}}}
        result = registration_displacement(nested, "A")
        self.assertAlmostEqual(result["median_m"], 1.2983883, places=6)

    def test_missing_residual_block_is_an_error_and_not_a_zero(self):
        with self.assertRaisesRegex(KeyError, "no residual block"):
            registration_displacement(self.PAYLOAD, "C")

    def test_unknown_pair_raises(self):
        with self.assertRaises(KeyError):
            registration_displacement(self.PAYLOAD, "Z")


if __name__ == "__main__":
    unittest.main()
