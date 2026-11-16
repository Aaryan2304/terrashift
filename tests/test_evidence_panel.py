"""Tests for the evidence-panel builder, on synthetic fixtures only.

No real imagery and no real register row is used: there are no reviewed
polygons yet, so a panel built from the depot would be a fabricated detection.
"""

import os
import tempfile
import unittest

import numpy as np
from rasterio.transform import from_origin

from scripts.make_evidence_panel import (_window_bounds, polygon_rings,
                                         read_register_row, render_panel,
                                         window_for_bounds, world_to_pixel)

TRANSFORM = from_origin(500000.0, 1000000.0, 20.0, 20.0)   # origin top left


class WorldToPixelTests(unittest.TestCase):
    def test_origin_maps_to_the_top_left_corner(self):
        self.assertEqual(world_to_pixel(TRANSFORM, 500000.0, 1000000.0), (0.0, 0.0))

    def test_one_pixel_shift(self):
        self.assertEqual(world_to_pixel(TRANSFORM, 500020.0, 999980.0), (1.0, 1.0))


class WindowForBoundsTests(unittest.TestCase):
    def test_window_covers_the_box_and_adds_the_margin(self):
        bounds = (500100.0, 999800.0, 500200.0, 999900.0)
        row0, col0, row1, col1 = window_for_bounds(
            TRANSFORM, bounds, (200, 200), margin_px=5, minimum_px=1)
        self.assertEqual((row0, col0), (0, 0))          # margin clipped at the edge
        self.assertEqual((row1, col1), (15, 15))        # 10 px box plus 5 px margin

    def test_margin_is_applied_on_every_side_when_space_allows(self):
        # Rows count down from the top-left origin, columns count right, so the
        # same y range gives different row indices than the x range gives columns.
        bounds = (501000.0, 999000.0, 501100.0, 999100.0)   # cols 50..55, rows 45..50
        row0, col0, row1, col1 = window_for_bounds(
            TRANSFORM, bounds, (200, 200), margin_px=4, minimum_px=1)
        self.assertEqual((row0, col0, row1, col1), (41, 46, 54, 59))

    def test_box_outside_the_raster_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the raster"):
            window_for_bounds(TRANSFORM, (600000.0, 900000.0, 600100.0, 900100.0),
                              (200, 200))

    def test_tiny_box_is_rejected_rather_than_rendered(self):
        bounds = (500100.0, 999900.0, 500120.0, 999980.0)   # 1 x 4 px
        with self.assertRaisesRegex(ValueError, "below the 32 px minimum"):
            window_for_bounds(TRANSFORM, bounds, (200, 200), margin_px=0)

    def test_empty_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "bounds are empty"):
            window_for_bounds(TRANSFORM, (500100.0, 999900.0, 500100.0, 999900.0),
                              (200, 200))

    def test_window_bounds_round_trip(self):
        rows, cols = (4, 10), (6, 12)
        left, bottom, right, top = _window_bounds(TRANSFORM, rows, cols)
        self.assertEqual((left, top), (500000.0 + 6 * 20.0, 1000000.0 - 4 * 20.0))
        self.assertEqual((right, bottom), (500000.0 + 12 * 20.0, 1000000.0 - 10 * 20.0))


class PolygonTests(unittest.TestCase):
    def test_ring_and_bounds_come_from_the_wkt(self):
        wkt = ("POLYGON ((500100 999900, 500200 999900, 500200 999800, "
               "500100 999800, 500100 999900))")
        ring, bounds = polygon_rings(wkt)
        self.assertEqual(ring.shape[1], 2)
        self.assertEqual(ring[0].tolist(), [500100.0, 999900.0])
        self.assertEqual(bounds, (500100.0, 999800.0, 500200.0, 999900.0))

    def test_non_polygon_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected a Polygon"):
            polygon_rings("LINESTRING (0 0, 1 1)")

    def test_empty_polygon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            polygon_rings("POLYGON EMPTY")


class RegisterRowTests(unittest.TestCase):
    def test_row_is_found_by_record_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "register.csv")
            with open(path, "w") as handle:
                handle.write("record_id,pair,change_type\n")
                handle.write("REC-0001,A,construction\n")
                handle.write("REC-0002,B,unknown\n")
            self.assertEqual(read_register_row(path, "REC-0002")["pair"], "B")

    def test_missing_record_id_raises(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "register.csv")
            with open(path, "w") as handle:
                handle.write("record_id,pair\nREC-0001,A\n")
            with self.assertRaises(KeyError):
                read_register_row(path, "REC-9999")


class RenderPanelTests(unittest.TestCase):
    def _views(self, value=0.5):
        image = np.full((32, 32, 3), value, dtype=np.float32)
        score = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
        mask = score > 0.9
        return [
            {"image": image, "label": "date 1"},
            {"image": image, "label": "date 2"},
            {"image": score, "label": "score", "cmap": "viridis", "colorbar": True},
            {"image": image, "label": "date 2 with mask", "overlay": mask},
        ]

    def test_panel_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "panel.png")
            render_panel(self._views(), path, "A / REC-0001 / synthetic",
                         "fixture footer", (0.0, 640.0, 0.0, 640.0))
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 10_000)

    def test_wrong_view_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "panel.png")
            with self.assertRaisesRegex(ValueError, "exactly four"):
                render_panel(self._views()[:3], path, "t", "f", (0.0, 1.0, 0.0, 1.0))

    def test_all_invalid_score_does_not_raise(self):
        views = self._views()
        views[2]["image"] = np.full((32, 32), np.nan)
        views[3]["overlay"] = np.zeros((32, 32), dtype=bool)
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "panel.png")
            render_panel(views, path, "t", "f", (0.0, 640.0, 0.0, 640.0))
            self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()
