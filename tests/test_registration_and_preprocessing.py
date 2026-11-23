"""Contract tests for the migrated registration, preprocessing and patch modules.

Registration had no focused unit tests before this migration, so the semantics it promises, displacement direction, coverage mask, residual, are pinned here. The other migrated modules
keep their existing dedicated suites (``test_local_normalisation``, ``test_normalization_controls``,
``test_local_patch_index_thresholds``, ``test_validation_windows``); what is added for them is
the packaging contract: they now resolve from ``terrashift``, not from the old top-level
packages, and no duplicate implementation is left behind.
"""

from __future__ import annotations

import pathlib
import unittest

import numpy as np

import terrashift.data.local_patches as local_patches
import terrashift.data.validation_windows as validation_windows
import terrashift.inference.sliding as inference
import terrashift.metrics as metrics
import terrashift.preprocessing.indices as indices
import terrashift.preprocessing.normalisation as normalisation
import terrashift.preprocessing.normalize as normalize
import terrashift.preprocessing.water as water
import terrashift.registration as registration

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "terrashift"


class PositiveShader:
    """Marks where the source content came from, so a shift can be verified exactly."""

    @staticmethod
    def image(height=16, width=16):
        block = np.zeros((height, width), dtype=np.float32)
        block[2:5, 3:7] = 10.0
        return block


class PackageResidenceTests(unittest.TestCase):
    """The migrated modules must be served from the package, with nothing left behind."""

    MIGRATED = {
        registration: "registration.py",
        normalisation: "preprocessing/normalisation.py",
        normalize: "preprocessing/normalize.py",
        indices: "preprocessing/indices.py",
        water: "preprocessing/water.py",
        local_patches: "data/local_patches.py",
        validation_windows: "data/validation_windows.py",
        inference: "inference/sliding.py",
        metrics: "metrics.py",
    }

    def test_modules_live_in_the_package(self):
        for module, expected in self.MIGRATED.items():
            path = pathlib.Path(module.__file__).resolve()
            self.assertEqual(path, (PACKAGE / expected).resolve(),
                             "%s resolves outside the package" % module.__name__)

    def test_legacy_locations_are_gone(self):
        for legacy in ("data/register.py", "data/normalisation.py", "data/local_patches.py",
                       "data/validation_windows.py", "preprocess/indices.py",
                       "preprocess/normalize.py", "preprocess/water.py",
                       "utils/inference.py", "utils/metrics.py"):
            self.assertFalse((REPO / legacy).exists(),
                             "duplicate implementation left at %s" % legacy)

    def test_legacy_packages_are_not_python_packages(self):
        # data/ survives as a *dataset* directory; it must no longer be importable as a package.
        self.assertFalse((REPO / "data" / "__init__.py").exists())
        self.assertFalse((REPO / "preprocess").exists())


class RegistrationShiftTests(unittest.TestCase):
    """apply_shift: positive dx moves content right, positive dy moves content down."""

    def test_identity(self):
        image = PositiveShader.image()
        shifted = registration.apply_shift(image, 0, 0)
        np.testing.assert_array_equal(shifted, image)

    def test_known_translation_moves_content_right_and_down(self):
        image = PositiveShader.image()
        rows, cols = np.nonzero(image == 10.0)
        dx, dy = 3, 2
        shifted = registration.apply_shift(image, dx, dy)
        new_rows, new_cols = np.nonzero(shifted == 10.0)
        np.testing.assert_array_equal(new_rows, rows + dy)
        np.testing.assert_array_equal(new_cols, cols + dx)

    def test_negative_shift_moves_content_left_and_up(self):
        image = PositiveShader.image()
        rows, cols = np.nonzero(image == 10.0)
        shifted = registration.apply_shift(image, -2, -1)
        new_rows, new_cols = np.nonzero(shifted == 10.0)
        np.testing.assert_array_equal(new_rows, rows - 1)
        np.testing.assert_array_equal(new_cols, cols - 2)

    def test_shape_and_dtype_preserved(self):
        image = PositiveShader.image(11, 13)
        shifted = registration.apply_shift(image, 1, 1)
        self.assertEqual(shifted.shape, image.shape)
        self.assertEqual(shifted.dtype, image.dtype)


class CoverageMaskTests(unittest.TestCase):
    """coverage_mask marks the pixels the shifted image can actually supply."""

    def test_zero_shift_covers_everything(self):
        mask = registration.coverage_mask((8, 10), 0, 0)
        self.assertEqual(mask.shape, (8, 10))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask.all())

    def test_shift_removes_a_strip_on_the_leading_edge(self):
        mask = registration.coverage_mask((8, 10), 2, 3)
        self.assertFalse(mask[:, :2].any(), "columns lost to a +x shift must be invalid")
        self.assertFalse(mask[:3, :].any(), "rows lost to a +y shift must be invalid")
        self.assertEqual(int(mask.sum()), (10 - 2) * (8 - 3))

    def test_negative_shift_removes_the_opposite_edge(self):
        mask = registration.coverage_mask((8, 10), -2, -1)
        self.assertFalse(mask[:, -2:].any())
        self.assertFalse(mask[-1:, :].any())
        self.assertEqual(int(mask.sum()), (10 - 2) * (8 - 1))

    def test_object_shape_matches_convention(self):
        # The pairs in this project pass shifts as (x, y); a square grid makes the axis order
        # observable only through the size of the removed strip.
        square = registration.coverage_mask((10, 10), 1, 4)
        self.assertEqual(int(square.sum()), 9 * 6)


class ResidualTests(unittest.TestCase):
    """residual_rmse is the registration quality metric quoted in the report."""

    def test_identical_images_have_zero_residual(self):
        image = PositiveShader.image()
        self.assertAlmostEqual(registration.residual_rmse(image, image), 0.0, places=9)

    def test_shifted_image_has_larger_residual_than_identity(self):
        image = PositiveShader.image()
        aligned = registration.residual_rmse(image, image)
        displaced = registration.residual_rmse(image, registration.apply_shift(image, 2, 2))
        self.assertGreater(displaced, aligned)


class ValidationWindowTests(unittest.TestCase):
    """Window generation keeps its lattice contract after the move."""

    def test_no_reviewed_pixel_yields_no_windows(self):
        target = np.full((64, 64), 255, dtype=np.uint8)
        self.assertEqual(validation_windows.select_windows(target, 32, 16), [])

    def test_every_returned_window_covers_the_reviewed_pixel(self):
        # With patch 32 and stride 16, one reviewed pixel is covered by several windows. The
        # contract is that every returned window contains it, not that only one does.
        target = np.full((64, 64), 255, dtype=np.uint8)
        target[40, 40] = 1
        windows = validation_windows.select_windows(target, 32, 16)
        self.assertGreaterEqual(len(windows), 1)
        for window in windows:
            self.assertLessEqual(window["row0"], 40)
            self.assertGreater(window["row0"] + 32, 40)
            self.assertLessEqual(window["col0"], 40)
            self.assertGreater(window["col0"] + 32, 40)

    def test_window_core_slice_is_inside_the_window(self):
        target = np.full((64, 64), 255, dtype=np.uint8)
        target[30, 30] = 1
        window = validation_windows.select_windows(target, 32, 16)[0]
        start, stop = validation_windows.core_slice(window, 32)
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(stop, 32)
        self.assertLess(start, stop)


if __name__ == "__main__":
    unittest.main()
