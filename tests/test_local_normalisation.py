"""Tests for per-pair normalisation: statistics, invalid handling, isolation.

All synthetic. The fake products return arrays whose valid-pixel counts and means are
known by construction, so every assertion is exact.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from terrashift.preprocessing.normalisation import (BandMoments, NormalisationError, PairStatistics,
                                compute_pair_statistics, compute_window_statistics,
                                require_statistics,
                                normalise_reflectance, pair_statistics)

BANDS = ("B02", "B03", "B04", "B08")
HEIGHT, WIDTH = 32, 64
BLOCK = 16


class FakeProduct:
    """Returns one constant plane per band, with a NaN corner standing in for nodata."""

    def __init__(self, offsets, nodata_rows=0):
        self.offsets = offsets
        self.nodata_rows = nodata_rows
        self.reads = []

    def read_band(self, band, window=None):
        (r0, r1), (c0, c1) = window
        values = np.full((r1 - r0, c1 - c0), self.offsets[band], dtype=np.float32)
        if self.nodata_rows:
            rows = min(self.nodata_rows, r1 - r0)
            values[:rows, :] = np.nan
        self.reads.append((band, window))
        return values, None, None


def partition_grid():
    partition = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    partition[:, 0:32] = 1      # train
    partition[:, 32:48] = 2     # validation
    partition[:, 48:64] = 3     # test
    return partition


class StreamingTests(unittest.TestCase):
    def test_streaming_matches_direct_computation(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.15, 0.03, size=100_000)
        moments = BandMoments()
        for chunk in np.array_split(values, 7):
            moments.update(chunk)
        summary = moments.summary()
        self.assertEqual(summary["count"], values.size)
        self.assertAlmostEqual(summary["mean"], float(values.mean()), places=12)
        self.assertAlmostEqual(summary["std"], float(values.std()), places=12)

    def test_merge_of_two_halves_equals_the_whole(self):
        rng = np.random.default_rng(1)
        values = rng.normal(0.1, 0.05, size=4097)
        whole = BandMoments().update(values).summary()
        left = BandMoments().update(values[:2000])
        right = BandMoments().update(values[2000:])
        left.merge(right)
        merged = left.summary()
        self.assertAlmostEqual(merged["mean"], whole["mean"], places=12)
        self.assertAlmostEqual(merged["std"], whole["std"], places=12)
        self.assertEqual(merged["count"], whole["count"])

    def test_stable_with_large_offset_and_small_spread(self):
        # A naive sum-of-squares loses the variance here; Welford does not.
        values = np.full(100_000, 1.0e6) + np.linspace(-0.5, 0.5, 100_000)
        moments = BandMoments()
        for chunk in np.array_split(values, 11):
            moments.update(chunk)
        summary = moments.summary()
        self.assertAlmostEqual(summary["mean"], float(values.mean()), places=6)
        self.assertAlmostEqual(summary["std"], float(values.std()), places=6)

    def test_single_value_has_no_standard_deviation(self):
        with self.assertRaisesRegex(ValueError, "at least two pixels"):
            BandMoments().update([1.0]).summary()


class StatisticsIsolationTests(unittest.TestCase):
    def test_only_train_columns_and_valid_pixels_count(self):
        offsets = {"B02": 0.10, "B03": 0.11, "B04": 0.12, "B08": 0.13}
        date1 = FakeProduct(offsets)
        date2 = FakeProduct(offsets)
        partition = partition_grid()
        statistics, meta = compute_pair_statistics(
            date1, date2, partition, None, BANDS,
            block=BLOCK, stride=BLOCK, max_blocks=1000)
        summary = statistics.summary()
        # The loop gives 2 row blocks x 2 column blocks of 16x16 inside the 32 train
        # columns, over both dates: 4 * 256 * 2 = 2,048 pixels per band.
        expected = 4 * BLOCK * BLOCK * 2
        self.assertEqual(summary["B02"]["count"], expected)
        self.assertAlmostEqual(summary["B02"]["mean"], 0.10, places=6)
        self.assertAlmostEqual(summary["B02"]["std"], 0.0, places=6)
        self.assertEqual(meta["train_columns"], [0, 32])

    def test_validation_and_test_columns_are_never_read(self):
        offsets = {band: 0.1 for band in BANDS}
        product = FakeProduct(offsets)
        partition = partition_grid()
        compute_pair_statistics(product, product, partition, None, BANDS,
                                block=BLOCK, stride=BLOCK, max_blocks=1000)
        for _, ((r0, r1), (c0, c1)) in product.reads:
            with self.subTest(window=(r0, r1, c0, c1)):
                self.assertGreaterEqual(c0, 0)
                self.assertLessEqual(c1, 32, "a validation or test column was read")

    def test_nodata_is_excluded(self):
        offsets = {band: 0.1 for band in BANDS}
        product = FakeProduct(offsets, nodata_rows=4)
        partition = partition_grid()
        statistics, meta = compute_pair_statistics(
            product, product, partition, None, BANDS,
            block=BLOCK, stride=BLOCK, max_blocks=1000)
        expected = 4 * (BLOCK - 4) * BLOCK * 2
        self.assertEqual(statistics.summary()["B02"]["count"], expected)
        self.assertAlmostEqual(statistics.summary()["B02"]["mean"], 0.1, places=6)
        self.assertGreater(meta["rejected"]["nodata"], 0)

    def test_validity_mask_removes_pixels(self):
        offsets = {band: 0.1 for band in BANDS}
        product = FakeProduct(offsets)
        partition = partition_grid()
        validity = np.zeros((HEIGHT, WIDTH), dtype=bool)
        validity[:, 0:16] = True     # half the train columns are valid
        statistics, _ = compute_pair_statistics(
            product, product, partition, validity, BANDS,
            block=BLOCK, stride=BLOCK, max_blocks=1000)
        # Only the column block at 0..16 survives; the one at 16..32 is invalid.
        self.assertEqual(statistics.summary()["B02"]["count"], 2 * BLOCK * BLOCK * 2)

    def test_both_dates_use_one_pooled_statistic(self):
        offsets = {"B02": 0.10, "B03": 0.10, "B04": 0.10, "B08": 0.10}
        date1 = FakeProduct(offsets)
        date2 = FakeProduct({band: 0.20 for band in BANDS})
        partition = partition_grid()
        statistics, _ = compute_pair_statistics(
            date1, date2, partition, None, BANDS,
            block=BLOCK, stride=BLOCK, max_blocks=1000)
        summary = statistics.summary()["B02"]
        self.assertAlmostEqual(summary["mean"], 0.15, places=6)
        self.assertAlmostEqual(summary["std"], 0.05, places=6)
        self.assertEqual(summary["count"], 4 * BLOCK * BLOCK * 2)


class NormalisationTests(unittest.TestCase):
    def test_valid_pixels_are_standardised(self):
        image = np.array([[[0.2, 0.1]]], dtype=np.float32)      # (1, 1, 2)
        out = normalise_reflectance(image, [0.15], [0.05])
        np.testing.assert_allclose(out, [[[1.0, -1.0]]], rtol=1e-6)

    def test_invalid_pixels_become_zero_not_negative_sigma(self):
        image = np.array([[[0.2, np.nan]]], dtype=np.float32)
        out = normalise_reflectance(image, [0.15], [0.05])
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0, places=6)
        self.assertEqual(float(out[0, 0, 1]), 0.0)

    def test_infinite_values_are_treated_as_invalid(self):
        image = np.array([[[0.2, np.inf, -np.inf]]], dtype=np.float32)
        out = normalise_reflectance(image, [0.15], [0.05])
        self.assertEqual(float(out[0, 0, 1]), 0.0)
        self.assertEqual(float(out[0, 0, 2]), 0.0)

    def test_all_outputs_finite(self):
        rng = np.random.default_rng(3)
        image = rng.normal(0.15, 0.02, size=(4, 32, 32)).astype(np.float32)
        image[0, :4, :4] = np.nan
        mean = [0.13, 0.14, 0.15, 0.16]
        std = [0.02, 0.02, 0.03, 0.04]
        out = normalise_reflectance(image, mean, std)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out[0, :4, :4] == 0.0))

    def test_per_pair_statistics_differ_and_are_selected_by_pair(self):
        payload = {"pairs": {
            "A": {band: {"mean": 0.10 + i * 0.01, "std": 0.02} for i, band in enumerate(BANDS)},
            "B": {band: {"mean": 0.20 + i * 0.01, "std": 0.05} for i, band in enumerate(BANDS)},
        }}
        mean_a, std_a = pair_statistics(payload, "A")
        mean_b, std_b = pair_statistics(payload, "B")
        self.assertAlmostEqual(float(mean_a[0, 0, 0]), 0.10, places=6)
        self.assertAlmostEqual(float(mean_b[0, 0, 0]), 0.20, places=6)
        self.assertAlmostEqual(float(std_a[0, 0, 0]), 0.02, places=6)
        self.assertAlmostEqual(float(std_b[0, 0, 0]), 0.05, places=6)

    def test_unusable_statistics_are_rejected(self):
        payload = {"pairs": {"A": {band: {"mean": 0.1, "std": 0.0} for band in BANDS}}}
        with self.assertRaisesRegex(ValueError, "unusable statistics"):
            pair_statistics(payload, "A")


class UniquePixelTests(unittest.TestCase):
    """Overlapping windows must not inflate the statistics."""

    def _rows(self, origins, size=8):
        return [{"pair": "A", "row_off": r, "col_off": c, "height": size}
                for r, c in origins]

    def _loader(self, value=0.2):
        def loader(pair, row_off, col_off, size):
            return (np.full((4, size, size), value, dtype=np.float32),
                    np.full((4, size, size), value, dtype=np.float32))
        return loader

    def test_overlapping_windows_count_unique_coordinates_only(self):
        # Four windows (2x2 origins) at stride 4, size 8, over a 12x12 area:
        # 256 pixel instances, but only 144 unique coordinates.
        origins = [(r, c) for r in (0, 4) for c in (0, 4)]
        statistics, counts = compute_window_statistics(
            self._rows(origins), self._loader(), {"A": None}, BANDS,
            shapes={"A": (12, 12)})
        self.assertEqual(counts["instances"], 4 * 64)
        self.assertEqual(counts["unique"], 12 * 12)
        self.assertEqual(counts["repeated"], 4 * 64 - 12 * 12)
        # Both dates pooled, each unique coordinate once per date.
        self.assertEqual(statistics.summary()["B02"]["count"], 2 * 12 * 12)
        self.assertAlmostEqual(statistics.summary()["B02"]["mean"], 0.2, places=6)

    def test_duplicate_weighting_would_shift_the_mean(self):
        # A window covering a bright island must not dominate just because it
        # overlaps its neighbours.
        origins = [(0, 0), (0, 4), (4, 0), (4, 4)]

        def loader(pair, row_off, col_off, size):
            base = np.full((4, size, size), 0.1, dtype=np.float32)
            if (row_off, col_off) == (0, 0):
                base[:, :2, :2] = 0.9        # bright corner, inside one window only
            return base, base.copy()

        statistics, counts = compute_window_statistics(
            self._rows(origins), loader, {"A": None}, BANDS, shapes={"A": (12, 12)})
        # 4 bright pixels out of 144, per date: (4*0.9 + 140*0.1) / 144
        expected = (4 * 0.9 + 140 * 0.1) / 144
        self.assertAlmostEqual(statistics.summary()["B02"]["mean"], expected, places=6)

    def test_without_dedupe_the_count_would_be_the_instance_count(self):
        # Guards against a regression to the previous implementation, which looped
        # over windows and counted every instance.
        origins = [(0, 0), (0, 4)]
        statistics, counts = compute_window_statistics(
            self._rows(origins), self._loader(), {"A": None}, BANDS,
            shapes={"A": (8, 12)})
        self.assertLess(statistics.summary()["B02"]["count"], counts["instances"] * 2)


class FailFastTests(unittest.TestCase):
    """A dataset that declares per-pair normalisation must not run without it."""

    def setUp(self):
        import json
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="normfail_")
        self.path = os.path.join(self.tmp, "normalisation.json")
        self.payload = {
            "scheme": "per-pair per-band z-score, pooled over both dates",
            "band_order": list(BANDS),
            "pairs": {pair: {band: {"mean": 0.1, "std": 0.02, "count": 1000}
                             for band in BANDS} for pair in ("A", "B")},
        }
        with open(self.path, "w") as handle:
            json.dump(self.payload, handle)

    def test_valid_file_is_accepted_and_hashed(self):
        payload = require_statistics(self.path, BANDS, ("A", "B"))
        self.assertEqual(len(payload["_sha256"]), 64)
        self.assertTrue(payload["_file"].endswith("normalisation.json"))

    def test_missing_file_raises(self):
        with self.assertRaisesRegex(NormalisationError, "does not exist"):
            require_statistics(os.path.join(self.tmp, "absent.json"), BANDS, ("A",))

    def test_malformed_json_raises(self):
        open(self.path, "w").write("{not json")
        with self.assertRaisesRegex(NormalisationError, "not readable JSON"):
            require_statistics(self.path, BANDS, ("A",))

    def test_missing_pair_raises(self):
        del self.payload["pairs"]["B"]
        import json
        json.dump(self.payload, open(self.path, "w"))
        with self.assertRaisesRegex(NormalisationError, "no statistics for pair\(s\) B"):
            require_statistics(self.path, BANDS, ("A", "B"))

    def test_band_order_mismatch_raises(self):
        self.payload["band_order"] = ["B08", "B04", "B03", "B02"]
        import json
        json.dump(self.payload, open(self.path, "w"))
        with self.assertRaisesRegex(NormalisationError, "do not match the model bands"):
            require_statistics(self.path, BANDS, ("A",))

    def test_incomplete_band_set_raises(self):
        del self.payload["pairs"]["A"]["B08"]
        import json
        json.dump(self.payload, open(self.path, "w"))
        with self.assertRaisesRegex(NormalisationError, "expected \['B02'"):
            require_statistics(self.path, BANDS, ("A",))

    def test_zero_std_raises(self):
        self.payload["pairs"]["A"]["B02"]["std"] = 0.0
        import json
        json.dump(self.payload, open(self.path, "w"))
        with self.assertRaisesRegex(NormalisationError, "unusable"):
            require_statistics(self.path, BANDS, ("A",))

    def test_wrong_scheme_raises(self):
        self.payload["scheme"] = "single set of DN statistics"
        import json
        json.dump(self.payload, open(self.path, "w"))
        with self.assertRaisesRegex(NormalisationError, "not the expected"):
            require_statistics(self.path, BANDS, ("A",))

    def test_trainer_refuses_to_start_without_the_file(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        self.assertIn("declares_per_pair", source)
        self.assertIn("refusing to fall back", source)
        self.assertIn("SystemExit", source)

    def test_inference_refuses_without_the_file(self):
        import inspect
        import scripts.inference.predict_safe_pair as predict

        source = inspect.getsource(predict)
        self.assertIn("was trained with per-pair normalisation", source)


class DatasetIntegrationTests(unittest.TestCase):
    def test_validation_dataset_zeroes_invalid_pixels_after_normalisation(self):
        import torch
        from terrashift.data.validation_windows import IGNORE_VALUE, ValidationLatticeDataset

        target = np.full((16, 16), IGNORE_VALUE, dtype=np.uint8)
        target[4:12, 4:12] = 1

        def loader(row_off, col_off, size):
            image1 = np.full((4, size, size), 0.20, dtype=np.float32)
            image1[:, :2, :2] = np.nan        # nodata corner in the first date
            image2 = np.full((4, size, size), 0.20, dtype=np.float32)
            return image1, image2

        payload = {"pairs": {"X": {band: {"mean": 0.15, "std": 0.05} for band in BANDS}}}
        dataset = ValidationLatticeDataset(
            ["X"], target, 0, 16, 16, 8, image_loader=loader, normalisation=payload,
            block=16, halo=8)
        sample = dataset[0]
        image1 = sample["image1"].numpy()
        self.assertTrue(np.all(np.isfinite(image1)))
        self.assertTrue(np.all(image1[:, :2, :2] == 0.0))
        self.assertTrue(np.allclose(image1[:, 4:6, 4:6], 1.0, atol=1e-6))
        self.assertIsInstance(sample["mask"], torch.Tensor)
        self.assertIsInstance(sample["valid_mask"], torch.Tensor)


class PathConsistencyTests(unittest.TestCase):
    """Training, validation and inference must consume the same statistics."""

    def test_trainer_passes_normalisation_to_both_datasets(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        # Exactly two consumers: the training set and the validation lattice. The
        # raw dataset used for statistics must NOT be normalised.
        self.assertEqual(source.count("normalisation=normalisation"), 2)
        self.assertIn("require_statistics", source)

    def test_inference_can_use_the_per_pair_file(self):
        import inspect
        import scripts.inference.predict_safe_pair as predict

        source = inspect.getsource(predict)
        self.assertIn("--normalisation", source)
        self.assertIn("pair_statistics", source)

    def test_raw_statistics_dataset_is_not_normalised(self):
        import inspect
        import train_local_manual

        source = inspect.getsource(train_local_manual)
        raw_block = source.split("train_raw = ")[1].split("statistics =")[0]
        self.assertNotIn("normalisation=normalisation", raw_block)


if __name__ == "__main__":
    unittest.main()
