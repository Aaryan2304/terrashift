"""Tests for the receptive-field measurement helpers, on synthetic tensors.

Every expected value here is computed by hand from small arrays. The point is that
the measurement cannot silently change shape semantics again: an earlier version
summed ``axis=0`` on a ``(1, C, H, W)`` tensor, which reduced over the batch axis and
left a ``(C, H, W)`` volume, and every reported radius came out capped at the channel
count. ``channel_aggregates`` now refuses that shape.
"""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluation.measure_energy_radius import (MEASURES, channel_aggregates,
                                           cumulative_radius, phase_positions)


class ShapeContractTests(unittest.TestCase):
    """The batch dimension must never be reduced over."""

    def test_returns_one_plane_per_measure(self):
        grad = np.ones((1, 4, 6, 7), dtype=np.float32)
        measures = channel_aggregates(grad, grad)
        self.assertEqual(set(measures), set(MEASURES))
        for name, plane in measures.items():
            with self.subTest(measure=name):
                self.assertEqual(plane.shape, (6, 7))

    def test_channel_volume_is_rejected(self):
        # A (C, H, W) array is what the batch-axis bug produced. It must not be
        # accepted, because it would silently cap every radius at C.
        with self.assertRaisesRegex(ValueError, r"\(batch, channels, H, W\)"):
            channel_aggregates(np.ones((4, 6, 7)), np.ones((4, 6, 7)))

    def test_batch_greater_than_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch must be 1"):
            channel_aggregates(np.ones((2, 4, 6, 7)), np.ones((2, 4, 6, 7)))

    def test_mismatched_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same gradient shape"):
            channel_aggregates(np.ones((1, 4, 6, 7)), np.ones((1, 4, 7, 6)))


class AggregationTests(unittest.TestCase):
    """Exact values from arrays whose sums are known by construction."""

    def test_l1_sums_absolute_values_over_channels(self):
        grad1 = np.zeros((1, 4, 2, 2), dtype=np.float32)
        grad1[0, :, 0, 0] = [1.0, -2.0, 3.0, -4.0]     # |sum of abs| = 10
        grad1[0, :, 0, 1] = [0.5, 0.5, 0.5, 0.5]       # 2
        measures = channel_aggregates(grad1, np.zeros_like(grad1))
        self.assertAlmostEqual(float(measures["l1_date1"][0, 0]), 10.0, places=6)
        self.assertAlmostEqual(float(measures["l1_date1"][0, 1]), 2.0, places=6)
        self.assertAlmostEqual(float(measures["l1_date1"][1, 0]), 0.0, places=6)

    def test_squared_gradient_sums_squares_over_channels(self):
        grad1 = np.zeros((1, 4, 1, 1), dtype=np.float32)
        grad1[0, :, 0, 0] = [1.0, -2.0, 2.0, 0.0]      # 1 + 4 + 4 + 0 = 9
        measures = channel_aggregates(grad1, np.zeros_like(grad1))
        self.assertAlmostEqual(float(measures["l2sq_date1"][0, 0]), 9.0, places=6)
        # The two aggregations differ: L1 gives 5, L2² gives 9.
        self.assertAlmostEqual(float(measures["l1_date1"][0, 0]), 5.0, places=6)

    def test_dates_are_summed_for_the_combined_measure(self):
        grad1 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        grad2 = np.zeros((1, 1, 1, 1), dtype=np.float32)
        grad1[0, 0, 0, 0] = 3.0
        grad2[0, 0, 0, 0] = -4.0
        measures = channel_aggregates(grad1, grad2)
        self.assertAlmostEqual(float(measures["l1_date1"][0, 0]), 3.0, places=6)
        self.assertAlmostEqual(float(measures["l1_date2"][0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(measures["l1_both"][0, 0]), 7.0, places=6)
        self.assertAlmostEqual(float(measures["l2sq_both"][0, 0]), 25.0, places=6)

    def test_no_reduction_over_the_image(self):
        # Every measure must keep the full spatial extent, and the two aggregations
        # must stay distinguishable: with gradient magnitude 2 everywhere, L1 sums to
        # 2 per channel-pixel and L2² to 4.
        grad = np.full((1, 3, 5, 9), 2.0, dtype=np.float32)
        measures = channel_aggregates(grad, grad)
        for name, plane in measures.items():
            with self.subTest(measure=name):
                self.assertEqual(plane.shape, (5, 9))
        pixels = 3 * 5 * 9
        self.assertAlmostEqual(float(measures["l1_date1"].sum()), 2.0 * pixels, places=3)
        self.assertAlmostEqual(float(measures["l2sq_date1"].sum()), 4.0 * pixels, places=3)


class CumulativeRadiusTests(unittest.TestCase):
    """Chebyshev radius, not Euclidean, and cumulative rather than per-shell."""

    def test_single_pixel_mass_gives_radius_zero(self):
        mass = np.zeros((11, 11), dtype=np.float64)
        mass[5, 5] = 1.0
        for fraction in (0.5, 0.9, 0.99):
            self.assertEqual(cumulative_radius(mass, (5, 5), fraction), 0)

    def test_uniform_square_support(self):
        # 3x3 block centred at (5,5): radius 1 covers all 9 pixels, so 50 % is
        # reached at radius 1.
        mass = np.zeros((11, 11), dtype=np.float64)
        mass[4:7, 4:7] = 1.0
        self.assertEqual(cumulative_radius(mass, (5, 5), 0.5), 1)
        self.assertEqual(cumulative_radius(mass, (5, 5), 0.99), 1)

        # 5x5 block: radius 1 holds only 9 of 25 pixels (36 %), so 50 % needs radius 2.
        mass = np.zeros((11, 11), dtype=np.float64)
        mass[3:8, 3:8] = 1.0
        self.assertEqual(cumulative_radius(mass, (5, 5), 0.5), 2)
        self.assertEqual(cumulative_radius(mass, (5, 5), 0.99), 2)

    def test_chebyshev_not_euclidean(self):
        # Mass at (5,5) and at (8,8). Euclidean distance is 4.24, Chebyshev is 3.
        # A Chebyshev radius of 3 reaches it; a Euclidean radius would need 5.
        mass = np.zeros((11, 11), dtype=np.float64)
        mass[5, 5] = 1.0
        mass[8, 8] = 1.0
        self.assertEqual(cumulative_radius(mass, (5, 5), 1.0), 3)

    def test_cumulative_not_shell(self):
        # Mass ring at radius 2 only. A per-shell reading would never reach 50 %; the
        # cumulative reading is satisfied at radius 2 because everything inside
        # radius 2 (including the ring) is counted.
        mass = np.zeros((9, 9), dtype=np.float64)
        mass[2, 2] = 1.0
        mass[2, 6] = 1.0
        mass[6, 2] = 1.0
        mass[6, 6] = 1.0
        self.assertEqual(cumulative_radius(mass, (4, 4), 0.5), 2)

    def test_zero_mass_returns_none(self):
        self.assertIsNone(cumulative_radius(np.zeros((5, 5)), (2, 2), 0.9))


class PhaseTests(unittest.TestCase):
    def test_all_64_combinations_are_present(self):
        positions = phase_positions()
        self.assertEqual(len(positions), 64)
        self.assertEqual({p[0] for p in positions}, set(range(8)))
        self.assertEqual({p[1] for p in positions}, set(range(8)))
        self.assertEqual(len(set((p[0], p[1]) for p in positions)), 64)

    def test_every_position_carries_its_own_phase(self):
        for row_phase, col_phase, row, col in phase_positions():
            with self.subTest(phase=(row_phase, col_phase)):
                self.assertEqual(row % 8, row_phase)
                self.assertEqual(col % 8, col_phase)
                # Interior, so the full support fits inside a 128 px patch.
                self.assertTrue(51 <= row <= 79)
                self.assertTrue(51 <= col <= 79)


if __name__ == "__main__":
    unittest.main()
