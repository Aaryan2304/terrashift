"""Tests for the calibration shortlist builder.

Synthetic rows only. No candidate layer, no raster and no label is read or
written; the selection rules are checked in isolation.
"""

import unittest

from scripts.review.build_calibration_shortlist import (navigation_order, select,
                                                 size_targets)


def make_row(candidate_id, area, easting, northing, score=10.0, pair="A"):
    return {"candidate_id": candidate_id, "pair": pair, "partition": "train",
            "component_area_px_20m": area, "max_score": score,
            "mean_score": score - 1.0, "touches_guard_band": "False",
            "centroid_easting": easting, "centroid_northing": northing}


class SizeTargetTests(unittest.TestCase):
    def test_targets_span_the_range(self):
        rows = [make_row("A_%d" % i, area, 500000 + i * 1000, 1000000)
                for i, area in enumerate([9, 9, 12, 20, 40, 80, 120])]
        targets = size_targets(rows, 5)
        self.assertEqual(len(targets), 5)
        self.assertAlmostEqual(targets[0], 9.0)
        self.assertAlmostEqual(targets[-1], 120.0)
        self.assertEqual(targets, sorted(targets))

    def test_fewer_candidates_than_targets(self):
        rows = [make_row("A_1", 10, 0, 0), make_row("A_2", 20, 1, 1)]
        self.assertEqual(len(size_targets(rows, 5)), 2)


class SelectTests(unittest.TestCase):
    def pool(self):
        areas = [9, 9, 9, 12, 13, 15, 18, 22, 30, 45, 60, 90, 130]
        return [make_row("A_%02d" % i, area, 500000 + i * 900, 1000000 + i * 700,
                         score=8.0 + i)
                for i, area in enumerate(areas)]

    def test_extremes_are_always_chosen(self):
        rows = self.pool()
        chosen = select(rows, 5)
        areas = sorted(row["component_area_px_20m"] for row in chosen)
        self.assertEqual(areas[0], 9)
        self.assertEqual(areas[-1], 130)

    def test_five_distinct_candidates(self):
        chosen = select(self.pool(), 5)
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len({row["candidate_id"] for row in chosen}), 5)

    def test_size_rank_is_assigned_in_target_order(self):
        chosen = select(self.pool(), 5)
        self.assertEqual([row["area_stratum"] for row in chosen], [1, 2, 3, 4, 5])

    def test_selection_ignores_score(self):
        rows = self.pool()
        baseline = [row["candidate_id"] for row in select(rows, 5)]
        for row in rows:                       # invert every score
            row["max_score"] = 100.0 - row["max_score"]
        self.assertEqual([row["candidate_id"] for row in select(rows, 5)], baseline)

    def test_selection_is_deterministic(self):
        first = [row["candidate_id"] for row in select(self.pool(), 5)]
        second = [row["candidate_id"] for row in select(self.pool(), 5)]
        self.assertEqual(first, second)

    def test_spatial_spread_is_preferred_where_area_allows(self):
        # Two candidates with the same area, far apart; the second pick should be
        # the distant one rather than the neighbour.
        rows = [make_row("A_near", 20, 500000, 1000000),
                make_row("A_far", 20, 540000, 1000000),
                make_row("A_mid", 21, 500100, 1000000)]
        chosen = {row["candidate_id"] for row in select(rows, 2)}
        self.assertIn("A_far", chosen)


class NavigationTests(unittest.TestCase):
    def test_each_step_moves_to_the_nearest_remaining(self):
        rows = [make_row("A_%d" % i, 10, 500000 + i * 1000, 1000000)
                for i in range(5)]
        rows.append(make_row("A_far", 10, 900000, 1000000))
        order = navigation_order(rows)
        self.assertEqual(order[0]["candidate_id"], "A_0")   # westernmost start
        self.assertEqual(order[-1]["candidate_id"], "A_far")  # isolated one last
        self.assertEqual(len(order), len(rows))

    def test_empty_input(self):
        self.assertEqual(navigation_order([]), [])


if __name__ == "__main__":
    unittest.main()
