"""Tests for the pair filter on the local patch index builder.

The filter exists because Pair B annotations lag Pair A. Without it, indexing
Pair A alone raises on the missing Pair B raster, and the alternative, a
builder that silently skips missing pairs, would hide a real problem in a
production run.
"""

import unittest

from terrashift.data.local_patches import select_pairs


class SelectPairsTests(unittest.TestCase):
    def test_default_is_every_pair_sorted(self):
        self.assertEqual(select_pairs(["B", "A"]), ["A", "B"])

    def test_requested_subset_is_honoured(self):
        self.assertEqual(select_pairs(["A", "B"], ["A"]), ["A"])

    def test_requested_subset_keeps_the_caller_order(self):
        self.assertEqual(select_pairs(["A", "B"], ["B", "A"]), ["B", "A"])

    def test_empty_request_indexes_nothing(self):
        self.assertEqual(select_pairs(["A", "B"], []), [])

    def test_unknown_pair_raises_and_names_it(self):
        with self.assertRaisesRegex(ValueError, "pairs not in the manifest: Z"):
            select_pairs(["A", "B"], ["A", "Z"])

    def test_unknown_pair_raises_even_when_the_rest_are_valid(self):
        with self.assertRaises(ValueError):
            select_pairs(["A"], ["A", "B"])

    def test_input_is_not_mutated(self):
        available = ["B", "A"]
        select_pairs(available, None)
        self.assertEqual(available, ["B", "A"])


if __name__ == "__main__":
    unittest.main()
