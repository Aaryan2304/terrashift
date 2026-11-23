"""The affine-equivalence identity behind the reflectance transformation.

``R = (DN - offset) / quantification`` is affine, so normalising R with the
correspondingly transformed statistics gives exactly the z-score of DN normalised
with the original statistics. The identity is exact in real arithmetic and exact to
float rounding in float32, and it holds only when *both* the mean and the standard
deviation are transformed by the same affine map. These tests pin that, and pin the
failure mode when they are not.

They do not claim the pipeline does this. The pipeline recomputes local statistics,
which is a different operation on the same data.
"""

from __future__ import annotations

import unittest

import numpy as np

QUANTIFICATION = 10000.0
OFFSET = 1000.0


def to_reflectance(dn):
    return (np.asarray(dn) - OFFSET) / QUANTIFICATION


def equivalent_statistics(mean_dn, std_dn):
    """The only mean/std pairing for which the z-scores agree."""
    return (np.asarray(mean_dn) - OFFSET) / QUANTIFICATION, np.asarray(std_dn) / QUANTIFICATION


def z_from_dn(dn, mean_dn, std_dn):
    return (np.asarray(dn) - mean_dn) / std_dn


def z_from_reflectance(dn, mean_dn, std_dn):
    mean_r, std_r = equivalent_statistics(mean_dn, std_dn)
    return (to_reflectance(dn) - mean_r) / std_r


class EquivalenceTests(unittest.TestCase):
    def test_z_scores_are_identical_on_realistic_dn(self):
        dn = np.array([1001.0, 1500.0, 2482.0, 3500.0, 12000.0, 15001.0])
        mean_dn, std_dn = 1395.5416259765625, 443.1914978027344
        source = z_from_dn(dn, mean_dn, std_dn)
        reflected = z_from_reflectance(dn, mean_dn, std_dn)
        self.assertLess(float(np.abs(source - reflected).max()), 1e-9)
        np.testing.assert_allclose(source, reflected, rtol=0, atol=1e-9)

    def test_identity_holds_in_float32_to_rounding(self):
        rng = np.random.default_rng(0)
        dn = rng.uniform(1, 15000, size=200_000).astype(np.float32)
        mean_dn = np.float32(1996.9359130859375)
        std_dn = np.float32(902.7374267578125)
        source = ((dn - mean_dn) / std_dn).astype(np.float32)
        mean_r = np.float32((float(mean_dn) - OFFSET) / QUANTIFICATION)
        std_r = np.float32(float(std_dn) / QUANTIFICATION)
        reflected = (((dn - np.float32(OFFSET)) / np.float32(QUANTIFICATION)) - mean_r) / std_r
        difference = np.abs(source.astype(np.float64) - reflected.astype(np.float64))
        self.assertLess(float(difference.max()), 1e-3)
        self.assertLess(float(difference.mean()), 1e-5)

    def test_untransformed_offset_breaks_the_equivalence(self):
        # Using a reflectance mean without removing the DN offset is the mistake the
        # identity guards against: the z-scores differ by the offset over the std.
        dn = np.array([1001.0, 2482.0, 3500.0])
        mean_dn, std_dn = 1395.5416259765625, 443.1914978027344
        mean_r, std_r = equivalent_statistics(mean_dn, std_dn)
        wrong_mean_r = mean_dn / QUANTIFICATION          # offset not applied
        wrong = (to_reflectance(dn) - wrong_mean_r) / std_r
        source = z_from_dn(dn, mean_dn, std_dn)
        # source - wrong = OFFSET / std_DN, i.e. a constant shift of 1000/std in
        # normalised units, not zero.
        offset_in_sigma = OFFSET / QUANTIFICATION / std_r
        np.testing.assert_allclose(source - wrong, offset_in_sigma, rtol=1e-9)

    def test_untransformed_std_breaks_the_equivalence(self):
        dn = np.array([1001.0, 2482.0, 3500.0])
        mean_dn, std_dn = 1395.5416259765625, 443.1914978027344
        mean_r, _ = equivalent_statistics(mean_dn, std_dn)
        wrong = (to_reflectance(dn) - mean_r) / std_dn      # std not divided by 10^4
        source = z_from_dn(dn, mean_dn, std_dn)
        # Leaving the std in DN units shrinks every z-score by 10^4.
        np.testing.assert_allclose(source, wrong * QUANTIFICATION, rtol=1e-9)

    def test_local_statistics_are_not_the_equivalent_statistics(self):
        # The pipeline's operation: recompute mean/std from the local pixels. That
        # produces a different z-score from z_source, by design.
        dn = np.array([1100.0, 1200.0, 1300.0, 1400.0])
        local_mean_r = to_reflectance(dn).mean()
        local_std_r = to_reflectance(dn).std()
        mean_dn, std_dn = 1395.5416259765625, 443.1914978027344
        mean_r, std_r = equivalent_statistics(mean_dn, std_dn)
        self.assertNotAlmostEqual(float(local_mean_r), float(mean_r), places=6)
        self.assertNotAlmostEqual(float(local_std_r), float(std_r), places=6)


if __name__ == "__main__":
    unittest.main()
