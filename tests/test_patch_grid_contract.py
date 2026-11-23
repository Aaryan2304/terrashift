"""One canonical patch grid; the copies must be the same function object."""

import unittest

import terrashift.data.local_patches as local_patches
import terrashift.data.oscd_original as oscd_original
import terrashift.data.patch_sampling as patch_sampling


class PatchGridIdentityTests(unittest.TestCase):
    def test_all_three_names_are_one_function(self):
        self.assertIs(local_patches.patch_origins, oscd_original.patch_origins)
        self.assertIs(patch_sampling.patch_origins, oscd_original.patch_origins)


class PatchGridContractTests(unittest.TestCase):
    def test_exact_origin_sequences(self):
        cases = {
            (10, 4, 2): [0, 2, 4, 6],
            (10, 4, 4): [0, 4, 6],
            (11, 4, 2): [0, 2, 4, 6, 7],
            (4, 4, 2): [0],
            (5, 4, 2): [0, 1],
            (10, 4, 1): list(range(0, 7)),
        }
        for args, expected in cases.items():
            self.assertEqual(oscd_original.patch_origins(*args), expected, args)

    def test_scene_smaller_than_patch_yields_no_origin(self):
        self.assertEqual(oscd_original.patch_origins(3, 8, 8), [])

    def test_invalid_arguments_raise(self):
        for args in ((10, 0, 2), (10, 4, 0), (10, 4, -1), (10, 4, 6)):
            with self.assertRaises(ValueError, msg=str(args)):
                oscd_original.patch_origins(*args)

    def test_sequences_are_identical_across_the_supported_domain(self):
        # Every realistic tile/stride combination must give one answer, not three.
        for length in (128, 130, 256, 257, 512, 600, 1000):
            for size in (32, 64, 128):
                for stride in (1, 8, size // 2, size):
                    if length < size:
                        continue
                    self.assertEqual(oscd_original.patch_origins(length, size, stride),
                                     patch_sampling.patch_origins(length, size, stride),
                                     (length, size, stride))


if __name__ == "__main__":
    unittest.main()
